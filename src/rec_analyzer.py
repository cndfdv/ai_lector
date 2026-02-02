import datetime
import json
import os
import re
import uuid
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from dotenv import load_dotenv
import torch
import torchaudio
from langchain_openai import ChatOpenAI
from pyannote.audio import Pipeline
from pydub import AudioSegment
from ruaccent import RUAccent
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from F5TTS.f5_tts.api import F5TTS
from src.llm_models import (
    PodcastPart,
    PodcastScript,
    QuestionsResult,
)
from src.prompts import (
    ABSTRACT_PROMPT,
    CLEAN_PODCAST_PROMPT,
    EMOT_AN_PROMPT,
    MINDMAP_PROMPT,
    PODCAST_PROMPT,
    QUESTIONS_PROMPT,
)

STOPWORDS_PATH = "src/stopwords.txt"
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class AudioData:
    """Audio file data after preparation."""

    wav_path: str
    waveform: torch.Tensor
    sample_rate: int


@dataclass
class DiarizationResult:
    """Speaker diarization results."""

    diagram: Dict[str, float]  # {lecturer: %, discussion: %, quiet: %}
    labeled_chunks: List[List]  # [[speaker_id, start, end], ...]


@dataclass
class TranscriptionResult:
    """Speech transcription results."""

    chunks: List[List]  # [[speaker, text, (start, end)], ...]
    lecture_text: str


@dataclass
class TextAnalysis:
    """Text analysis results."""

    popular_words_no_stopw: List[Dict]
    popular_words_w_stopw: List[Dict]
    syllables_per_minute: List[float]
    speed: Dict[int, float]
    transcripted_chunks: List[List]


# =============================================================================
# Utility Functions
# =============================================================================


def clear_gpu_cache() -> None:
    """Clear the GPU cache if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def latex_to_md(text: str) -> str:
    """Convert LaTeX syntax to Markdown math syntax.

    Args:
        text: Text to convert.

    Returns:
        Converted text with Markdown math delimiters.
    """
    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text)
    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)
    return text


def fill_silence_intervals(
    data: List[Tuple[str, float, float]],
) -> List[List]:
    """Fill intervals when no words were spoken.

    Args:
        data: List of [speaker_id, start, end] tuples.

    Returns:
        Data with silence intervals (speaker_id=3) filled in.
    """
    filled_data = []

    if data[0][1] > 0:
        filled_data.append([3, 0, data[0][1]])
    for i, entry in enumerate(data):
        speaker, start, end = entry
        filled_data.append(entry)

        if i < len(data) - 1:
            next_start = data[i + 1][1]
            if end < next_start:
                filled_data.append([3, end, next_start])
    return filled_data


# =============================================================================
# Processing Functions
# =============================================================================


def prepare_audio(recording_path: str) -> AudioData:
    """Prepare audio file for processing.

    Converts mp3 to wav if necessary and loads the waveform.

    Args:
        recording_path: Path to audio file (mp3 or wav).

    Returns:
        AudioData with wav_path, waveform, and sample_rate.

    Raises:
        FileNotFoundError: If recording_path does not exist.
    """
    if not os.path.exists(recording_path):
        raise FileNotFoundError(f"Audio path {recording_path} does not exist")

    waveform, sample_rate = torchaudio.load(recording_path)

    if ".mp3" in recording_path:
        wav_path = recording_path[: recording_path.find(".")] + ".wav"
        torchaudio.save(wav_path, waveform, sample_rate)
    else:
        wav_path = recording_path

    return AudioData(wav_path=wav_path, waveform=waveform, sample_rate=sample_rate)


def diarize(wav_path: str, diarization_pipeline: Pipeline) -> DiarizationResult:
    """Perform speaker diarization on audio file.

    Args:
        wav_path: Path to wav audio file.
        diarization_pipeline: Pre-loaded pyannote diarization pipeline.

    Returns:
        DiarizationResult with diagram and labeled_chunks.
    """
    clear_gpu_cache()

    diarization = diarization_pipeline(file=wav_path)

    time_allocation = diarization.chart()
    t_lecturer = time_allocation[0][1]
    t_audience = sum([time_allocation[i][1] for i in range(1, len(time_allocation))])
    t_silence = (
        max([segment.end for segment in diarization.itersegments()])
        - t_lecturer
        - t_audience
    )

    timestamps_of_speakers = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        timestamps_of_speakers.append(
            [speaker, round(segment.start, 2), round(segment.end, 2)]
        )

    lector_id = time_allocation[0][0]
    for i in range(len(timestamps_of_speakers)):
        if timestamps_of_speakers[i][0] == lector_id:
            timestamps_of_speakers[i][0] = 1
        else:
            timestamps_of_speakers[i][0] = 2

    time_of_events = t_lecturer + t_audience + t_silence

    diagram = {
        "lecturer": t_lecturer / time_of_events * 100.0,
        "discussion": t_audience / time_of_events * 100.0,
        "quiet": t_silence / time_of_events * 100.0,
    }
    labeled_chunks = fill_silence_intervals(timestamps_of_speakers)

    return DiarizationResult(diagram=diagram, labeled_chunks=labeled_chunks)


def transcribe(
    audio: AudioData,
    diarization: DiarizationResult,
    speech_pipe,
    processor,
) -> TranscriptionResult:
    """Transcribe audio using speech recognition.

    Args:
        audio: AudioData with waveform and sample_rate.
        diarization: DiarizationResult with labeled_chunks.
        speech_pipe: Pre-loaded speech recognition pipeline.
        processor: Pre-loaded audio processor.

    Returns:
        TranscriptionResult with chunks and lecture_text.
    """
    clear_gpu_cache()
    chunks = []

    target_sample_rate = processor.feature_extractor.sampling_rate

    for speaker, start, end in diarization.labeled_chunks:
        start_sample = int(start * audio.sample_rate)
        end_sample = int(end * audio.sample_rate)
        fragment = audio.waveform[:, start_sample:end_sample]

        if fragment.shape[0] > 1:
            fragment = fragment.mean(dim=0, keepdim=True)

        fragment = fragment.squeeze(0)
        fragment_np = fragment.numpy()

        if audio.sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio.sample_rate, new_freq=target_sample_rate
            )
            fragment_resampled = resampler(fragment.unsqueeze(0))
            fragment_resampled = fragment_resampled.squeeze(0)
            fragment_np = fragment_resampled.numpy()

        text = ""
        if speaker != 3:
            text = speech_pipe(
                inputs=fragment_np,
                generate_kwargs={"language": "russian"},
                return_timestamps=True,
            )["text"]

        if speaker == 3 or text.strip() == "" or text == " Продолжение следует...":
            text = ""
            speaker = 3

        chunks.append([speaker, text.strip(), (start, end)])

    lecture_text = ""
    for _, text, _ in chunks:
        lecture_text += " " + text.strip()
    lecture_text = lecture_text.replace("  ", " ").strip()

    return TranscriptionResult(chunks=chunks, lecture_text=lecture_text)


def analyze_words(
    chunks: List[List], stopwords_path: str = STOPWORDS_PATH
) -> Tuple[List[Dict], List[Dict]]:
    """Analyze word frequency in chunks.

    Args:
        chunks: List of [speaker, text, timestamp] items.
        stopwords_path: Path to stopwords file.

    Returns:
        Tuple of (popular_words_no_stopw, popular_words_w_stopw).
    """
    with open(stopwords_path) as f:
        stopwords = set(f.read().splitlines())

    word_no_stopw = {1: [], 2: []}
    word_w_stopw = {1: [], 2: []}

    for speaker, text, _ in chunks:
        if speaker != 3:
            word_no_stopw[speaker].extend(
                [
                    word
                    for word in text.lower().split()
                    if word not in stopwords and word.isalpha()
                ]
            )
            word_w_stopw[speaker].extend(
                [word for word in text.lower().split() if word.isalpha()]
            )

    word_counts_lector_no_stopw = Counter(word_no_stopw[1])
    word_counts_audience_no_stopw = Counter(word_no_stopw[2])
    popular_words_no_stopw = [
        dict(word_counts_audience_no_stopw.most_common()[:10]),
        dict(word_counts_lector_no_stopw.most_common()[:10]),
    ]

    word_counts_lector_w_stopw = Counter(word_w_stopw[1])
    word_counts_audience_w_stopw = Counter(word_w_stopw[2])
    popular_words_w_stopw = [
        dict(word_counts_audience_w_stopw.most_common()[:10]),
        dict(word_counts_lector_w_stopw.most_common()[:10]),
    ]

    return popular_words_no_stopw, popular_words_w_stopw


def calculate_speech_speed(chunks: List[List]) -> Tuple[List[float], Dict[int, float]]:
    """Calculate speech speed in syllables per minute.

    Args:
        chunks: List of [speaker, text, timestamp] items.

    Returns:
        Tuple of (syllables_per_minute, speed_dict).
    """
    vowels = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
    total_syllables = 0
    syllables_per_minute = {}

    for speaker, text, timestamp in chunks:
        if speaker != 3:
            _, end = timestamp
            end = end // 60
            if end not in syllables_per_minute.keys():
                syllables_per_minute[end] = 0
            total_syllables += sum(text.count(vowel) for vowel in vowels)
            syllables_per_minute[end] = total_syllables

    syllables_list = np.gradient(
        list(syllables_per_minute.values()), list(syllables_per_minute.keys())
    ).tolist()

    seconds = [0]
    for _, _, timestamps in chunks:
        start, end = timestamps
        seconds.append(end)
    minutes = sorted(list(set([second // 60 for second in seconds])))
    speed = dict(zip(minutes, syllables_list))

    return syllables_list, speed


def format_chunks(chunks: List[List]) -> List[List]:
    """Format chunks for display, merging short silence intervals.

    Args:
        chunks: List of [speaker, text, timestamp] items.

    Returns:
        Formatted chunks with readable timestamps.
    """
    _transcripted_chunks = deepcopy(chunks)
    del_ind = []

    silence_intervals = [i[2][1] - i[2][0] for i in chunks if i[0] == 3]
    if not silence_intervals:
        mean = 0
    else:
        mean = sum(silence_intervals) / len(silence_intervals)

    for i in range(len(_transcripted_chunks)):
        if _transcripted_chunks[i][0] == 3:
            if (
                float(_transcripted_chunks[i][2][1])
                - float(_transcripted_chunks[i][2][0])
                <= mean
            ):
                del_ind.append(i)

    for i in del_ind[::-1]:
        del _transcripted_chunks[i]
        if i < len(_transcripted_chunks) and i > 0:
            if _transcripted_chunks[i - 1][0] == _transcripted_chunks[i][0]:
                _transcripted_chunks[i - 1][1] += _transcripted_chunks[i][1]
                _transcripted_chunks[i - 1][2] = list(_transcripted_chunks[i - 1][2])
                _transcripted_chunks[i - 1][2][1] = _transcripted_chunks[i][2][1]
                del _transcripted_chunks[i]

    for i in range(len(_transcripted_chunks)):
        start, end = _transcripted_chunks[i][2]
        _transcripted_chunks[i][2] = f"{int(start // 60)}:{int(start % 60)}"

    return _transcripted_chunks


def analyze_text(chunks: List[List]) -> TextAnalysis:
    """Perform complete text analysis on chunks.

    Args:
        chunks: List of [speaker, text, timestamp] items.

    Returns:
        TextAnalysis with all text metrics.
    """
    popular_words_no_stopw, popular_words_w_stopw = analyze_words(chunks)
    syllables_per_minute, speed = calculate_speech_speed(chunks)
    transcripted_chunks = format_chunks(chunks)

    return TextAnalysis(
        popular_words_no_stopw=popular_words_no_stopw,
        popular_words_w_stopw=popular_words_w_stopw,
        syllables_per_minute=syllables_per_minute,
        speed=speed,
        transcripted_chunks=transcripted_chunks,
    )


def extract_clean_fragment(chunks: List[List], audio: AudioData) -> str:
    """Extract a clean fragment of lecturer speech for voice cloning.

    Args:
        chunks: List of [speaker, text, timestamp] items.
        audio: AudioData with waveform and sample_rate.

    Returns:
        Path to saved clean audio fragment.
    """
    fragment = None
    for chunk in chunks[10:]:
        if chunk[0] == 1:
            if chunk[2][1] - chunk[2][0] >= 15:
                fragment = chunk
                break

    if fragment is None:
        raise ValueError("No suitable lecturer fragment found for voice cloning")

    start_sample = int(fragment[2][0] * audio.sample_rate)
    end_sample = int(fragment[2][1] * audio.sample_rate)
    segment = audio.waveform[:, start_sample:end_sample]

    output = audio.wav_path[: audio.wav_path.rfind(".")] + "_clean.wav"
    torchaudio.save(output, segment, audio.sample_rate)

    return output


def generate_podcast(
    podcast_text: dict,
    clean_fragment_path: str,
    wav_path: str,
    f5tts: F5TTS,
    accentizer: RUAccent,
) -> str:
    """Generate podcast audio from text.

    Args:
        podcast_text: Dict with part_N keys containing presenter/lector text.
        clean_fragment_path: Path to clean lecturer audio for voice cloning.
        wav_path: Path to original wav file (for cleanup).
        f5tts: Pre-loaded F5TTS model.
        accentizer: Pre-loaded RUAccent accentizer.

    Returns:
        Path to generated podcast mp3 file.
    """
    podcast_text = deepcopy(podcast_text)

    presenter_speech = ""
    lector_speech = ""
    final_audio = np.array([])

    for i in podcast_text.keys():
        for speaker in podcast_text[i].keys():
            if "presenter" in speaker:
                presenter_speech, sr, spec = f5tts.infer(
                    ref_file="utils/podcast_host.wav",
                    ref_text="",
                    gen_text=accentizer.process_all(podcast_text[i]["presenter"])
                    .strip()
                    .lower(),
                    cfg_strength=3,
                    seed=42,
                    nfe_step=64,
                    remove_silence=True,
                )
            if "lector" in speaker:
                lector_speech, sr, spec = f5tts.infer(
                    ref_file=clean_fragment_path,
                    ref_text="",
                    gen_text=accentizer.process_all(podcast_text[i]["lector"])
                    .strip()
                    .lower(),
                    cfg_strength=2,
                    seed=42,
                    nfe_step=64,
                    remove_silence=True,
                )
        final_audio = np.concatenate((final_audio, presenter_speech, lector_speech))

    output_file_path = str(uuid.uuid4())

    torchaudio.save(
        output_file_path + ".wav", torch.tensor(final_audio).unsqueeze(0), 24000
    )

    audio_segment = AudioSegment.from_file(output_file_path + ".wav", format="wav")
    audio_segment.export(output_file_path + ".mp3", format="mp3", bitrate="192k")

    # Cleanup
    os.remove(output_file_path + ".wav")
    os.remove(wav_path)
    os.remove(clean_fragment_path)

    clear_gpu_cache()

    return output_file_path + ".mp3"


# =============================================================================
# Main Service Class
# =============================================================================


class LectureAnalyzer:
    """Lecture analysis service.

    Initialize once at service startup, then call process() for each recording.
    Models are loaded once during initialization for efficiency.

    Example:
        analyzer = LectureAnalyzer(pyannote_api_key="...")
        result = analyzer.process("lecture.mp3", "record_123", "group_A")
    """

    def __init__(
        self,
    ):
        """Initialize the analyzer service.

        Loads all ML models once. This should be called once at service startup.

        Args:
            pyannote_api_key: API key for pyannote speaker diarization.
            llm_base_url: Base URL for the LLM API endpoint.
        """
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.diarization_pipeline = self._load_diarization_pipeline(os.getenv("HUGGINGFACEHUB_API_TOKEN"))
        self.speech_pipe, self.processor = self._load_whisper()
        self.f5tts, self.accentizer = self._load_tts()

        self.llm = ChatOpenAI(
            model=os.getenv("LLM_NAME"),
            base_url=os.getenv("LLM_URL"),
            api_key=os.getenv("LLM_API_KEY"),
            temperature=0,
        )

        # Structured outputs for validated responses
        self.questions_llm = self.llm.with_structured_output(QuestionsResult)
        self.podcast_llm = self.llm.with_structured_output(PodcastScript)

    def _load_diarization_pipeline(self, api_key: str) -> Pipeline:
        """Load pyannote speaker diarization pipeline."""
        return Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=api_key,
        ).to(torch.device(self.device))

    def _load_whisper(self):
        """Load Whisper speech recognition model."""
        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
        return pipe, processor

    def _load_tts(self):
        """Load F5TTS and RUAccent for podcast generation."""
        f5tts = F5TTS(
            ckpt_file="F5TTS/ckpts/model_last_inference.safetensors",
            vocab_file="F5TTS/ckpts/vocab.txt",
            device="cuda",
        )
        accentizer = RUAccent()
        accentizer.load(
            omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False
        )
        return f5tts, accentizer

    # =========================================================================
    # LLM Methods (langchain + structured outputs)
    # =========================================================================

    def _parse_json(self, text: str) -> dict:
        """Extract JSON from LLM response.

        Args:
            text: Raw LLM response that may contain JSON.

        Returns:
            Parsed JSON as dict.
        """
        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*", "", text)

        # Find JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"No valid JSON found in: {text[:100]}...")

    def _generate_abstract(self, lecture_text: str) -> str:
        """Generate lecture abstract (markdown).

        Args:
            lecture_text: Full lecture transcription.

        Returns:
            Markdown abstract text.
        """
        response = self.llm.invoke(f"{ABSTRACT_PROMPT}\n\nТекст лекции:\n{lecture_text}")
        return latex_to_md(response.content)

    def _generate_questions(self, lecture_text: str) -> List[str]:
        """Generate self-check questions.

        Args:
            lecture_text: Full lecture transcription.

        Returns:
            List of 10-12 questions.
        """
        result = self.questions_llm.invoke(
            f"{QUESTIONS_PROMPT}\n\nТекст лекции:\n{lecture_text}"
        )
        return result.questions

    def _generate_mindmap(self, lecture_text: str) -> dict:
        """Generate mind map structure.

        Args:
            lecture_text: Full lecture transcription.

        Returns:
            Dict with mind map structure.
        """
        response = self.llm.invoke(f"{MINDMAP_PROMPT}\n\nТекст лекции:\n{lecture_text}")
        return self._parse_json(response.content)

    def _analyze_emotion(self, text: str) -> str:
        """Analyze emotional tone of a text fragment.

        Args:
            text: Text fragment to analyze.

        Returns:
            Adjectives describing the fragment.
        """
        response = self.llm.invoke(f"{EMOT_AN_PROMPT}\n\nФрагмент:\n{text}")
        return response.content.strip()

    def _generate_podcast_script(self, lecture_text: str, questions: List[str]) -> dict:
        """Generate podcast script.

        Args:
            lecture_text: Full lecture transcription.
            questions: List of questions for the podcast.

        Returns:
            Dict with podcast parts.
        """
        questions_str = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        result = self.podcast_llm.invoke(
            f"{PODCAST_PROMPT}\n\nВопросы:\n{questions_str}\n\nЛекция:\n{lecture_text}"
        )
        return result.model_dump()

    def _clean_podcast_part(self, part: PodcastPart) -> dict:
        """Clean a single podcast part for TTS.

        Args:
            part: PodcastPart with presenter and lector text.

        Returns:
            Cleaned podcast part as dict.
        """
        presenter_response = self.llm.invoke(
            f"{CLEAN_PODCAST_PROMPT}\n\nТекст:\n{part.presenter}"
        )
        cleaned_presenter = presenter_response.content

        lector_response = self.llm.invoke(
            f"{CLEAN_PODCAST_PROMPT}\n\nТекст:\n{part.lector}"
        )
        cleaned_lector = lector_response.content

        return {"presenter": cleaned_presenter, "lector": cleaned_lector}

    def _clean_podcast(self, podcast_script: dict) -> dict:
        """Clean entire podcast script for TTS.

        Args:
            podcast_script: Dict with podcast parts.

        Returns:
            Cleaned podcast script formatted for generate_podcast function.
        """
        cleaned_parts = {}
        for i, part in enumerate(podcast_script["parts"]):
            part_obj = PodcastPart(**part) if isinstance(part, dict) else part
            cleaned_parts[f"part_{i + 1}"] = self._clean_podcast_part(part_obj)
        return cleaned_parts

    def process(
        self,
        recording_path: str,
        record_id: str,
        group: str,
        lection_date: datetime.date = None,
    ) -> dict:
        """Process a single lecture recording.

        Args:
            recording_path: Path to audio file (mp3 or wav).
            record_id: Unique identifier for this recording.
            group: Student group identifier.
            lection_date: Date of the lecture (defaults to today).

        Returns:
            Dict with all analysis results.

        Raises:
            FileNotFoundError: If recording_path does not exist.
        """
        if lection_date is None:
            lection_date = datetime.date.today()

        audio = prepare_audio(recording_path)

        diarization_result = diarize(audio.wav_path, self.diarization_pipeline)

        transcription = transcribe(
            audio, diarization_result, self.speech_pipe, self.processor
        )

        text_analysis = analyze_text(transcription.chunks)

        llm_result = self._llm_analyze(
            transcription.lecture_text, text_analysis.transcripted_chunks
        )

        clean_fragment_path = extract_clean_fragment(transcription.chunks, audio)
        podcast_path = generate_podcast(
            llm_result["podcast_text"],
            clean_fragment_path,
            audio.wav_path,
            self.f5tts,
            self.accentizer,
        )

        clear_gpu_cache()

        return {
            "lecture_text": transcription.lecture_text,
            "abstract_text": llm_result["abstract_text"],
            "speech_speed": text_analysis.speed,
            "mindmap": llm_result["mind_map"],
            "popular_words_no_stopw": text_analysis.popular_words_no_stopw,
            "popular_words_w_stopw": text_analysis.popular_words_w_stopw,
            "conversation_static": diarization_result.diagram,
            "lecture_timeline": llm_result["final_chunks"],
            "questions": llm_result["questions"],
            "podcast": podcast_path,
        }

    def get_results(
        self,
        recording_path: str,
        record_id: str,
        group: str,
        lection_date: datetime.date = None,
    ) -> str:
        """Process recording and return results as JSON string.

        This method provides backward compatibility with the old API.

        Args:
            recording_path: Path to audio file.
            record_id: Unique identifier for this recording.
            group: Student group identifier.
            lection_date: Date of the lecture.

        Returns:
            JSON string with all analysis results.
        """
        result = self.process(recording_path, record_id, group, lection_date)
        return json.dumps(result, default=str)

    def _llm_analyze(
        self, lecture_text: str, transcripted_chunks: List[List]
    ) -> dict:
        """Analyze lecture using LLM with structured outputs.

        Generates abstract, questions, mind map, podcast text, and emotional analysis.

        Args:
            lecture_text: Full lecture transcription.
            transcripted_chunks: Formatted chunks with timestamps.

        Returns:
            Dict with abstract_text, questions, mind_map, podcast_text, final_chunks.
        """
        abstract_text = self._generate_abstract(lecture_text)

        final_chunks = deepcopy(transcripted_chunks)
        for i in range(len(final_chunks)):
            if final_chunks[i][0] == 1:  # lecturer
                emotion = self._analyze_emotion(final_chunks[i][1])
                final_chunks[i].append(emotion)

        mind_map = self._generate_mindmap(lecture_text)

        questions = self._generate_questions(lecture_text)

        podcast_script = self._generate_podcast_script(lecture_text, questions)

        podcast_text = self._clean_podcast(podcast_script)

        return {
            "abstract_text": abstract_text,
            "final_chunks": final_chunks,
            "mind_map": mind_map,
            "questions": questions,
            "podcast_text": podcast_text,
        }
