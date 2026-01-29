import datetime
import json
import os
import re
import uuid
from collections import Counter
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import requests
import torch
import torchaudio
from pyannote.audio import Pipeline
from pydub import AudioSegment
from ruaccent import RUAccent
from sqlalchemy import text
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from F5TTS.f5_tts.api import F5TTS
from src.rag import RAG

STOPWORDS_PATH = "src/stopwords.txt"

ABSTRACT_PROMPT = """Ты — ИИ-помощник. Получишь текст лекции и должен сделать подробный, структурированный конспект.

Инструкция:
1. Отрази все ключевые идеи, определения, примеры и выводы.
2. Пиши просто и понятно, чтобы понял человек без подготовки.
3. Используй заголовки, списки и выделение для структуры.

Формат вывода:
- Верни только сам конспект без вводных фраз и пояснений.
- Язык: русский. Латиница разрешена только для терминов, формул, имён собственных и кода (например, API, Python, O(n)).
Не объясняй, не комментируй, просто выведи готовый конспект.
Правила:
1. Заголовки: #, ##, ### и т.д.
2. Списки: "-" или "1."
3. Формулы: 
    Все inline формулы заключай в `$...$`
    Все блочные формулы заключай в `$$...$$`
4. Не используй \(...\) или \[...\]
5. Не добавляй лишних пояснений, текста, json или html.
6. Верни только Markdown-текст."""

EMOT_AN_PROMPT = """Ты — ИИ-оценщик лекционных фрагментов. Получишь отрывок лекции и должен оценить его по содержанию и эмоциональной подаче.

Инструкция:
- Выбери два прилагательных, описывающих фрагмент.
- Язык: русский. Используй только кириллицу.
- Не используй латиницу, не смешивай алфавиты.
- Верни только эти два слова, без комментариев, вводных и знаков препинания."""

MINDMAP_PROMPT = """Ты — ИИ-помощник. Преврати текст лекции в иерархическую интеллект-карту (mindmap).

Задача:
Проанализируй текст лекции, выдели основные темы, подтемы и детали. Структурируй материал в виде вложенного JSON-объекта, отражающего смысловую структуру лекции.

Формат JSON:
{
    "title": "название лекции",
    "nodes": [
    {
        "id": "основная тема",
        "label": "основная тема",
        "children": [
        {
            "id": "подтема",
            "label": "подтема",
            "children": []
        }
        ]
    }
    ]
}

Требования:
1. Используй строго этот формат JSON.
2. Если у темы нет подтем, оставь `"children": []`.
3. Язык — русский, только кириллица (латиница запрещена).
4. Не добавляй пояснений, форматирования, markdown, комментариев или текста до/после JSON.
5. Верни только чистый корректный JSON-объект.
"""

QUESTIONS_PROMPT = """Ты — ИИ-методист. Получишь текст лекции и должен составить список вопросов для самопроверки.

Инструкция:
1. Сформулируй 10 вопросов для самопроверки только по материалу лекции.
2. Следуй порядку изложения лекции.
3. Вопросы должны охватывать ключевые моменты материала.

Формат вывода:
Верни только нумерованный список вопросов.
Язык: русский. Латиница допустима только для терминов и имён собственных.
Не добавляй пояснений и комментариев.
Если нужно форматирование - используй ТОЛЬКО markdown (.md)
"""

PODCAST_PROMPT = """Ты — ИИ-сценарист. Напиши сценарий подкаста в виде диалога между ведущим и лектором, используя вопросы и текст лекции.

Инструкция:
1. Начни с приветствия — это part_1. Пример: 'Здравствуйте, слушатели! Сегодня у нас в гостях автор лекции по 'тема лекции'.'
2. **ОБЯЗАТЕЛЬНО: Соблюдай строгую гендерную нейтральность.** Никаких слов типа 'рад', 'готов', 'написал'. Используй нейтральные формы: 'приятно', 'готовность', 'автор лекции', 'благодарю'.
3. Для каждого вопроса сделай новый part_N. Ведущий задаёт вопрос, лектор отвечает подробно по лекции.
4. Заверши сценарием прощания.
5. Стиль речи: живой, дружелюбный, естественный.
6. Не используй переносы строк внутри строк.
7. Все строки должны быть в двойных кавычках.
8. Верни **ТОЛЬКО КОРРЕКТНЫЙ JSON**, без лишних символов, комментариев или пояснений.
9. JSON должен быть минимально отформатированным (можно без отступов).

Формат вывода:
{
    "part_1": {
        "presenter": "Здравствуйте, слушатели! Сегодня у нас гость — автор лекции.",
        "lector": "Добрый день! Приятно быть здесь."
},
    "part_2": {
        "presenter": "Начнём с первого вопроса...",
        "lector": "Согласно лекции, ..."
    }
}
"""

CLEAN_PODCAST = """Ты — умный литературный редактор для системы синтеза речи (Text-to-Speech).
Тебе даётся JSON-структура, содержащая реплики ведущего ('presenter') и лектора ('lector'). Твоя задача — вернуть ТОТ ЖЕ JSON, но с фразами, адаптированными для естественного произнесения вслух.

ПРИОРИТЕТНЫЕ ПРАВИЛА (Соблюдай СТРОГО для ВСЕХ реплик — 'presenter' и 'lector'):

1. КРИТИЧЕСКОЕ ПРАВИЛО (Числа): Переведи **АБСОЛЮТНО ВСЕ ЧИСЛА** (включая порядковые номера, годы, проценты, дроби и т.д.) в слова и согласуй их с падежом.
    Примеры: 'Вопрос 1' → 'Вопрос первый'; '3.14' → 'три целых четырнадцать сотых'.

2. КРИТИЧЕСКОЕ ПРАВИЛО (Формулы и Символы):
    Формулы и символы ($\sigma^2$, $N$, $k$, $x$) должны быть прописаны словами: «сигма в квадрате», «эн большое», «ка-тый», «икс».
    Удаляй ВСЕ кавычки (`«`, `»`, `"`, `'`) вокруг прописанных формул и символов, чтобы избежать интонационных сбоев.
    Замени все разновидности тире (`—`, `–`) на **обычный дефис** (`-`) или пробел, если тире не обозначает паузу, а является разделителем.

3. Грамматика и Плавность: Исправь все грамматические ошибки, несогласованные падежи, числа и роды. Добавляй связующие слова для плавности ('поэтому', 'кроме того').

4. Сохрани естественный, живой, но чистый стиль речи.

Формат вывода:
    Верни СТРОГО КОРРЕКТНЫЙ JSON-объект, без каких-либо вводных слов, комментариев или пояснений.
    Не изменяй структуру, ключи и вложенность.

Входные данные:
"""


class LectureHelper:
    """Audio recording analyzer class.
    NOTE: All attributes that correspond to metrics are stored in _cache, which is used to provide lazy initialization functionality.

    Attributes:
        _cache (dict): Stores calculated metrics
    Attributes stored in _cache:
        lecture_text (str): Full text of lection
        abstract_text (str): Summarized text of lection
        questions (str): Generated questions for lection
        answers (str): Generated podcast text with answers om questions
        mind_map (str): JSON-like mindmap of lecture
        popular_words_no_stopw (List[Dict[str, int]]): List of the most popular words and number of their occasions without stopwords
        popular_words_w_stopw (List[Dict[str, int]]): List of the most popular words and number of their occasions with stopwords
        diagram (List[Tuple[str, float]]): Statistics for pie chart representing active time for each speaker
        syllables_per_minute (List[float]): Speed of speach in syllables/min
        speed (Dict[int, int]): Speed of speech at each minute
        chunks (List[dict]]): Full text of lection splitted in chunks. Each item in list consists of a speaker id, text and timestamp
        transcripted_chunks (List[list]): Chunks in readable format
        final chunks (List[list]): Chunks in readable format with emotional analysis
        wav_path (str): Path to wav audio of lection
        path_to_podcast (str): Path to wav audio of podcast
        labeled_chunks (List[List]): time allocation of speakers
    """

    def __init__(
        self,
        recording_path: str,
        pyannote_api_key: str,
        recordId: str,
        group: str,
        lection_date: datetime.date = datetime.date.today(),
    ):
        """Initializes an analyzer object.

        Args:
            recording_path (str): path to file with the necessary audio file
            gigachat_api_key (str): secret api key for accessing GigaChat api service
            pyannote_api_key (str): secret api key for accessing pyannote model from Huggingface
            recorID (str): ID of audio in database

        Raises:
            FileNotFoundError: raised if path to the file could not be found
        """
        self.recordId = recordId
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.pyannote_api_key = pyannote_api_key
        self.group = group
        self.lection_date = lection_date
        if os.path.exists(recording_path):
            self.recording_path = recording_path
        else:
            raise FileNotFoundError(f"Audio_path {recording_path} does not exist")

        # stores attributes with already assigned values
        self._cache = {}

        self.computations = {
            "diagram": self._set_stat,
            "labeled_chunks": self._set_stat,
            "chunks": self._set_chunks,
            "lecture_text": self._set_lecture_text,
            "popular_words_no_stopw": self._set_popular_words,
            "popular_words_w_stopw": self._set_popular_words,
            "syllables_per_minute": self._set_syllables_per_minute,
            "speed": self._set_speech_speed,
            "transcripted_chunks": self._set_transcripted_chunks,
            "wav_path": self._prepair_audio,
            "waveform": self._prepair_audio,
            "sample_rate": self._prepair_audio,
            "abstract_text": self.LLM_analyze,
            "questions": self.LLM_analyze,
            "podcast_text": self.LLM_analyze,
            "mind_map": self.LLM_analyze,
            "final_chunks": self.LLM_analyze,
            "clean_fragment": self._extract_clean_fragment,
            "path_to_podcast": self._generate_podcast,
        }

    def __getattr__(self, name: str):
        """Method that is raised when the attribute is called.
        Used to provide lazy initialization functionality:
        metrics are calculated only when the atribute is called for the first time.

        Args:
            name (str): name of an attribute to reach

        Raises:
            AttributeError: raised only if the attribute doesn't exist (metric is not specified)

        Returns:
            Metric corresponding to the attribute
        """

        if name in self.computations:
            if name not in self._cache:  # Compute and store only if not already set
                self.computations[name]()
            return self._cache[name]

        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def get_results(self):
        """Json format of some attributes."""

        return json.dumps(
            {
                "lecture_text": self.lecture_text,
                "abstract_text": self.abstract_text,
                "speech_speed": self.speed,
                "mindmap": self.mind_map,
                "popular_words_no_stopw": self.popular_words_no_stopw,
                "popular_words_w_stopw": self.popular_words_w_stopw,
                "conversation_static": self.diagram,
                "lecture_timeline": self.final_chunks,
                "questions": self.questions,
                "podcast": self.path_to_podcast,
            },
            default=str,
        )

    def _fill_silence_intervals(
        self,
        data: List[Tuple[str, float, float]],
    ) -> List[Tuple[str, float, float]]:
        """Fills intervals when no words were spoken."""
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

    def _clean_json(self, text: str) -> dict:
        """
        Попытка извлечь и очистить JSON-строку, сгенерированную моделью.
        Включает несколько шагов исправления до использования LLM для реконструкции.
        """

        # --- Шаг 0: Подготовка текста

        # 0.1. Извлекаем JSON (только между первой { и последней })
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            # Невозможно начать, если нет скобок.
            raise ValueError("В тексте не найден JSON-объект (нет {}).")
        text = text[start : end + 1]

        # 0.2. Предварительная очистка и нормализация
        # Удаляем невидимые символы (включая неразрывные пробелы)
        text = re.sub(r"[\u0000-\u001F\u007F-\u009F\u00AD\u202f\xa0]", " ", text)

        # Нормализуем типографские кавычки
        text = (
            text.replace("“", '"').replace("”", '"').replace("′", '"').replace("″", '"')
        )

        # 0.3. Исправляем "голые" обратные слэши
        text = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", text)

        # --- Шаг 1: Попытка 1 (Базовый парсинг)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e1:
            error_details = e1

        # --- Шаг 2: Агрессивное исправление формата (Ключи и одинарные кавычки)

        # 2.1. Исправляем "голые" ключи и значения: key: value → "key": "value"
        def fix_key_value_and_quotes(match):
            key = match.group(1).strip()
            val = match.group(2).strip()

            # Убираем завершающую запятую, если есть
            val = re.sub(r",$", "", val)

            # Экранируем ключ и добавляем кавычки
            key_fixed = key.replace('"', '\\"')
            if not (key_fixed.startswith('"') and key_fixed.endswith('"')):
                key_fixed = f'"{key_fixed}"'

            # Если значение не в кавычках — добавим, заменяя одинарные на двойные
            if not (val.startswith('"') and val.endswith('"')):
                # Заменяем одинарные кавычки внутри строки на двойные для совместимости с JSON
                val = val.replace("'", '"')
                # Экранируем внутренние двойные кавычки
                val = val.replace('"', '\\"')
                val = f'"{val}"'
            # Если значение уже в кавычках (одинарных или двойных), гарантируем, что двойных
            elif val.startswith("'") and val.endswith("'"):
                val = (
                    val[1:-1].replace('"', '\\"').replace("'", '"')
                )  # Замена одинарных на двойные
                val = f'"{val}"'

            return f"{key_fixed}: {val}"

        # Находим и исправляем пары key: value, где ключ или значение не в кавычках.
        text_fixed_quotes = re.sub(
            r'(\w+|"[^"]+"|' + r"'[^']+'" + r")\s*:\s*([^,{}\n]+)",
            fix_key_value_and_quotes,
            text,
            flags=re.DOTALL,
        )

        # 2.2. Удаляем висячие запятые (Trailing Commas)
        # Находим запятую, за которой следуют пробелы/переводы строки и сразу } или ]
        text_fixed_quotes = re.sub(r",\s*([}\]])", r"\1", text_fixed_quotes)

        # --- Шаг 3: Попытка 2 (Парсинг после исправления кавычек и запятых)
        try:
            return json.loads(text_fixed_quotes)
        except json.JSONDecodeError as e2:
            error_details = e2
            text_to_reconstruct = (
                text_fixed_quotes  # Сохраняем последний исправленный текст
            )

        # --- Шаг 4: Последняя попытка: Мягкая реконструкция JSON через LLM

        clear_json_raw = self._llm_generating(
            f"""Произошла ошибка при json.loads: {error_details}. 
            Исправь этот текст, чтобы он стал **строго** корректным JSON. 
            **Верни ТОЛЬКО исправленный JSON (один объект)** без комментариев, форматирования, 
            Markdown-блоков (например, ```json) или Unicode-пробелов (\\u202f, \\xa0). 
            
            Изначальный текст, который не удалось распарсить:
            {text_to_reconstruct}
            """
        )

        # 4.1. Снова извлекаем JSON из ответа LLM
        start = clear_json_raw.find("{")
        end = clear_json_raw.rfind("}")

        if start != -1 and end != -1:
            clear_json = clear_json_raw[start : end + 1]
            clear_json = re.sub(r"[\u0000-\u001F\u007F-\u009F\u00AD]", "", clear_json)

            # Попытка 3: Парсинг реконструированного LLM текста
            try:
                return json.loads(clear_json)
            except Exception as e3:
                # Если LLM-реконструкция не помогла, поднимаем конечную ошибку
                raise ValueError(
                    f"Не удалось распарсить JSON после всех попыток. Ошибка: {e3}"
                )
        else:
            # Если LLM не вернул {}
            raise ValueError(
                f"Не удалось распарсить JSON. LLM-реконструкция не вернула JSON-объект. Последняя ошибка: {error_details}"
            )

    def _latex_to_md(self, text: str) -> str:
        """Convert LaTeX syntax to Markdown math syntax.

        Args:
            text (str): Text to convert.

        Returns:
            str: Converted text with Markdown math delimiters.
        """
        text = re.sub(r"\\\(\s*(.*?)\s*\\\)", r"$\1$", text)

        text = re.sub(r"\\\[\s*(.*?)\s*\\\]", r"$$\1$$", text, flags=re.DOTALL)

        return text

    def _clear_gpu_cache(self):
        """Clear the GPU cache if CUDA is available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _set_stat(self):
        """Calculates statistics for diagram, and creates chunks labeled by speaker."""
        self._clear_gpu_cache()
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.pyannote_api_key,
        ).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        diarization = pipeline(file=self.wav_path)

        time_allocation = diarization.chart()
        t_lecturer = time_allocation[0][1]
        t_audience = sum(
            [time_allocation[i][1] for i in range(1, len(time_allocation))]
        )
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

        self._cache["diagram"] = {
            "lecturer": t_lecturer / time_of_events * 100.0,
            "discussion": t_audience / time_of_events * 100.0,
            "quiet": t_silence / time_of_events * 100.0,
        }
        self._cache["labeled_chunks"] = self._fill_silence_intervals(
            timestamps_of_speakers
        )

    def _set_chunks(self):
        """Creates chunks in the folowing format: [speaker_id, text, (time_of_start, time_of_end)]."""
        self._clear_gpu_cache()
        chunks = []

        model_id = "openai/whisper-large-v3"
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self.device)

        processor = AutoProcessor.from_pretrained(model_id)
        speech_recognition_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=self.torch_dtype,
            device=self.device,
        )

        for speaker, start, end in self.labeled_chunks:
            start_sample = int(start * self.sample_rate)
            end_sample = int(end * self.sample_rate)
            fragment = self.waveform[:, start_sample:end_sample]

            if fragment.shape[0] > 1:
                fragment = fragment.mean(dim=0, keepdim=True)

            fragment = fragment.squeeze(0)
            fragment_np = fragment.numpy()

            target_sample_rate = processor.feature_extractor.sampling_rate
            if self.sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=self.sample_rate, new_freq=target_sample_rate
                )
                fragment_resampled = resampler(fragment.unsqueeze(0))
                fragment_resampled = fragment_resampled.squeeze(0)
                fragment_np = fragment_resampled.numpy()

            if speaker != 3:
                text = speech_recognition_pipe(
                    inputs=fragment_np,
                    generate_kwargs={"language": "russian"},
                    return_timestamps=True,
                )["text"]
            if speaker == 3 or text.strip() == "" or text == " Продолжение следует...":
                text = ""
                speaker = 3
            chunks.append([speaker, text.strip(), (start, end)])
        self._cache["chunks"] = chunks

    def _set_lecture_text(self):
        """Creates transcription of the recording and text of the lection splitted into chunks."""
        lecture_text = ""
        for _, text, _ in self.chunks:
            lecture_text += " " + text.strip()

        self._cache["lecture_text"] = lecture_text.replace("  ", " ").strip()

    def _set_popular_words(self):
        """Calculates the most common words."""
        with open(STOPWORDS_PATH) as f:
            stopwords = set(f.read().splitlines())
        word_no_stopw = {1: [], 2: []}

        for speaker, text, _ in self.chunks:
            # if not silence
            if speaker != 3:
                word_no_stopw[speaker].extend(
                    [
                        word
                        for word in text.lower().split()
                        if word not in stopwords and word.isalpha()
                    ]
                )
        word_counts_lector_no_stopw = Counter(word_no_stopw[1])
        word_counts_audience_no_stopw = Counter(word_no_stopw[2])
        popular_words_no_stopw = [
            dict(word_counts_audience_no_stopw.most_common()[:10]),
            dict(word_counts_lector_no_stopw.most_common()[:10]),
        ]

        self._cache["popular_words_no_stopw"] = popular_words_no_stopw

        word_w_stopw = {1: [], 2: []}
        for speaker, text, _ in self.chunks:
            # if not silence
            if speaker != 3:
                word_w_stopw[speaker].extend(
                    [word for word in text.lower().split() if word.isalpha()]
                )
        word_counts_lector_w_stopw = Counter(word_w_stopw[1])
        word_counts_audience_w_stpow = Counter(word_w_stopw[2])
        popular_words_w_stopw = [
            dict(word_counts_audience_w_stpow.most_common()[:10]),
            dict(word_counts_lector_w_stopw.most_common()[:10]),
        ]

        self._cache["popular_words_w_stopw"] = popular_words_w_stopw

    def _set_syllables_per_minute(self):
        """Calculates speed of speech in syllables per minute."""
        vowels = ["а", "е", "ё", "и", "о", "у", "ы", "э", "ю", "я"]
        total_syllables = 0
        syllables_per_minute = {}
        for speaker, text, timestamp in self.chunks:
            if speaker != 3:
                _, end = timestamp
                end = end // 60
                if end not in syllables_per_minute.keys():
                    syllables_per_minute[end] = 0
                total_syllables += sum(text.count(vowel) for vowel in vowels)
                syllables_per_minute[end] = total_syllables
        self._cache["syllables_per_minute"] = np.gradient(
            list(syllables_per_minute.values()), list(syllables_per_minute.keys())
        ).tolist()

    def _set_speech_speed(self):
        """Calculates speed of speech at each minute."""
        seconds = [0]
        for _, _, timestamps in self.chunks:
            start, end = timestamps
            seconds.append(end)
        minutes = sorted(list(set([second // 60 for second in seconds])))
        speed = dict(zip(minutes, self.syllables_per_minute))
        self._cache["speed"] = speed

    def _set_transcripted_chunks(self):
        """Creates transcripted chunks in readable format."""
        _transcripted_chunks = deepcopy(self.chunks)
        del_ind = []
        silence_intervals = [i[2][1] - i[2][0] for i in self.chunks if i[0] == 3]
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
            if _transcripted_chunks[i - 1][0] == _transcripted_chunks[i][0]:
                _transcripted_chunks[i - 1][1] += _transcripted_chunks[i][1]
                _transcripted_chunks[i - 1][2] = list(_transcripted_chunks[i - 1][2])
                _transcripted_chunks[i - 1][2][1] = _transcripted_chunks[i][2][1]
                del _transcripted_chunks[i]

        for i in range(len(_transcripted_chunks)):
            start, end = _transcripted_chunks[i][2]
            _transcripted_chunks[i][2] = f"{int(start // 60)}:{int(start % 60)}"
        self._cache["transcripted_chunks"] = _transcripted_chunks

    def _llm_generating(self, task: str) -> str:
        """Send request to local LLM model

        Args:
            task (str): task from LLM

        Returns:
            str: LLM output
        """
        url = "http://10.162.1.92:1234/v1/chat/completions"

        model_name = "gpt-oss-lab"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": task}],
            "reasoning": {"effort": "high", "summary": "auto"},
        }
        response = requests.post(url, headers=headers, json=payload)
        reply = response.json()["choices"][0]["message"]["content"]

        return reply

    def LLM_analyze(self):
        """Analyzes text using yandexgpt to generate abstract of text, questions, podcast text with answers, mind map and summarized."""

        self._cache["abstract_text"] = self._latex_to_md(
            self._llm_generating(f"{ABSTRACT_PROMPT}: {self.lecture_text}")
        )

        final_chunks = deepcopy(self.transcripted_chunks)
        for i in range(len(final_chunks)):
            if final_chunks[i][0] == 1:
                emot_analyz = self._llm_generating(
                    f"{EMOT_AN_PROMPT}: {final_chunks[i][1]}"
                )
                final_chunks[i].append(emot_analyz)
        self._cache["final_chunks"] = final_chunks

        self._cache["mind_map"] = self._clean_json(
            self._latex_to_md(
                self._llm_generating(f"{MINDMAP_PROMPT}: {self.lecture_text}")
            )
        )

        self._cache["questions"] = self._latex_to_md(
            self._llm_generating(f"{QUESTIONS_PROMPT}: {self.lecture_text}")
        )

        podcast_text = self._clean_json(
            self._latex_to_md(
                self._llm_generating(
                    f"{PODCAST_PROMPT}: Лекция: {self.lecture_text}. Вопросы: {self.questions}"
                )
            )
        )

        try:
            for i in podcast_text.keys():
                podcast_text[i] = self._clean_json(
                    self._llm_generating(f"{CLEAN_PODCAST}: {podcast_text[i]}")
                )
        except:
            podcast_text = self._clean_json(
                self._llm_generating(
                    f"{PODCAST_PROMPT}: Лекция: {self.lecture_text}. Вопросы: {self.questions}"
                )
            )
            for i in podcast_text.keys():
                podcast_text[i] = self._clean_json(
                    self._llm_generating(f"{CLEAN_PODCAST}: {podcast_text[i]}")
                )

        self._cache["podcast_text"] = podcast_text

    def _prepair_audio(self):
        """Converts audio from mp3 to wav."""
        if ".mp3" in self.recording_path:
            waveform, sample_rate = torchaudio.load(self.recording_path)
            wav_path = self.recording_path[: self.recording_path.find(".")] + ".wav"
            torchaudio.save(wav_path, waveform, sample_rate)
            self._cache["wav_path"] = wav_path
            self._cache["waveform"] = waveform
            self._cache["sample_rate"] = sample_rate
        elif ".wav" in self.recording_path:
            waveform, sample_rate = torchaudio.load(self.recording_path)
            self._cache["wav_path"] = self.recording_path
            self._cache["waveform"] = waveform
            self._cache["sample_rate"] = sample_rate

    def _extract_clean_fragment(self):
        """exctract fragment of lektor speach for clonning"""
        for chunk in self.chunks[10:]:
            if chunk[0] == 1:
                if chunk[2][1] - chunk[2][0] >= 15:
                    fragment = chunk
                    break

        start_sample = int(fragment[2][0] * self.sample_rate)
        end_sample = int(fragment[2][1] * self.sample_rate)
        segment = self.waveform[:, start_sample:end_sample]

        output = self.wav_path[: self.wav_path.rfind(".")] + "_clean.wav"

        torchaudio.save(output, segment, self.sample_rate)

        self._cache["clean_fragment"] = output

    def _generate_podcast(self):
        podcast_text = deepcopy(self.podcast_text)

        f5tts = F5TTS(
            ckpt_file="F5TTS/ckpts/model_last_inference.safetensors",
            vocab_file="F5TTS/ckpts/vocab.txt",
            device="cuda",
        )
        accentizer = RUAccent()
        accentizer.load(
            omograph_model_size="turbo3.1", use_dictionary=True, tiny_mode=False
        )

        presenter_speach = ""
        lector_speach = ""
        final_audio = np.array([])
        for i in podcast_text.keys():
            for speaker in podcast_text[i].keys():
                if "presenter" in speaker:
                    presenter_speach, sr, spec = f5tts.infer(
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
                    lector_speach, sr, spec = f5tts.infer(
                        ref_file=self.clean_fragment,
                        ref_text="",
                        gen_text=accentizer.process_all(podcast_text[i]["lector"])
                        .strip()
                        .lower(),
                        cfg_strength=2,
                        seed=42,
                        nfe_step=64,
                        remove_silence=True,
                    )
            final_audio = np.concatenate((final_audio, presenter_speach, lector_speach))

        output_file_path = str(uuid.uuid4())

        torchaudio.save(
            output_file_path + ".wav", torch.tensor(final_audio).unsqueeze(0), 24000
        )

        audio = AudioSegment.from_file(output_file_path + ".wav", format="wav")

        audio.export(output_file_path + ".mp3", format="mp3", bitrate="192k")

        self._cache["path_to_podcast"] = output_file_path + ".mp3"

        os.remove(output_file_path + ".wav")
        os.remove(self.wav_path)
        os.remove(self.clean_fragment)

        self._clear_gpu_cache()

    def insert_lecture_note(self):
        rag = RAG()
        session = rag.Session()

        insert_sql = text(
            """
            INSERT INTO lecture_notes (date, student_group, lection_text)
            VALUES (:date, :student_group, :lection_text)
            RETURNING id
            """
        )

        res = session.execute(
            insert_sql,
            {
                "date": self.lection_date,
                "student_group": self.group,
                "lection_text": self.lecture_text,
            },
        )
        session.commit()
        inserted_id = res.scalar()
        session.close()
        return inserted_id

    def insert_milvus(self):
        rag = RAG()
        lecture_id = rag.add_lecture(self.lecture_text)
        return lecture_id
