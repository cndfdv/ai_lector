-- Основная таблица лекций с метаданными
CREATE TABLE IF NOT EXISTS lectures (
    id VARCHAR(36) PRIMARY KEY,
    record_id VARCHAR(255) UNIQUE,
    student_group TEXT NOT NULL,
    lecture_date DATE NOT NULL,
    content TEXT NOT NULL,
    abstract TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Таблица чанков для отслеживания связи с Milvus
CREATE TABLE IF NOT EXISTS lecture_chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    lecture_id VARCHAR(36) REFERENCES lectures(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Индексы для быстрого поиска
CREATE INDEX IF NOT EXISTS idx_lectures_group ON lectures(student_group);
CREATE INDEX IF NOT EXISTS idx_lectures_date ON lectures(lecture_date);
CREATE INDEX IF NOT EXISTS idx_lectures_record ON lectures(record_id);
CREATE INDEX IF NOT EXISTS idx_chunks_lecture ON lecture_chunks(lecture_id);
