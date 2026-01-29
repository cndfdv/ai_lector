CREATE TABLE IF NOT EXISTS lecture_notes (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    student_group TEXT NOT NULL,
    lection_text TEXT NOT NULL
);
