import json
import sqlite3
from datetime import datetime
from pathlib import Path

from app.core.config import settings

BASE_DIR = Path(__file__).resolve().parents[2]
DATABASE_PATH = BASE_DIR / settings.DATABASE_PATH


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def init_db() -> None:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audios (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                duration REAL,
                size INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS transcriptions (
                id TEXT PRIMARY KEY,
                audio_id TEXT NOT NULL,
                text TEXT NOT NULL,
                language TEXT NOT NULL,
                segments_json TEXT,
                confidence REAL NOT NULL,
                processing_time REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(audio_id) REFERENCES audios(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcription_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                method TEXT NOT NULL,
                original_length INTEGER NOT NULL,
                summary_length INTEGER NOT NULL,
                compression_ratio REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
            );
            """
        )


def create_audio_record(
    audio_id: str,
    filename: str,
    duration: float | None,
    size: int,
    file_path: str,
    status: str,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO audios (id, filename, duration, size, file_path, created_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (audio_id, filename, duration, size, file_path, datetime.now().isoformat(), status),
        )


def get_audio_record(audio_id: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM audios WHERE id = ?",
            (audio_id,),
        ).fetchone()


def list_audio_records(limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM audios ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def delete_audio_record(audio_id: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM audios WHERE id = ?",
            (audio_id,),
        )
        return cursor.rowcount


def update_audio_record(audio_id: str, filename: str | None = None, status: str | None = None) -> int:
    fields = []
    values = []

    if filename is not None:
        fields.append("filename = ?")
        values.append(filename)
    if status is not None:
        fields.append("status = ?")
        values.append(status)

    if not fields:
        return 0

    values.append(audio_id)

    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE audios SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        return cursor.rowcount


def create_transcription_record(
    transcription_id: str,
    audio_id: str,
    text: str,
    language: str,
    segments: list[dict] | None,
    confidence: float,
    processing_time: float,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO transcriptions (
                id, audio_id, text, language, segments_json, confidence, processing_time, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transcription_id,
                audio_id,
                text,
                language,
                json.dumps(segments or []),
                confidence,
                processing_time,
                datetime.now().isoformat(),
            ),
        )


def get_transcription_record(transcription_id: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM transcriptions WHERE id = ?",
            (transcription_id,),
        ).fetchone()


def list_transcription_records(limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM transcriptions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()


def delete_transcription_record(transcription_id: str) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM transcriptions WHERE id = ?",
            (transcription_id,),
        )
        return cursor.rowcount


def update_transcription_record(
    transcription_id: str,
    text: str | None = None,
    language: str | None = None,
) -> int:
    fields = []
    values = []

    if text is not None:
        fields.append("text = ?")
        values.append(text)
    if language is not None:
        fields.append("language = ?")
        values.append(language)

    if not fields:
        return 0

    values.append(transcription_id)

    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE transcriptions SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        return cursor.rowcount


def create_summary_record(
    transcription_id: str,
    summary: str,
    method: str,
    original_length: int,
    summary_length: int,
    compression_ratio: float,
) -> None:
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO summaries (
                transcription_id, summary, method, original_length, summary_length, compression_ratio, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transcription_id,
                summary,
                method,
                original_length,
                summary_length,
                compression_ratio,
                datetime.now().isoformat(),
            ),
        )


def get_latest_summary_record(transcription_id: str) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT * FROM summaries
            WHERE transcription_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (transcription_id,),
        ).fetchone()


def list_summary_records(limit: int = 50) -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM summaries ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()


def get_summary_record(summary_id: int) -> sqlite3.Row | None:
    with get_connection() as connection:
        return connection.execute(
            "SELECT * FROM summaries WHERE id = ?",
            (summary_id,),
        ).fetchone()


def delete_summary_record(summary_id: int) -> int:
    with get_connection() as connection:
        cursor = connection.execute(
            "DELETE FROM summaries WHERE id = ?",
            (summary_id,),
        )
        return cursor.rowcount


def update_summary_record(summary_id: int, summary: str | None = None, method: str | None = None) -> int:
    fields = []
    values = []

    if summary is not None:
        fields.append("summary = ?")
        values.append(summary)
        fields.append("summary_length = ?")
        values.append(len(summary))
    if method is not None:
        fields.append("method = ?")
        values.append(method)

    if not fields:
        return 0

    values.append(summary_id)

    with get_connection() as connection:
        cursor = connection.execute(
            f"UPDATE summaries SET {', '.join(fields)} WHERE id = ?",
            values,
        )
        return cursor.rowcount
