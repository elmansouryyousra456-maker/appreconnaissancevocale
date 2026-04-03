from pathlib import Path
import contextlib
import wave


def get_audio_duration(file_path: str | Path) -> float | None:
    path = Path(file_path)
    if path.suffix.lower() != ".wav":
        return None

    try:
        with contextlib.closing(wave.open(str(path), "rb")) as audio_file:
            frame_rate = audio_file.getframerate()
            frame_count = audio_file.getnframes()
            if frame_rate <= 0:
                return None
            return round(frame_count / float(frame_rate), 2)
    except (wave.Error, FileNotFoundError, OSError):
        return None
