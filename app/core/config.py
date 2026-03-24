from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "AssistEduc API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Uploads
    UPLOAD_DIR: str = "uploads"
    MAX_UPLOAD_SIZE: int = 100 * 1024 * 1024  # 100 MB
    ALLOWED_AUDIO_EXTENSIONS: list = [".mp3", ".wav", ".m4a", ".ogg", ".mp4"]

settings = Settings()