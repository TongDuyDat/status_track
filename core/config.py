import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    APP_NAME: str = "FastAPI Redis Queue"
    VERSION: str = "1.0.0"
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))

    # Memory Management
    RAM_THRESHOLD: float = float(os.getenv("RAM_THRESHOLD", 0.85))  # 85%
    GPU_THRESHOLD: float = float(os.getenv("GPU_THRESHOLD", 0.90))  # 90%
    MAX_QUEUE_SIZE: int = int(os.getenv("MAX_QUEUE_SIZE", 1000))

    # Batch Manager Limits
    TRUCK_MAX_BATCH: int = int(os.getenv("TRUCK_MAX_BATCH", 32))
    TEXT_MAX_BATCH: int = int(os.getenv("TEXT_MAX_BATCH", 16))
    OCR_MAX_BATCH: int = int(os.getenv("OCR_MAX_BATCH", 64))
    
    # Redis configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    RESULT_EXPIRE: int = 300  # ✅ Add this
    
    @property
    def REDIS_URL(self) -> str:
        """Build Redis URL from components"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    class Config:
        env_file = ".env"
settings = Settings()
