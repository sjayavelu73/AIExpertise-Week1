import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file
base_dir = Path(__file__).parent.parent  # src/
load_dotenv(base_dir / ".env")

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in .env")
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))

settings = Settings()
