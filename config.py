import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    nextjs_url: str
    callback_secret: str
    allowed_origins: list[str]


def load_settings() -> Settings:
    secret = os.environ.get("CALLBACK_SECRET", "").strip()
    if not secret:
        raise RuntimeError("Set CALLBACK_SECRET before starting the API.")

    nextjs_url = os.environ.get("NEXTJS_URL", "http://localhost:3000").strip().rstrip("/")
    raw_origins = os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000")
    origins = [
        origin.strip().rstrip("/")
        for origin in raw_origins.split(",")
        if origin.strip()
    ]

    return Settings(
        nextjs_url=nextjs_url,
        callback_secret=secret,
        allowed_origins=origins or ["http://localhost:3000"],
    )


settings = load_settings()
