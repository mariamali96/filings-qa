"""Settings read from the environment (see .env.example)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so the project has no extra dependency for it."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    sec_user_agent: str
    database_url: str
    llm_provider: str
    anthropic_model: str
    embeddings: str
    embedding_model: str
    data_dir: Path


def get_settings() -> Settings:
    _load_dotenv()
    return Settings(
        sec_user_agent=os.getenv("SEC_USER_AGENT", ""),
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://filings:filings@localhost:5432/filings"
        ),
        llm_provider=os.getenv("LLM_PROVIDER", "anthropic"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
        embeddings=os.getenv("EMBEDDINGS", "local"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        data_dir=Path(os.getenv("DATA_DIR", "data")),
    )
