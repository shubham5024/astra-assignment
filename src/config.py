"""Application configuration management using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Intelligent Knowledge Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project metadata
    PROJECT_NAME: str = "Intelligent Knowledge Assistant"
    VERSION: str = "0.1.0"
    API_PREFIX: str = "/api/v1"

    # LLM & Embedding Settings
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API Key")
    OPENAI_BASE_URL: str = Field(default="https://api.openai.com/v1", description="OpenAI API Base URL")
    LLM_MODEL: str = Field(default="gpt-4o-mini", description="Default LLM for generation & agent reasoning")
    EMBEDDING_MODEL: str = Field(default="text-embedding-3-small", description="OpenAI embedding model")
    EMBEDDING_DIMENSIONS: int = Field(default=1536, description="Embedding vector dimensionality")

    # Document Chunking Settings
    CHUNK_SIZE_TOKENS: int = Field(default=650, description="Target chunk size in tokens")
    CHUNK_OVERLAP_TOKENS: int = Field(default=100, description="Token overlap between consecutive chunks")

    # Hybrid Search & RRF Parameters
    TOP_K_RETRIEVAL: int = Field(default=5, description="Number of top chunks returned to generator")
    SIMILARITY_THRESHOLD: float = Field(default=0.35, description="Cosine similarity cutoff to prevent hallucination")
    RRF_K: int = Field(default=60, description="Reciprocal Rank Fusion constant (k)")
    DENSE_WEIGHT: float = Field(default=0.6, description="RRF weight for dense vector search")
    SPARSE_WEIGHT: float = Field(default=0.4, description="RRF weight for sparse BM25 search")

    # Agentic Engine Settings
    AGENT_MAX_ITERATIONS: int = Field(default=5, description="Max reasoning loops for ReAct agent")
    AGENT_TEMPERATURE: float = Field(default=0.0, description="Sampling temperature for deterministic tool calls")

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DOCUMENTS_DIR: Path = Field(default=Path("data/documents"))
    STORAGE_DIR: Path = Field(default=Path("data/storage"))

    # Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    UI_PORT: int = 8501

    @property
    def resolved_docs_dir(self) -> Path:
        if self.DOCUMENTS_DIR.is_absolute():
            return self.DOCUMENTS_DIR
        return self.BASE_DIR / self.DOCUMENTS_DIR

    @property
    def resolved_storage_dir(self) -> Path:
        if self.STORAGE_DIR.is_absolute():
            return self.STORAGE_DIR
        return self.BASE_DIR / self.STORAGE_DIR


settings = Settings()
