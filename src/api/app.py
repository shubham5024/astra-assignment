"""FastAPI Application Entrypoint with Lifespan State Management."""

from contextlib import asynccontextmanager
from typing import Dict, Any, Optional
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router as api_router
from src.embeddings.openai_embedder import OpenAIEmbedder
from src.vector_store.store import DenseVectorStore
from src.search.bm25 import BM25SearchEngine
from src.search.hybrid import HybridSearchEngine
from src.rag.generator import LLMGenerator
from src.rag.pipeline import SimpleRAGPipeline
from src.agent.tools import AgentToolRegistry
from src.agent.orchestrator import AgentOrchestrator
from src.agent.router import QueryRouter
from src.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Global runtime services container
_GLOBAL_SERVICES: Dict[str, Any] = {}


def init_services() -> Dict[str, Any]:
    """Initialize or reload all RAG and Agentic runtime services."""
    logger.info("Initializing knowledge assistant services...")
    storage_dir = settings.resolved_storage_dir

    embedder = OpenAIEmbedder()
    vector_store = DenseVectorStore(storage_dir=storage_dir)
    vector_store.load()

    bm25_engine = BM25SearchEngine(storage_dir=storage_dir)
    bm25_engine.load()

    hybrid_searcher = HybridSearchEngine(
        vector_store=vector_store,
        bm25_engine=bm25_engine,
        embedder=embedder,
    )

    generator = LLMGenerator()
    simple_rag = SimpleRAGPipeline(hybrid_searcher=hybrid_searcher, generator=generator)
    tool_registry = AgentToolRegistry(search_engine=hybrid_searcher)
    agent_orchestrator = AgentOrchestrator(tool_registry=tool_registry, generator=generator)
    router = QueryRouter()

    services = {
        "embedder": embedder,
        "vector_store": vector_store,
        "bm25_engine": bm25_engine,
        "hybrid_searcher": hybrid_searcher,
        "generator": generator,
        "simple_rag": simple_rag,
        "tool_registry": tool_registry,
        "agent_orchestrator": agent_orchestrator,
        "router": router,
    }

    _GLOBAL_SERVICES.clear()
    _GLOBAL_SERVICES.update(services)
    return _GLOBAL_SERVICES


def get_app_services() -> Dict[str, Any]:
    """Access global initialized services."""
    if not _GLOBAL_SERVICES:
        return init_services()
    return _GLOBAL_SERVICES


def reload_app_services() -> Dict[str, Any]:
    """Reload indexes from disk after ingestion."""
    return init_services()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event to bootstrap vector indexes and models on startup."""
    init_services()
    yield
    logger.info("Shutting down Intelligent Knowledge Assistant API.")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description="Production-oriented Intelligent Knowledge Assistant combining Simple RAG and Multi-Step Agentic workflows.",
        lifespan=lifespan,
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount API routes
    app.include_router(api_router, prefix=settings.API_PREFIX)

    @app.get("/", summary="Root Health Check")
    async def root():
        return {
            "name": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "docs_url": "/docs",
            "api_prefix": settings.API_PREFIX,
        }

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host=settings.API_HOST, port=settings.API_PORT, reload=True)
