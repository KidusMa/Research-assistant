import os
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for the RAG Research Assistant"""
    
    # LLM Configuration
    llm_provider: str = "groq"  # "groq" or "openai"
    groq_model: str = "llama3-8b-8192"
    openai_model: str = "gpt-3.5-turbo"
    
    # Embedding Configuration
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_documents: int = 1000
    
    # Retrieval Configuration
    retrieval_k: int = 10
    max_expanded_queries: int = 5
    reranking_enabled: bool = True
    
    # Web Search Configuration
    max_web_results: int = 5
    max_wikipedia_results: int = 3
    max_arxiv_results: int = 3
    
    # Vector Store Configuration
    vector_store_persist_dir: str = "./chroma_db"
    similarity_threshold: float = 0.7
    
    # UI Configuration
    streamlit_port: int = 8501
    streamlit_host: str = "localhost"
    
    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "rag_assistant.log"
    
    @classmethod
    def from_env(cls) -> 'RAGConfig':
        """Create configuration from environment variables"""
        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "groq"),
            groq_model=os.getenv("GROQ_MODEL", "llama3-8b-8192"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            max_documents=int(os.getenv("MAX_DOCUMENTS", "1000")),
            retrieval_k=int(os.getenv("RETRIEVAL_K", "10")),
            max_expanded_queries=int(os.getenv("MAX_EXPANDED_QUERIES", "5")),
            reranking_enabled=os.getenv("RERANKING_ENABLED", "true").lower() == "true",
            max_web_results=int(os.getenv("MAX_WEB_RESULTS", "5")),
            max_wikipedia_results=int(os.getenv("MAX_WIKIPEDIA_RESULTS", "3")),
            max_arxiv_results=int(os.getenv("MAX_ARXIV_RESULTS", "3")),
            vector_store_persist_dir=os.getenv("VECTOR_STORE_DIR", "./chroma_db"),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7")),
            streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
            streamlit_host=os.getenv("STREAMLIT_HOST", "localhost"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_file=os.getenv("LOG_FILE", "rag_assistant.log")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "llm_provider": self.llm_provider,
            "groq_model": self.groq_model,
            "openai_model": self.openai_model,
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "max_documents": self.max_documents,
            "retrieval_k": self.retrieval_k,
            "max_expanded_queries": self.max_expanded_queries,
            "reranking_enabled": self.reranking_enabled,
            "max_web_results": self.max_web_results,
            "max_wikipedia_results": self.max_wikipedia_results,
            "max_arxiv_results": self.max_arxiv_results,
            "vector_store_persist_dir": self.vector_store_persist_dir,
            "similarity_threshold": self.similarity_threshold,
            "streamlit_port": self.streamlit_port,
            "streamlit_host": self.streamlit_host,
            "log_level": self.log_level,
            "log_file": self.log_file
        }

# Default configuration
DEFAULT_CONFIG = RAGConfig()

# Environment-specific configurations
DEVELOPMENT_CONFIG = RAGConfig(
    log_level="DEBUG",
    retrieval_k=5,
    max_web_results=3
)

PRODUCTION_CONFIG = RAGConfig(
    log_level="WARNING",
    retrieval_k=15,
    max_web_results=8,
    reranking_enabled=True
) 