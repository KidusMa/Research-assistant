#!/usr/bin/env python3
"""
Example usage of the RAG Research Assistant

This script demonstrates how to use the research assistant programmatically
for various research tasks.
"""

import os
import logging
from typing import List, Dict, Any

# Import our modules
from main import ResearchAgent
from advanced_rag import AdvancedRAG
from document_loader import DocumentLoader
from config import RAGConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def basic_research_example():
    """Example of basic research using web search"""
    print("🔍 Basic Research Example")
    print("=" * 50)
    
    # Initialize research agent
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set GROQ_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    agent = ResearchAgent(llm_provider="groq", api_key=api_key)
    
    # Perform research
    query = "What are the latest developments in quantum computing?"
    print(f"Query: {query}")
    
    try:
        result = agent.research(query)
        
        print("\n📋 Research Plan:")
        print(json.dumps(result["research_plan"], indent=2))
        
        print(f"\n📚 Information Sources ({len(result['information_sources'])} found):")
        for i, source in enumerate(result["information_sources"][:3], 1):
            print(f"{i}. {source['title']} ({source['source']})")
        
        print("\n💡 Answer:")
        print(result["answer"])
        
    except Exception as e:
        print(f"❌ Research failed: {e}")

def advanced_rag_example():
    """Example of advanced RAG with custom documents"""
    print("\n🔬 Advanced RAG Example")
    print("=" * 50)
    
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Please set GROQ_API_KEY or OPENAI_API_KEY environment variable")
        return
    
    rag = AdvancedRAG(llm_provider="groq", api_key=api_key)
    
    # Example: Load documents from URLs (you can replace with local files)
    sample_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://en.wikipedia.org/wiki/Machine_learning"
    ]
    
    print("📥 Loading documents...")
    try:
        documents = rag.load_documents(sample_urls)
        print(f"✅ Loaded {len(documents)} document chunks")
        
        # Create vector store
        rag.create_vector_store(documents)
        
        # Perform advanced research
        query = "How does machine learning relate to artificial intelligence?"
        print(f"\nQuery: {query}")
        
        result = rag.advanced_research(query)
        
        print(f"\n📊 Retrieved {result['retrieved_documents']} documents")
        print(f"🔄 Reranked {result['reranked_documents']} documents")
        
        print("\n📋 Top Documents:")
        for i, doc in enumerate(result["top_documents"][:3], 1):
            print(f"{i}. Score: {doc['score']:.3f}")
            print(f"   Content: {doc['content'][:100]}...")
        
        print("\n🧠 Reasoning Result:")
        print(result["reasoning_result"])
        
    except Exception as e:
        print(f"❌ Advanced RAG failed: {e}")

def document_processing_example():
    """Example of document processing capabilities"""
    print("\n📄 Document Processing Example")
    print("=" * 50)
    
    loader = DocumentLoader()
    
    # Example sources (replace with actual files/URLs)
    example_sources = [
        # "path/to/local/document.pdf",
        # "https://arxiv.org/abs/2303.08774",
        # "./research_documents/"
    ]
    
    if not example_sources or not any(os.path.exists(src) for src in example_sources if not src.startswith('http')):
        print("ℹ️  No local documents found. Create some test documents or use URLs.")
        print("Example sources you can try:")
        print("- Local PDF files")
        print("- Wikipedia URLs")
        print("- ArXiv paper URLs")
        return
    
    try:
        documents = loader.load_multiple_sources(example_sources)
        print(f"✅ Processed {len(documents)} documents")
        
        for i, doc in enumerate(documents[:3], 1):
            print(f"\nDocument {i}:")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Type: {doc.metadata.get('type', 'Unknown')}")
            print(f"  Content preview: {doc.page_content[:100]}...")
            
    except Exception as e:
        print(f"❌ Document processing failed: {e}")

def configuration_example():
    """Example of configuration usage"""
    print("\n⚙️ Configuration Example")
    print("=" * 50)
    
    # Load configuration from environment
    config = RAGConfig.from_env()
    
    print("Current Configuration:")
    config_dict = config.to_dict()
    for key, value in config_dict.items():
        print(f"  {key}: {value}")
    
    # Example of custom configuration
    custom_config = RAGConfig(
        llm_provider="openai",
        retrieval_k=15,
        max_web_results=8,
        reranking_enabled=True
    )
    
    print(f"\nCustom Configuration:")
    print(f"  LLM Provider: {custom_config.llm_provider}")
    print(f"  Retrieval K: {custom_config.retrieval_k}")
    print(f"  Max Web Results: {custom_config.max_web_results}")
    print(f"  Reranking Enabled: {custom_config.reranking_enabled}")

def main():
    """Run all examples"""
    print("🚀 RAG Research Assistant - Example Usage")
    print("=" * 60)
    
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No API key found. Set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        print("Some examples may not work without an API key.")
        print()
    
    # Run examples
    try:
        basic_research_example()
        advanced_rag_example()
        document_processing_example()
        configuration_example()
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
    
    print("\n✅ Examples completed!")
    print("\nTo run the web interface:")
    print("  streamlit run main.py")

if __name__ == "__main__":
    import json
    main() 