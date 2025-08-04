#!/usr/bin/env python3
"""
Test script to verify RAG Research Assistant installation

This script checks that all required dependencies are installed and working.
"""

import sys
import importlib
from typing import List, Dict, Tuple

def test_imports() -> List[Tuple[str, bool, str]]:
    """Test if all required packages can be imported"""
    required_packages = [
        # Core LangChain
        ("langchain", "LangChain core"),
        ("langchain_core", "LangChain core components"),
        ("langchain_community", "LangChain community components"),
        ("langchain_groq", "Groq integration"),
        ("langchain_openai", "OpenAI integration"),
        
        # Vector stores and embeddings
        ("chromadb", "ChromaDB vector store"),
        ("sentence_transformers", "Sentence transformers for embeddings"),
        ("langchain_community.vectorstores", "Vector store components"),
        ("langchain_community.embeddings", "Embedding components"),
        
        # Document processing
        ("langchain.text_splitter", "Text splitting"),
        ("langchain.schema", "LangChain schemas"),
        
        # Web search and data processing
        ("duckduckgo_search", "DuckDuckGo search"),
        ("wikipedia", "Wikipedia API"),
        ("arxiv", "ArXiv API"),
        ("requests", "HTTP requests"),
        ("bs4", "BeautifulSoup for HTML parsing"),
        
        # Vector operations
        ("numpy", "NumPy for numerical operations"),
        ("faiss", "FAISS for vector operations"),
        
        # UI
        ("streamlit", "Streamlit for web interface"),
        
        # Additional utilities
        ("tiktoken", "Token counting"),
        ("python_dotenv", "Environment variable management"),
    ]
    
    results = []
    
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            results.append((package, True, f"✅ {description} - OK"))
        except ImportError as e:
            results.append((package, False, f"❌ {description} - FAILED: {e}"))
        except Exception as e:
            results.append((package, False, f"⚠️ {description} - ERROR: {e}"))
    
    return results

def test_basic_functionality():
    """Test basic functionality without API keys"""
    print("\n🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        # Test document loader
        from document_loader import DocumentLoader
        loader = DocumentLoader()
        print("✅ DocumentLoader - OK")
        
        # Test configuration
        from config import RAGConfig
        config = RAGConfig()
        print("✅ RAGConfig - OK")
        
        # Test embeddings (this might take a moment)
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            print("✅ HuggingFace Embeddings - OK")
        except Exception as e:
            print(f"⚠️ HuggingFace Embeddings - WARNING: {e}")
        
        # Test text splitter
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        print("✅ Text Splitter - OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_api_connectivity():
    """Test API connectivity (if keys are available)"""
    print("\n🔗 Testing API Connectivity")
    print("=" * 40)
    
    import os
    
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if not groq_key and not openai_key:
        print("ℹ️  No API keys found. Skipping API connectivity tests.")
        print("   Set GROQ_API_KEY or OPENAI_API_KEY to test LLM connectivity.")
        return True
    
    results = []
    
    # Test Groq
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(groq_api_key=groq_key, model_name="llama3-8b-8192")
            # Test with a simple prompt
            response = llm.invoke("Say 'Hello, World!'")
            if response and response.content:
                results.append(("Groq API", True, "✅ Connected successfully"))
            else:
                results.append(("Groq API", False, "❌ No response received"))
        except Exception as e:
            results.append(("Groq API", False, f"❌ Connection failed: {e}"))
    else:
        results.append(("Groq API", False, "ℹ️ No API key provided"))
    
    # Test OpenAI
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-3.5-turbo")
            # Test with a simple prompt
            response = llm.invoke("Say 'Hello, World!'")
            if response and response.content:
                results.append(("OpenAI API", True, "✅ Connected successfully"))
            else:
                results.append(("OpenAI API", False, "❌ No response received"))
        except Exception as e:
            results.append(("OpenAI API", False, f"❌ Connection failed: {e}"))
    else:
        results.append(("OpenAI API", False, "ℹ️ No API key provided"))
    
    # Print results
    for service, success, message in results:
        print(f"{message}")
    
    return any(success for _, success, _ in results)

def main():
    """Run all tests"""
    print("🔍 RAG Research Assistant - Installation Test")
    print("=" * 60)
    
    # Test imports
    print("📦 Testing Package Imports")
    print("=" * 30)
    
    import_results = test_imports()
    
    # Count results
    successful_imports = sum(1 for _, success, _ in import_results if success)
    total_imports = len(import_results)
    
    # Print results
    for _, _, message in import_results:
        print(message)
    
    print(f"\n📊 Import Results: {successful_imports}/{total_imports} packages imported successfully")
    
    # Test basic functionality
    basic_success = test_basic_functionality()
    
    # Test API connectivity
    api_success = test_api_connectivity()
    
    # Summary
    print("\n📋 Test Summary")
    print("=" * 30)
    
    if successful_imports == total_imports:
        print("✅ All required packages imported successfully")
    else:
        print(f"⚠️ {total_imports - successful_imports} packages failed to import")
        print("   Run: pip install -r requirements.txt")
    
    if basic_success:
        print("✅ Basic functionality working")
    else:
        print("❌ Basic functionality failed")
    
    if api_success:
        print("✅ API connectivity working")
    else:
        print("ℹ️ API connectivity not tested (no API keys)")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if successful_imports < total_imports:
        print("- Run: pip install -r requirements.txt")
    
    if not basic_success:
        print("- Check Python version (3.8+ required)")
        print("- Verify all dependencies are installed")
    
    if not api_success:
        print("- Set GROQ_API_KEY or OPENAI_API_KEY environment variable")
        print("- Test with: python example_usage.py")
    
    print("\n🚀 To run the research assistant:")
    print("  streamlit run main.py")

if __name__ == "__main__":
    main() 