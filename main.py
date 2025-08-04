import os
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# Web search and data processing
from duckduckgo_search import DDGS
import wikipedia
import arxiv
import requests
from bs4 import BeautifulSoup
import re

# Vector store and embeddings
import chromadb
from sentence_transformers import SentenceTransformer

# Streamlit for UI
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ResearchQuery:
    """Data class for research queries"""
    query: str
    context: Optional[str] = None
    sources: List[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.sources is None:
            self.sources = []
        if self.timestamp is None:
            self.timestamp = datetime.now()

class DocumentProcessor:
    """Handles document processing and vector store management"""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        self.embedding_model = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        
    def process_documents(self, documents: List[Document]) -> Chroma:
        """Process documents and create vector store"""
        if not documents:
            return None
            
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = Chroma.from_documents(
            documents=texts,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        logger.info(f"Processed {len(texts)} document chunks")
        return self.vector_store
    
    def add_documents(self, documents: List[Document]):
        """Add new documents to existing vector store"""
        if self.vector_store is None:
            return self.process_documents(documents)
            
        texts = self.text_splitter.split_documents(documents)
        self.vector_store.add_documents(texts)
        logger.info(f"Added {len(texts)} new document chunks")
        
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=k)

class WebSearcher:
    """Handles web search functionality"""
    
    def __init__(self):
        self.ddgs = DDGS()
        
    def search_web(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search the web using DuckDuckGo"""
        try:
            results = []
            search_results = self.ddgs.text(query, max_results=max_results)
            
            for result in search_results:
                results.append({
                    'title': result.get('title', ''),
                    'link': result.get('link', ''),
                    'snippet': result.get('body', ''),
                    'source': 'web_search'
                })
                
            return results
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return []
    
    def search_wikipedia(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search Wikipedia"""
        try:
            results = []
            # Search for Wikipedia pages
            search_results = wikipedia.search(query, results=max_results)
            
            for title in search_results:
                try:
                    page = wikipedia.page(title)
                    results.append({
                        'title': page.title,
                        'link': page.url,
                        'snippet': page.summary[:500] + "...",
                        'source': 'wikipedia'
                    })
                except:
                    continue
                    
            return results
        except Exception as e:
            logger.error(f"Wikipedia search error: {e}")
            return []
    
    def search_arxiv(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """Search arXiv for academic papers"""
        try:
            results = []
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            for result in search.results():
                results.append({
                    'title': result.title,
                    'link': result.entry_id,
                    'snippet': result.summary[:500] + "...",
                    'source': 'arxiv',
                    'authors': [author.name for author in result.authors],
                    'published': result.published.strftime("%Y-%m-%d")
                })
                
            return results
        except Exception as e:
            logger.error(f"arXiv search error: {e}")
            return []

class ResearchAgent:
    """Main research agent with multi-step reasoning"""
    
    def __init__(self, llm_provider: str = "groq", api_key: str = None):
        self.llm_provider = llm_provider
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        
        # Initialize LLM
        if llm_provider == "groq":
            self.llm = ChatGroq(
                groq_api_key=self.groq.api_key,
                model_name="llama3-8b-8192"
            )
        else:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model="gpt-3.5-turbo"
            )
        
        # Initialize components
        self.document_processor = DocumentProcessor()
        self.web_searcher = WebSearcher()
        
        # Research planning prompt
        self.research_planning_prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Given the user's query, create a comprehensive research plan.
        
        User Query: {query}
        
        Analyze the query and create a research plan with the following structure:
        1. Key research questions to answer
        2. Types of sources to search (web, academic papers, etc.)
        3. Specific search terms and strategies
        4. Information gaps that need to be filled
        
        Respond in JSON format:
        {{
            "research_questions": ["question1", "question2"],
            "search_strategies": ["strategy1", "strategy2"],
            "source_types": ["web", "academic", "wikipedia"],
            "search_terms": ["term1", "term2"],
            "information_gaps": ["gap1", "gap2"]
        }}
        """)
        
        # Information synthesis prompt
        self.synthesis_prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Synthesize the gathered information into a comprehensive answer.
        
        Original Query: {query}
        
        Gathered Information:
        {information}
        
        Create a comprehensive, well-structured answer that:
        1. Directly addresses the user's query
        2. Synthesizes information from multiple sources
        3. Provides clear, accurate information
        4. Cites sources appropriately
        5. Identifies any remaining information gaps
        
        Format your response as:
        ## Answer
        [Comprehensive answer]
        
        ## Sources
        [List of sources used]
        
        ## Information Gaps
        [Any remaining gaps or areas needing more research]
        """)
        
        self.parser = JsonOutputParser()
    
    def plan_research(self, query: str) -> Dict[str, Any]:
        """Create a research plan for the query"""
        try:
            chain = self.research_planning_prompt | self.llm | self.parser
            result = chain.invoke({"query": query})
            return result
        except Exception as e:
            logger.error(f"Research planning error: {e}")
            return {
                "research_questions": [query],
                "search_strategies": ["web_search"],
                "source_types": ["web"],
                "search_terms": [query],
                "information_gaps": []
            }
    
    def gather_information(self, query: str, research_plan: Dict[str, Any]) -> List[Dict[str, str]]:
        """Gather information from multiple sources"""
        all_results = []
        
        # Web search
        if "web" in research_plan.get("source_types", []):
            for search_term in research_plan.get("search_terms", [query]):
                web_results = self.web_searcher.search_web(search_term)
                all_results.extend(web_results)
        
        # Wikipedia search
        if "wikipedia" in research_plan.get("source_types", []):
            for search_term in research_plan.get("search_terms", [query]):
                wiki_results = self.web_searcher.search_wikipedia(search_term)
                all_results.extend(wiki_results)
        
        # Academic search
        if "academic" in research_plan.get("source_types", []):
            for search_term in research_plan.get("search_terms", [query]):
                arxiv_results = self.web_searcher.search_arxiv(search_term)
                all_results.extend(arxiv_results)
        
        # Remove duplicates based on title
        seen_titles = set()
        unique_results = []
        for result in all_results:
            if result['title'] not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result['title'])
        
        return unique_results
    
    def synthesize_answer(self, query: str, information: List[Dict[str, str]]) -> str:
        """Synthesize gathered information into a comprehensive answer"""
        if not information:
            return "I couldn't find relevant information to answer your query. Please try rephrasing your question or providing more specific details."
        
        # Format information for the LLM
        info_text = "\n\n".join([
            f"Source: {item['source']}\nTitle: {item['title']}\nContent: {item['snippet']}\nLink: {item['link']}"
            for item in information
        ])
        
        try:
            chain = self.synthesis_prompt | self.llm
            result = chain.invoke({
                "query": query,
                "information": info_text
            })
            return result.content
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return f"Error synthesizing information: {e}"
    
    def research(self, query: str) -> Dict[str, Any]:
        """Main research method"""
        logger.info(f"Starting research for query: {query}")
        
        # Step 1: Plan research
        research_plan = self.plan_research(query)
        logger.info(f"Research plan created: {research_plan}")
        
        # Step 2: Gather information
        information = self.gather_information(query, research_plan)
        logger.info(f"Gathered {len(information)} information sources")
        
        # Step 3: Synthesize answer
        answer = self.synthesize_answer(query, information)
        
        return {
            "query": query,
            "research_plan": research_plan,
            "information_sources": information,
            "answer": answer,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main function to run the research assistant"""
    st.set_page_config(
        page_title="RAG Research Assistant",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 RAG Research Assistant")
    st.markdown("An intelligent research assistant powered by RAG (Retrieval-Augmented Generation)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["groq", "openai"],
            help="Choose your preferred LLM provider"
        )
        
        api_key = st.text_input(
            "API Key",
            type="password",
            help=f"Enter your {llm_provider.upper()} API key"
        )
        
        if not api_key:
            st.warning("Please enter your API key to continue")
            return
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Research Query")
        query = st.text_area(
            "Enter your research question:",
            height=100,
            placeholder="e.g., What are the latest developments in quantum computing?"
        )
        
        if st.button("🔍 Start Research", type="primary"):
            if query.strip():
                with st.spinner("Researching..."):
                    try:
                        # Initialize research agent
                        agent = ResearchAgent(llm_provider=llm_provider, api_key=api_key)
                        
                        # Conduct research
                        result = agent.research(query)
                        
                        # Display results
                        st.success("Research completed!")
                        
                        # Display answer
                        st.markdown("## Research Results")
                        st.markdown(result["answer"])
                        
                        # Display research plan
                        with st.expander("📋 Research Plan"):
                            st.json(result["research_plan"])
                        
                        # Display sources
                        with st.expander("📚 Information Sources"):
                            for i, source in enumerate(result["information_sources"], 1):
                                st.markdown(f"**{i}. {source['title']}**")
                                st.markdown(f"*Source: {source['source']}*")
                                st.markdown(f"*Link: {source['link']}*")
                                st.markdown(f"{source['snippet']}")
                                st.divider()
                        
                    except Exception as e:
                        st.error(f"Research failed: {str(e)}")
                        logger.error(f"Research error: {e}")
            else:
                st.warning("Please enter a research query")
    
    with col2:
        st.header("Recent Queries")
        st.info("Your recent research queries will appear here")
        
        st.header("Tips")
        st.markdown("""
        - Be specific in your queries
        - Use keywords for better results
        - The assistant searches multiple sources
        - Results are synthesized for accuracy
        """)

if __name__ == "__main__":
    main()
