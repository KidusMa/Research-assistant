import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Import our modules
from document_loader import DocumentLoader

logger = logging.getLogger(__name__)

@dataclass
class RetrievalResult:
    """Data class for retrieval results"""
    documents: List[Document]
    scores: List[float]
    query: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class AdvancedRAG:
    """Advanced RAG system with enhanced retrieval and reasoning"""
    
    def __init__(self, llm_provider: str = "groq", api_key: str = None):
        self.llm_provider = llm_provider
        self.api_key = api_key
        
        # Initialize LLM
        if llm_provider == "groq":
            self.llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name="llama3-8b-8192"
            )
        else:
            self.llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model="gpt-3.5-turbo"
            )
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self.vector_store = None
        
        # Advanced retrieval prompts
        self.query_expansion_prompt = ChatPromptTemplate.from_template("""
        You are an expert at expanding and refining search queries. Given a user query, create multiple search variations to improve retrieval.
        
        Original Query: {query}
        
        Create 3-5 search variations that:
        1. Use different terminology and synonyms
        2. Focus on different aspects of the query
        3. Include broader and narrower terms
        4. Use technical terms when appropriate
        
        Respond in JSON format:
        {{
            "expanded_queries": ["query1", "query2", "query3"],
            "reasoning": "explanation of why these queries were chosen"
        }}
        """)
        
        self.relevance_scoring_prompt = ChatPromptTemplate.from_template("""
        You are an expert at evaluating the relevance of documents to a query. Rate each document on a scale of 1-10.
        
        Query: {query}
        
        Document: {document}
        
        Rate the relevance (1-10) and explain why:
        {{
            "score": <1-10>,
            "reasoning": "explanation of relevance"
        }}
        """)
        
        self.multi_step_reasoning_prompt = ChatPromptTemplate.from_template("""
        You are an expert research assistant. Analyze the provided information and answer the query using multi-step reasoning.
        
        Query: {query}
        
        Retrieved Information:
        {context}
        
        Follow these steps:
        1. Identify the key information needed to answer the query
        2. Analyze the relevance and reliability of each source
        3. Synthesize information from multiple sources
        4. Identify any gaps or contradictions
        5. Formulate a comprehensive answer
        
        Provide your analysis in this format:
        
        ## Analysis
        [Step-by-step reasoning process]
        
        ## Answer
        [Comprehensive answer based on the analysis]
        
        ## Sources Used
        [List of most relevant sources]
        
        ## Confidence Level
        [High/Medium/Low] - [Explanation]
        
        ## Information Gaps
        [Any missing information or areas needing more research]
        """)
        
        self.parser = JsonOutputParser()
    
    def load_documents(self, sources: List[str]) -> List[Document]:
        """Load documents from multiple sources"""
        documents = self.document_loader.load_multiple_sources(sources)
        
        if not documents:
            logger.warning("No documents loaded from sources")
            return []
        
        # Split documents into chunks
        texts = self.text_splitter.split_documents(documents)
        logger.info(f"Processed {len(texts)} document chunks from {len(documents)} documents")
        
        return texts
    
    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents"""
        if not documents:
            raise ValueError("No documents provided")
        
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        logger.info(f"Created vector store with {len(documents)} documents")
        return self.vector_store
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query into multiple search variations"""
        try:
            chain = self.query_expansion_prompt | self.llm | self.parser
            result = chain.invoke({"query": query})
            return result.get("expanded_queries", [query])
        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            return [query]
    
    def retrieve_with_expansion(self, query: str, k: int = 10) -> RetrievalResult:
        """Retrieve documents using query expansion"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        # Expand query
        expanded_queries = self.expand_query(query)
        logger.info(f"Expanded query into {len(expanded_queries)} variations")
        
        # Retrieve from each expanded query
        all_documents = []
        all_scores = []
        
        for expanded_query in expanded_queries:
            # Get documents and scores
            docs_and_scores = self.vector_store.similarity_search_with_score(
                expanded_query, k=k//len(expanded_queries)
            )
            
            for doc, score in docs_and_scores:
                all_documents.append(doc)
                all_scores.append(score)
        
        # Remove duplicates while preserving order
        seen_content = set()
        unique_docs = []
        unique_scores = []
        
        for doc, score in zip(all_documents, all_scores):
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                unique_docs.append(doc)
                unique_scores.append(score)
                seen_content.add(content_hash)
        
        return RetrievalResult(
            documents=unique_docs[:k],
            scores=unique_scores[:k],
            query=query,
            metadata={"expanded_queries": expanded_queries}
        )
    
    def rerank_documents(self, query: str, retrieval_result: RetrievalResult) -> RetrievalResult:
        """Rerank documents using LLM-based relevance scoring"""
        reranked_docs = []
        reranked_scores = []
        
        for doc, original_score in zip(retrieval_result.documents, retrieval_result.scores):
            try:
                # Create relevance scoring chain
                scoring_chain = self.relevance_scoring_prompt | self.llm | self.parser
                
                result = scoring_chain.invoke({
                    "query": query,
                    "document": doc.page_content[:1000]  # Limit content for scoring
                })
                
                # Convert LLM score to float
                llm_score = float(result.get("score", 5)) / 10.0  # Normalize to 0-1
                
                # Combine with original similarity score
                combined_score = (original_score + llm_score) / 2
                
                reranked_docs.append(doc)
                reranked_scores.append(combined_score)
                
            except Exception as e:
                logger.error(f"Error scoring document: {e}")
                # Keep original score if scoring fails
                reranked_docs.append(doc)
                reranked_scores.append(original_score)
        
        # Sort by combined scores
        sorted_pairs = sorted(zip(reranked_docs, reranked_scores), 
                            key=lambda x: x[1], reverse=True)
        
        reranked_docs, reranked_scores = zip(*sorted_pairs)
        
        return RetrievalResult(
            documents=list(reranked_docs),
            scores=list(reranked_scores),
            query=query,
            metadata=retrieval_result.metadata
        )
    
    def multi_step_reasoning(self, query: str, documents: List[Document]) -> str:
        """Perform multi-step reasoning on retrieved documents"""
        if not documents:
            return "No relevant documents found to answer the query."
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}:\n{doc.page_content}\n")
        
        context = "\n".join(context_parts)
        
        try:
            chain = self.multi_step_reasoning_prompt | self.llm
            result = chain.invoke({
                "query": query,
                "context": context
            })
            return result.content
        except Exception as e:
            logger.error(f"Multi-step reasoning error: {e}")
            return f"Error in reasoning process: {e}"
    
    def advanced_research(self, query: str, documents: List[Document] = None) -> Dict[str, Any]:
        """Perform advanced research with enhanced retrieval and reasoning"""
        logger.info(f"Starting advanced research for query: {query}")
        
        # Step 1: Load documents if provided
        if documents is None:
            # Use existing vector store
            if self.vector_store is None:
                return {"error": "No documents loaded and no vector store available"}
        else:
            # Create vector store from provided documents
            self.create_vector_store(documents)
        
        # Step 2: Enhanced retrieval with query expansion
        retrieval_result = self.retrieve_with_expansion(query, k=15)
        logger.info(f"Retrieved {len(retrieval_result.documents)} documents")
        
        # Step 3: Rerank documents
        reranked_result = self.rerank_documents(query, retrieval_result)
        logger.info(f"Reranked {len(reranked_result.documents)} documents")
        
        # Step 4: Multi-step reasoning
        reasoning_result = self.multi_step_reasoning(query, reranked_result.documents[:8])
        
        return {
            "query": query,
            "retrieved_documents": len(retrieval_result.documents),
            "reranked_documents": len(reranked_result.documents),
            "top_documents": [
                {
                    "content": doc.page_content[:200] + "...",
                    "metadata": doc.metadata,
                    "score": score
                }
                for doc, score in zip(reranked_result.documents[:5], reranked_result.scores[:5])
            ],
            "reasoning_result": reasoning_result,
            "metadata": {
                "expanded_queries": retrieval_result.metadata.get("expanded_queries", []),
                "retrieval_scores": retrieval_result.scores[:5],
                "reranked_scores": reranked_result.scores[:5]
            }
        }
    
    def create_contextual_retriever(self) -> ContextualCompressionRetriever:
        """Create a contextual compression retriever for better relevance"""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        # Create base retriever
        base_retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )
        
        # Create compressor
        compressor_prompt = """Given the following question and context, extract any relevant text from the context that would help answer the question. If none of the context is relevant, return "No relevant information found."

Question: {question}
Context: {context}
Relevant text:"""

        compressor = LLMChainExtractor.from_llm(self.llm, prompt=compressor_prompt)
        
        # Create contextual compression retriever
        contextual_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        
        return contextual_retriever 