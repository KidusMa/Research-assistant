import os
import requests
from typing import List, Optional
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import PyPDF2
import io

from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredWordDocumentLoader,
    CSVLoader
)

class DocumentLoader:
    """Handles loading documents from various sources and formats"""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': self._load_text,
            '.pdf': self._load_pdf,
            '.md': self._load_markdown,
            '.docx': self._load_word,
            '.csv': self._load_csv,
            '.html': self._load_html,
            '.htm': self._load_html
        }
    
    def load_from_file(self, file_path: str) -> List[Document]:
        """Load documents from a file"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in self.supported_extensions:
            return self.supported_extensions[file_ext](file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
    
    def load_from_directory(self, directory_path: str) -> List[Document]:
        """Load all supported documents from a directory"""
        documents = []
        
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                if file_ext in self.supported_extensions:
                    try:
                        docs = self.load_from_file(file_path)
                        documents.extend(docs)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_from_url(self, url: str) -> List[Document]:
        """Load content from a URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse the URL to determine content type
            parsed_url = urlparse(url)
            path = parsed_url.path.lower()
            
            if path.endswith('.pdf'):
                return self._load_pdf_from_url(url)
            elif path.endswith(('.html', '.htm')) or 'text/html' in response.headers.get('content-type', ''):
                return self._load_html_from_url(url, response.text)
            else:
                # Treat as plain text
                return [Document(
                    page_content=response.text,
                    metadata={"source": url, "type": "web_content"}
                )]
                
        except Exception as e:
            raise Exception(f"Error loading from URL {url}: {e}")
    
    def _load_text(self, file_path: str) -> List[Document]:
        """Load text files"""
        loader = TextLoader(file_path)
        return loader.load()
    
    def _load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF files"""
        loader = PyPDFLoader(file_path)
        return loader.load()
    
    def _load_pdf_from_url(self, url: str) -> List[Document]:
        """Load PDF from URL"""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            documents = []
            for page_num, page in enumerate(pdf_reader.pages):
                text = page.extract_text()
                documents.append(Document(
                    page_content=text,
                    metadata={
                        "source": url,
                        "page": page_num + 1,
                        "type": "pdf"
                    }
                ))
            
            return documents
        except Exception as e:
            raise Exception(f"Error loading PDF from URL: {e}")
    
    def _load_markdown(self, file_path: str) -> List[Document]:
        """Load Markdown files"""
        loader = UnstructuredMarkdownLoader(file_path)
        return loader.load()
    
    def _load_word(self, file_path: str) -> List[Document]:
        """Load Word documents"""
        loader = UnstructuredWordDocumentLoader(file_path)
        return loader.load()
    
    def _load_csv(self, file_path: str) -> List[Document]:
        """Load CSV files"""
        loader = CSVLoader(file_path)
        return loader.load()
    
    def _load_html(self, file_path: str) -> List[Document]:
        """Load HTML files"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
        
        return [Document(
            page_content=text,
            metadata={"source": file_path, "type": "html"}
        )]
    
    def _load_html_from_url(self, url: str, html_content: str) -> List[Document]:
        """Load HTML content from URL"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return [Document(
            page_content=text,
            metadata={"source": url, "type": "html"}
        )]
    
    def load_multiple_sources(self, sources: List[str]) -> List[Document]:
        """Load documents from multiple sources (files, directories, URLs)"""
        all_documents = []
        
        for source in sources:
            try:
                if source.startswith(('http://', 'https://')):
                    # URL
                    docs = self.load_from_url(source)
                elif os.path.isfile(source):
                    # File
                    docs = self.load_from_file(source)
                elif os.path.isdir(source):
                    # Directory
                    docs = self.load_from_directory(source)
                else:
                    print(f"Invalid source: {source}")
                    continue
                
                all_documents.extend(docs)
                
            except Exception as e:
                print(f"Error loading source {source}: {e}")
        
        return all_documents 