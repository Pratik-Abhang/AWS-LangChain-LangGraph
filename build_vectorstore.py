"""
Build FAISS vectorstore from PDF document for RAG system.
"""
import os
import logging
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_PDF_PATH = os.getenv("PDF_PATH", r"C:\Users\Admin\Desktop\LangChain\data\AWS_Certified AI Practitioner_ Slides.pdf")
DEFAULT_INDEX_PATH = os.getenv("INDEX_PATH", "./vectorstore/faiss_index")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_pdf_path(pdf_path: str) -> bool:
    """Validate if PDF file exists and is readable."""
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found: {pdf_path}")
        return False
    if not pdf_path.lower().endswith('.pdf'):
        logger.error(f"File is not a PDF: {pdf_path}")
        return False
    return True

def build_vectorstore(pdf_path: str = DEFAULT_PDF_PATH, index_path: str = DEFAULT_INDEX_PATH) -> bool:
    """
    Build FAISS vectorstore from PDF document.
    
    Args:
        pdf_path: Path to the PDF file
        index_path: Path to save the vectorstore
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        load_dotenv()
        
        # Validate inputs
        if not validate_pdf_path(pdf_path):
            return False
            
        # Create output directory
        os.makedirs(index_path, exist_ok=True)
        
        logger.info(f"📄 Loading PDF from: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        if not docs:
            logger.error("No documents loaded from PDF")
            return False
            
        logger.info(f"✂️ Splitting {len(docs)} documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        
        if not chunks:
            logger.error("No chunks created from documents")
            return False
            
        logger.info(f"🧠 Creating embeddings for {len(chunks)} chunks...")
        embeddings = OpenAIEmbeddings()
        
        logger.info("📦 Building FAISS vectorstore...")
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        
        logger.info(f"💾 Saving vectorstore to: {index_path}")
        vectorstore.save_local(index_path)
        
        logger.info("✅ Vectorstore built successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error building vectorstore: {str(e)}")
        return False

if __name__ == "__main__":
    success = build_vectorstore()
    if not success:
        exit(1)
