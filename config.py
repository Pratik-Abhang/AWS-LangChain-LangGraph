"""
Configuration settings for the AWS AI Practitioner Chat Assistant.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# File paths
PDF_PATH = os.getenv("PDF_PATH", r"C:\Users\Admin\Downloads\AWS_Certified AI Practitioner_ Slides.pdf")
INDEX_PATH = os.getenv("INDEX_PATH", "./vectorstore/faiss_index")

# Model settings
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# Document processing
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# API limits
MAX_ARXIV_RESULTS = int(os.getenv("MAX_ARXIV_RESULTS", "3"))
WIKIPEDIA_SENTENCES = int(os.getenv("WIKIPEDIA_SENTENCES", "2"))
MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", "1000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
