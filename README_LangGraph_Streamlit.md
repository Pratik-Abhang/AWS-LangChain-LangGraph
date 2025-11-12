# AWS AI Practitioner Chat Assistant (LangGraph)

A Streamlit-based chat application that uses LangGraph to create an intelligent assistant for AWS AI Practitioner course materials, with additional capabilities for Wikipedia and ArXiv research.

## Features

- **RAG-based PDF Q&A**: Query your AWS AI Practitioner course PDF using retrieval-augmented generation
- **Wikipedia Integration**: Get quick summaries on any topic from Wikipedia
- **ArXiv Research**: Search and find relevant research papers with direct links
- **Memory Management**: Persistent conversation memory using LangGraph's MemorySaver
- **Clean UI**: Simple and intuitive Streamlit interface

## Prerequisites

- Python 3.8+
- OpenAI API key
- AWS AI Practitioner course PDF

## Installation

1. **Clone/Download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

## Configuration

1. **Update PDF path**: In `1_LangGraph_Streamlit.py`, update the PDF path:
   ```python
   pdf_path = r"C:\Users\Admin\Downloads\AWS_Certified AI Practitioner_ Slides.pdf"
   ```
   Change this to your actual PDF file location.

## Usage

1. **Run the application**:
   ```bash
   streamlit run 1_LangGraph_Streamlit.py
   ```

2. **Access the app**: Open your browser to `http://localhost:8501`

3. **Start chatting**: Ask questions about:
   - AWS AI Practitioner topics (uses PDF content)
   - General topics (uses Wikipedia)
   - Research papers (searches ArXiv)

## How It Works

### Architecture
- **LLM**: Uses GPT-4 via OpenAI API
- **Vector Store**: FAISS for document embeddings
- **Document Processing**: PyMuPDF for PDF loading, RecursiveCharacterTextSplitter for chunking
- **Agent Framework**: LangGraph with MemorySaver for conversation persistence
- **Tools**: Three specialized tools for different data sources

### Tools Available
1. **AWS PDF Q&A**: Retrieves relevant information from the course PDF
2. **Wikipedia Search**: Provides 2-sentence summaries from Wikipedia
3. **ArXiv Search**: Returns top 3 relevant research papers with links

### Agent Behavior
The agent intelligently decides which tool to use based on your question:
- AWS-related questions → PDF Q&A tool
- General knowledge → Wikipedia tool
- Research topics → ArXiv tool
- Complex queries → Multiple tools as needed

## File Structure

```
LangChain/
├── 1_LangGraph_Streamlit.py    # Main application
├── requirements.txt            # Dependencies
├── .env                       # Environment variables
└── README_LangGraph_Streamlit.md  # This file
```

## Dependencies

- `langchain>=0.1.0` - Core LangChain framework
- `langchain_openai` - OpenAI integration
- `langgraph` - Graph-based agent framework
- `faiss-cpu` - Vector similarity search
- `streamlit` - Web interface
- `python-dotenv` - Environment variable management
- `wikipedia` - Wikipedia API wrapper
- `arxiv` - ArXiv API wrapper
- `PyMuPDF` - PDF processing

## Troubleshooting

### Common Issues

1. **PDF not found**: Ensure the PDF path is correct and accessible
2. **OpenAI API errors**: Verify your API key is valid and has sufficient credits
3. **Import errors**: Make sure all dependencies are installed
4. **Memory issues**: Large PDFs may require more RAM for processing

### Error Handling
The application includes error handling for:
- Wikipedia search failures
- ArXiv API timeouts
- PDF loading issues
- OpenAI API rate limits

## Customization

### Modify System Prompt
Edit the system prompt in the `create_agent` function to change the assistant's behavior:

```python
system_prompt = (
    "Your custom instructions here..."
)
```

### Adjust Chunk Size
Modify document chunking parameters:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # Adjust chunk size
    chunk_overlap=150   # Adjust overlap
)
```

### Change Model
Switch to a different OpenAI model:

```python
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")  # Use GPT-3.5 instead
```

## License

This project is for educational purposes. Ensure compliance with OpenAI's usage policies and respect the terms of service for Wikipedia and ArXiv APIs.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are correctly installed
3. Ensure your OpenAI API key is valid
4. Check that the PDF path is accessible
