"""
AWS AI Practitioner Chat Assistant using LangGraph and Streamlit.
"""
import os
import logging
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import wikipedia
import arxiv
from dotenv import load_dotenv

# Constants
DEFAULT_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.3
DEFAULT_INDEX_PATH = os.getenv("INDEX_PATH", "./vectorstore/faiss_index")
MAX_ARXIV_RESULTS = 3
WIKIPEDIA_SENTENCES = 2

# System prompt
SYSTEM_PROMPT = """
You are an intelligent AI assistant specialized in helping users with the AWS AI Practitioner course.
Your primary knowledge source is the provided AWS course PDF (via the RAG system).
If a user's question is related to AWS, always try to retrieve and summarize the most relevant content from the PDF first.
If the information is not available or insufficient in the RAG source, then use your knowledge and Wikipedia to enrich or validate your response.
If both RAG and Wikipedia fail to provide enough relevant information, rely on your own general knowledge and reasoning.
When responding, clearly explain which sources were used. Always be helpful, concise, and logically structured.
"""

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def validate_input(user_input: str) -> str:
    """Validate and sanitize user input."""
    if not user_input or not user_input.strip():
        return ""
    return user_input.strip()[:1000]  # Limit input length

def check_vectorstore_exists(index_path: str) -> bool:
    """Check if vectorstore exists."""
    return os.path.exists(index_path) and os.path.exists(os.path.join(index_path, "index.faiss"))

@st.cache_resource
def setup_agent():
    """
    Setup the LangGraph agent with tools and memory.
    
    Returns:
        tuple: (agent, config) or (None, None) if setup fails
    """
    try:
        # Validate vectorstore exists
        if not check_vectorstore_exists(DEFAULT_INDEX_PATH):
            st.error(f"❌ Vectorstore not found at {DEFAULT_INDEX_PATH}. Please run build_vectorstore.py first.")
            return None, None
            
        llm = ChatOpenAI(
            temperature=DEFAULT_TEMPERATURE, 
            model_name=DEFAULT_MODEL
        )
        embeddings = OpenAIEmbeddings()

        # Load vectorstore
        vectorstore = FAISS.load_local(
            DEFAULT_INDEX_PATH, 
            embeddings, 
            allow_dangerous_deserialization=True
        )

        rag_qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever()
        )

        @tool
        def aws_pdf_qa(query: str) -> str:
            """Answer questions related to AWS AI Practitioner PDF course."""
            try:
                return rag_qa.run(query)
            except Exception as e:
                logger.error(f"AWS PDF QA error: {e}")
                return f"Error accessing course material: {str(e)}"

        @tool
        def wiki_search(query: str) -> str:
            """Get a brief summary about a topic from Wikipedia."""
            try:
                return wikipedia.summary(query, sentences=WIKIPEDIA_SENTENCES)
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Multiple topics found. Please be more specific. Options: {e.options[:3]}"
            except wikipedia.exceptions.PageError:
                return "No Wikipedia page found for this topic."
            except Exception as e:
                logger.error(f"Wikipedia search error: {e}")
                return f"Wikipedia search failed: {str(e)}"

        @tool
        def arxiv_search(query: str) -> str:
            """Search for research papers on ArXiv and provide clickable links."""
            try:
                search = arxiv.Search(
                    query=query, 
                    max_results=MAX_ARXIV_RESULTS, 
                    sort_by=arxiv.SortCriterion.Relevance
                )
                results = [f"{r.title} - [Link]({r.entry_id})" for r in search.results()]
                return "\n".join(results) if results else "No papers found."
            except Exception as e:
                logger.error(f"ArXiv search error: {e}")
                return f"ArXiv search failed: {str(e)}"

        tools = [aws_pdf_qa, wiki_search, arxiv_search]

        # Create agent
        checkpointer = MemorySaver()
        agent = create_agent(
            system_prompt=SYSTEM_PROMPT,
            tools=tools,
            model=llm,
            checkpointer=checkpointer,
        )

        config = {"configurable": {"thread_id": "1"}}
        return agent, config
        
    except Exception as e:
        logger.error(f"Agent setup error: {e}")
        st.error(f"❌ Failed to setup agent: {str(e)}")
        return None, None

def main():
    """Main Streamlit application."""
    # Page config
    st.set_page_config(
        page_title="AWS AI Practitioner Assistant", 
        page_icon="🤖", 
        layout="centered"
    )

    st.title("🤖 AWS AI Practitioner Chat Assistant")
    st.write("Ask me questions about your AWS course PDF, Wikipedia topics, or ArXiv research papers!")

    # Setup agent
    agent, config = setup_agent()
    if not agent or not config:
        st.stop()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Create input + exit button side-by-side
    col1, col2 = st.columns([4, 1])

    with col1:
        prompt = st.chat_input("Ask me anything about AWS AI, research, or general topics...")

    with col2:
        if st.button("🚪 Exit"):
            st.session_state["messages"] = []
            st.success("Chat ended. You can close the tab or ask a new question.")
            st.stop()

    # Process user input
    if prompt:
        # Validate input
        validated_prompt = validate_input(prompt)
        if not validated_prompt:
            st.warning("Please enter a valid question.")
            return

        # Add user message
        st.session_state.messages.append({"role": "user", "content": validated_prompt})
        with st.chat_message("user"):
            st.markdown(validated_prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = agent.invoke({"messages": st.session_state.messages}, config)
                    answer = result["messages"][-1].content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    error_msg = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Agent invoke error: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
