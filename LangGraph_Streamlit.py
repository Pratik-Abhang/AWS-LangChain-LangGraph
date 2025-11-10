import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
import wikipedia
import arxiv
import os
from dotenv import load_dotenv

# ---------- 0. Load environment variables ----------
load_dotenv()

# ---------- 1. Setup LLM, embeddings, and memory ----------
st.set_page_config(page_title="AWS AI Practitioner Assistant", page_icon="🤖", layout="centered")

st.title("🤖 AWS AI Practitioner Chat Assistant")
st.write("Ask me questions about your AWS course PDF, Wikipedia topics, or ArXiv research papers!")

@st.cache_resource
def setup_agent():
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    embeddings = OpenAIEmbeddings()

    pdf_path = r"C:\Users\Admin\Downloads\AWS_Certified AI Practitioner_ Slides.pdf"
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    rag_qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

    @tool
    def aws_pdf_qa(query: str) -> str:
        """Answer questions related to AWS AI Practitioner PDF course."""
        return rag_qa.run(query)

    @tool
    def wiki_search(query: str) -> str:
        """Get a brief summary about a topic from Wikipedia"""
        try:
            return wikipedia.summary(query, sentences=2)
        except Exception as e:
            return f"Wikipedia search failed: {e}"

    @tool
    def arxiv_search(query: str) -> str:
        """Search for research papers on ArXiv and provide clickable links"""
        try:
            search = arxiv.Search(query=query, max_results=3, sort_by=arxiv.SortCriterion.Relevance)
            results = [f"{r.title} - [Link]({r.entry_id})" for r in search.results()]
            return "\n".join(results) if results else "No papers found."
        except Exception as e:
            return f"ArXiv search failed: {e}"

    tools = [aws_pdf_qa, wiki_search, arxiv_search]

    checkpointer = MemorySaver()
    agent = create_agent(
        system_prompt = (
        "You are an intelligent AI assistant specialized in helping users with the AWS AI Practitioner course. "
        "Your primary knowledge source is the provided AWS course PDF (via the RAG system). "
        "If a user's question is related to AWS, always try to retrieve and summarize the most relevant content from the PDF first. "
        "If the information is not available or insufficient in the RAG source, then use Wikipedia to enrich or validate your response. "
        "If both RAG and Wikipedia fail to provide enough relevant information, rely on your own general knowledge and reasoning to give a complete, accurate, and confident answer. "
        "When responding, clearly explain which sources were used — for example, say 'Based on the course material...', or "
        "'The PDF did not contain enough relevant information, so I referred to Wikipedia...', or "
        "'I couldn’t find relevant details in the course or Wikipedia, so here’s my answer based on my knowledge and reasoning.' "
        "Always be helpful, concise, and logically structured. Do not refuse to answer simply because something was not found in RAG; "
        "instead, make a best-effort answer with clear context about your source reasoning."
    ),
        tools=tools,
        model=llm,
        checkpointer=checkpointer,
    )

    config = {"configurable": {"thread_id": "1"}}
    return agent, config

agent, config = setup_agent()

# ---------- 2. Streamlit chat interface ----------
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

# If user entered text, process it
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": st.session_state.messages}, config)
            answer = result["messages"][-1].content
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
