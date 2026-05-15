# rag_chain.py
# All RAG logic lives here — completely separate from the UI
# This file can be imported by Streamlit, FastAPI, tests, or anything else

import os
import tempfile
from typing import List, Tuple

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.documents import Document


# ── Embedding model ───────────────────────────────────────────────────────────
# This function is called by Streamlit with @st.cache_resource
# so the model is only loaded once per session, not on every rerun

def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Load and return the local embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


# ── Document processing ───────────────────────────────────────────────────────

def process_pdf(
    file_bytes: bytes,
    embeddings_model: HuggingFaceEmbeddings,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> Tuple[Chroma, int]:
    """
    Takes raw PDF bytes, processes them, returns a Chroma vectorstore.
    
    Steps:
    1. Write bytes to a temp file (PyPDFLoader needs a file path)
    2. Load PDF pages as Documents
    3. Split pages into chunks
    4. Embed chunks and store in Chroma (in-memory, not persisted)
    5. Return the vectorstore and chunk count
    
    Returns:
        vectorstore: Chroma instance ready for retrieval
        chunk_count: number of chunks stored (for display in UI)
    """

    # Step 1: Write bytes to temp file
    # delete=False because we need the file to exist after the 'with' block closes
    # suffix=".pdf" so PyPDFLoader recognizes the file type
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    try:
        # Step 2: Load PDF
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        if not pages:
            raise ValueError("PDF appears to be empty or unreadable.")

        # Step 3: Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        chunks = splitter.split_documents(pages)

        if not chunks:
            raise ValueError("Could not extract text from PDF. It may be scanned/image-based.")

        # Step 4: Embed and store in Chroma
        # Note: no persist_directory here — this is an in-memory store
        # It exists only for the current session
        # Reason: each uploaded PDF gets its own fresh vectorstore
        # We don't want chunks from a previous PDF contaminating the new one
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_model,
        )

        return vectorstore, len(chunks)

    finally:
        # Always clean up the temp file, even if an error occurred
        # os.unlink() deletes a file by path
        os.unlink(tmp_path)


# ── RAG chain builder ─────────────────────────────────────────────────────────

def build_rag_chain(vectorstore: Chroma) -> RunnableParallel:
    """
    Takes a Chroma vectorstore and returns a complete RAG chain.
    
    The chain takes a question string as input and returns:
        {
            "answer": "..string answer..",
            "sources": [...Document objects...]
        }
    """

    # Retriever: searches Chroma for relevant chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # LLM: temperature=0 for factual, consistent answers
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )

    # RAG prompt: strict instructions to only use provided context
    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are DocuMind, a helpful document assistant.

Your rules:
- Answer using ONLY the information from the context provided
- If the answer is not in the context, respond: "I don't find this information in the uploaded document."
- Always cite which page(s) you used at the end of your answer, like: (Source: Page 3)
- Be clear and concise
- Use bullet points for multi-part answers"""),
        ("human", """Document context:
{context}

Question: {question}""")
    ])

    def format_docs(docs: List[Document]) -> str:
        """Convert list of Documents into a formatted context string."""
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
            for doc in docs
        )

    # Core RAG chain (answer only)
    core_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    # Full chain with sources (answer + source documents)
    chain_with_sources = RunnableParallel(
        answer=core_chain,
        sources=retriever
    )

    return chain_with_sources


def get_streaming_chain(vectorstore: Chroma):
    """
    Returns a chain that streams the answer token by token.
    Used specifically for st.write_stream() in Streamlit.
    
    This is separate from build_rag_chain() because streaming
    and source retrieval need different chain structures.
    The Streamlit app uses this for display, then calls
    build_rag_chain() separately to get the source documents.
    """

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        streaming=True  # enables token-by-token streaming
    )

    RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are DocuMind, a helpful document assistant.

Your rules:
- Answer using ONLY the information from the context provided
- If the answer is not in the context, respond: "I don't find this information in the uploaded document."
- Always cite which page(s) you used at the end of your answer, like: (Source: Page 3)
- Be clear and concise
- Use bullet points for multi-part answers"""),
        ("human", """Document context:
{context}

Question: {question}""")
    ])

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"[Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
            for doc in docs
        )

    streaming_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return streaming_chain, retriever