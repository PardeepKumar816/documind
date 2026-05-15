# app.py
# The Streamlit UI for DocuMind
# Run with: streamlit run app.py

import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # must be before any LangChain imports

from rag_chain import get_embeddings_model, process_pdf, get_streaming_chain


# ── Page configuration ────────────────────────────────────────────────────────
# Must be the FIRST Streamlit call in the script
# page_title   = browser tab title
# page_icon    = emoji or URL to icon shown in browser tab
# layout       = "wide" uses full browser width, "centered" is narrower
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide"
)


# ── Load embedding model once ─────────────────────────────────────────────────
# @st.cache_resource caches the return value
# The function only runs ONCE per app session, no matter how many reruns happen
# Without this: the 80MB model would reload on every button click
@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    # This wrapper exists so @st.cache_resource can cache get_embeddings_model()
    return get_embeddings_model()

embeddings_model = load_model()


# ── Initialize session state ──────────────────────────────────────────────────
# session_state persists across reruns within one browser session
# We initialize all keys here at the top so we never get KeyError later
#
# "messages"     — list of chat messages: [{"role": "human"/"assistant", "content": "...", "sources": [...]}]
# "vectorstore"  — the Chroma instance for the current PDF
# "pdf_name"     — name of the currently loaded PDF (for display)
# "chain"        — the streaming chain for the current PDF

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
# st.sidebar is a context manager — everything inside renders in the left sidebar

with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("Chat with any PDF using AI")
    st.divider()

    st.subheader("Upload Document")

    # st.file_uploader renders a drag-and-drop file upload widget
    # type=["pdf"] restricts to PDF files only
    # When a file is uploaded, uploaded_file is a UploadedFile object
    # When nothing is uploaded, it's None
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF — research papers, books, reports, documentation"
    )

    if uploaded_file is not None:
        # Check if this is a NEW file or the same one already processed
        # We don't want to re-process the same PDF on every rerun
        # uploaded_file.name is the original filename from the user's computer
        if uploaded_file.name != st.session_state.pdf_name:

            # New PDF uploaded — process it
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    # process_pdf() returns (vectorstore, chunk_count)
                    # uploaded_file.read() reads all bytes from the uploaded file
                    vectorstore, chunk_count = process_pdf(
                        file_bytes=uploaded_file.read(),
                        embeddings_model=embeddings_model
                    )

                    # Store in session_state so it survives reruns
                    st.session_state.vectorstore = vectorstore
                    st.session_state.pdf_name = uploaded_file.name

                    # Clear previous chat when a new PDF is loaded
                    # Old chat about a different PDF would be confusing
                    st.session_state.messages = []

                    st.success(f"✅ Ready! Processed {chunk_count} chunks.")

                except ValueError as e:
                    # Our custom errors from process_pdf()
                    st.error(f"❌ {str(e)}")
                except Exception as e:
                    # Unexpected errors
                    st.error(f"❌ Unexpected error: {str(e)}")
        else:
            # Same file as before — already processed
            st.success(f"✅ {uploaded_file.name} loaded")

    # Show document stats if a PDF is loaded
    if st.session_state.vectorstore is not None:
        st.divider()
        st.subheader("Document Info")
        st.write(f"📄 **File:** {st.session_state.pdf_name}")
        chunk_count = st.session_state.vectorstore._collection.count()
        st.write(f"🔢 **Chunks:** {chunk_count}")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            # st.rerun() forces an immediate rerun of the script
            # Used here to instantly clear the chat UI
            st.rerun()

    st.divider()
    st.caption("Built with LangChain + Groq + ChromaDB")
    st.caption("Free & Open Source")


# ── Main chat area ────────────────────────────────────────────────────────────

# App title in the main area
st.title("DocuMind 🧠")
st.caption("Upload a PDF in the sidebar, then ask any question about it.")

# If no PDF loaded yet, show instructions
if st.session_state.vectorstore is None:
    # st.info renders a blue info box
    st.info("👈 Upload a PDF in the sidebar to get started.")

    # Show example use cases so users understand what to do
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📚 Research Papers")
        st.write("Ask about methodology, findings, conclusions")
    with col2:
        st.markdown("### 📋 Reports")
        st.write("Summarize sections, extract key data points")
    with col3:
        st.markdown("### 📖 Documentation")
        st.write("Find specific features, understand concepts")

else:
    # PDF is loaded — show the chat interface

    # Render all previous messages from session_state
    # This loop runs on every rerun, rebuilding the chat history from session_state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

            # Show sources for assistant messages if they exist
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("📚 Sources used", expanded=False):
                    for i, doc in enumerate(message["sources"]):
                        page_num = doc.metadata.get('page', 0) + 1
                        st.caption(f"**Chunk {i+1} — Page {page_num}**")
                        # st.info renders the chunk text in a subtle blue box
                        st.info(doc.page_content)

    # Chat input box — renders at the bottom of the page
    # Returns the typed text when user hits Enter, None otherwise
    # placeholder= is the greyed-out hint text inside the input
    if question := st.chat_input(
        placeholder=f"Ask anything about {st.session_state.pdf_name}...",
    ):
        # 1. Show the user's question immediately
        with st.chat_message("human"):
            st.write(question)

        # 2. Save to session_state
        st.session_state.messages.append({
            "role": "human",
            "content": question,
            "sources": []
        })

        # 3. Generate and stream the answer
        with st.chat_message("assistant"):
            try:
                # Get the streaming chain and retriever for the current vectorstore
                streaming_chain, retriever = get_streaming_chain(
                    st.session_state.vectorstore
                )

                # st.write_stream() takes a generator and displays tokens live
                # It returns the full assembled string when streaming completes
                # This is the key Streamlit function for streaming LLM responses
                full_answer = st.write_stream(
                    streaming_chain.stream(question)
                )

                # 4. Get source documents separately
                # We retrieve again to get the Document objects for display
                # (the streaming chain doesn't return them)
                source_docs = retriever.invoke(question)

                # 5. Show sources in an expander below the answer
                with st.expander("📚 Sources used", expanded=False):
                    for i, doc in enumerate(source_docs):
                        page_num = doc.metadata.get('page', 0) + 1
                        st.caption(f"**Chunk {i+1} — Page {page_num}**")
                        st.info(doc.page_content)

                # 6. Save assistant response to session_state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_answer,
                    "sources": source_docs
                })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "sources": []
                })