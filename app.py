"""
RAG PDF Question-Answering System
-----------------------------------
A simple, beginner-friendly Retrieval-Augmented Generation (RAG) project.

Pipeline:
1. Read a PDF and extract text
2. Split text into small chunks
3. Convert chunks into embeddings (numeric vectors) using a free local model
4. Store embeddings in a FAISS vector index
5. When user asks a question -> embed the question -> find closest chunks
6. Send question + retrieved chunks to an LLM (Groq API, free) -> get answer
"""

import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(page_title="RAG PDF Q&A", layout="wide")
st.title("📄 RAG-based PDF Question Answering System")
st.caption("Upload a PDF, ask questions, get answers grounded in that document.")


# ---------- STEP 1: Load embedding model (runs once, cached) ----------
@st.cache_resource
def load_embedding_model():
    # This model converts text into 384-dimensional vectors.
    # It runs locally on your machine - no API key needed for this step.
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedding_model()


# ---------- STEP 2: Extract text from PDF ----------
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


# ---------- STEP 3: Split text into overlapping chunks ----------
def chunk_text(text, chunk_size=300, overlap=50):
    """
    Splits text into word-based chunks.
    Overlap helps preserve context between chunks so we don't
    accidentally cut a sentence/idea in half.
    """
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


# ---------- STEP 4: Build FAISS vector store ----------
def build_vector_store(chunks):
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance search
    index.add(np.array(embeddings).astype("float32"))
    return index


# ---------- STEP 5: Retrieve most relevant chunks for a query ----------
def retrieve_relevant_chunks(query, chunks, index, top_k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(
        np.array(query_embedding).astype("float32"), top_k
    )
    return [chunks[i] for i in indices[0]]


# ---------- STEP 6: Generate answer using retrieved context ----------
def generate_answer(query, context_chunks, api_key):
    client = Groq(api_key=api_key)
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the answer is not present in the context, say "I don't have enough information in this document to answer that."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------- Streamlit UI ----------------
st.sidebar.header("⚙️ Setup")

# API key is loaded directly from .streamlit/secrets.toml — no user input needed.
api_key = st.secrets.get("GROQ_API_KEY", "")

if not api_key:
    st.sidebar.error("⚠️ GROQ_API_KEY not found in .streamlit/secrets.toml")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file and api_key:
    if "processed_file" not in st.session_state or st.session_state["processed_file"] != uploaded_file.name:
        with st.spinner("Reading PDF and building vector index..."):
            text = extract_text_from_pdf(uploaded_file)
            chunks = chunk_text(text)
            index = build_vector_store(chunks)

            st.session_state["chunks"] = chunks
            st.session_state["index"] = index
            st.session_state["processed_file"] = uploaded_file.name

        st.success(f"✅ PDF processed into {len(chunks)} chunks. Ready for questions!")

if "chunks" in st.session_state:
    query = st.text_input("💬 Ask a question about the PDF:")

    if query:
        with st.spinner("Searching document and generating answer..."):
            relevant_chunks = retrieve_relevant_chunks(
                query, st.session_state["chunks"], st.session_state["index"]
            )
            answer = generate_answer(query, relevant_chunks, api_key)

        st.markdown("### 🧠 Answer")
        st.write(answer)

        with st.expander("🔍 View retrieved chunks (what the AI actually read)"):
            for i, chunk in enumerate(relevant_chunks):
                st.markdown(f"**Chunk {i + 1}:**")
                st.write(chunk)
                st.divider()
else:
    st.info("👈 Upload a PDF to get started.")
