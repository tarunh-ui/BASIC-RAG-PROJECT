# 📄 RAG PDF Q&A

A simple, beginner-friendly **Retrieval-Augmented Generation (RAG)** app that lets you upload a PDF and ask questions about it — answers are generated using an LLM but grounded strictly in the document's content.

Built with **Streamlit**, **FAISS**, **Sentence Transformers**, and the **Groq API** (free, ultra-fast inference).

---

## 🚀 Features

- 📤 Upload any PDF and extract its text
- ✂️ Automatic chunking with overlap to preserve context
- 🧠 Local embedding generation (no API key needed for this step)
- 🔍 Fast similarity search using FAISS
- 💬 Ask natural-language questions and get answers grounded in the document
- 🔎 View exactly which chunks were retrieved to generate the answer

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| PDF parsing | [PyPDF2](https://pypi.org/project/PyPDF2/) |
| Embeddings | [Sentence Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) |
| LLM inference | [Groq API](https://console.groq.com) (`openai/gpt-oss-20b`) |

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/rag-pdf-qa.git
   cd rag-pdf-qa
   ```

2. **Create a virtual environment (Python 3.11)**
   ```bash
   py -3.11 -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get a free Groq API key**
   Sign up at [console.groq.com](https://console.groq.com) and generate an API key.

---

## ▶️ Usage

```bash
streamlit run app.py
```

Then in the browser:
1. Enter your Groq API key in the sidebar
2. Upload a PDF
3. Ask a question — the app retrieves the most relevant chunks and generates a grounded answer

---

## 🧩 How It Works

1. **Extract** — Text is pulled from the uploaded PDF
2. **Chunk** — Text is split into overlapping word-based chunks to preserve context
3. **Embed** — Each chunk is converted into a 384-dimension vector using a local embedding model
4. **Index** — Vectors are stored in a FAISS index for fast similarity search
5. **Retrieve** — When a question is asked, it's embedded and matched against the closest chunks
6. **Generate** — The question + retrieved chunks are sent to the Groq LLM, which answers using only that context

---

## 📁 Project Structure

```
rag-pdf-qa/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md
```

---

## ⚠️ Notes

- Answers are restricted to the uploaded document — if the info isn't in the PDF, the app will say so instead of hallucinating.
- Groq's free tier model list changes periodically — check [console.groq.com/docs/models](https://console.groq.com/docs/models) if you hit a `model_not_found` error.

---

## 📄 License

MIT License — feel free to use and modify.
