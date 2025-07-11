# 📚 Generative Semantic Search Engine for Local Files

This project is a **Streamlit-based semantic search engine** that allows users to upload local documents (PDF, DOCX, PPTX, TXT), ask natural language questions, and receive precise, context-based answers using a **local LLM (Ollama)**. It combines **vector search (FAISS)** with **generative AI** for efficient document understanding.

---

## 🚀 Features

- ✅ Upload multiple documents (PDF, DOCX, PPTX, TXT)
- ✂️ Text chunking with adjustable size and overlap
- 🔎 Semantic search using `sentence-transformers/msmarco-bert-base-dot-v5`
- ⚡ Fast retrieval via FAISS indexing
- 🤖 Local answer generation with Ollama models (`gemma`, `mistral`, `llama2`)
- 🧠 Context-aware answering
- 💬 Streamlit UI for interactive usage
- 📑 Reference display for full traceability

---

## 🧠 How It Works

1. **Document Upload & Parsing**  
   Documents are uploaded and parsed into plain text using specialized readers.

2. **Text Chunking**  
   Text is split into overlapping chunks for better semantic understanding.

3. **Embedding & Indexing**  
   Chunks are embedded using Sentence Transformers and indexed with FAISS.

4. **Semantic Search**  
   A query is embedded and searched against the FAISS index for top-k matches.

5. **Answer Generation**  
   Retrieved context is passed to a local LLM (via Ollama) to generate answers.


