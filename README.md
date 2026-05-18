# MediGuru AI — Medical Retrieval-Augmented Generation (RAG) Assistant

MediGuru AI is an end-to-end medical question-answering system built using a Retrieval-Augmented Generation (RAG) architecture. The system retrieves relevant medical information from curated documents and uses a local large language model to generate grounded, context-aware responses.

The goal of this project is to demonstrate how modern AI systems can combine information retrieval and language models to produce more reliable and explainable outputs compared to standalone LLMs.

---

## System Overview

MediGuru AI follows a structured pipeline that separates knowledge retrieval from answer generation. Instead of relying solely on model memory, it dynamically retrieves relevant information from a document corpus and uses that as context for response generation.

---

## RAG Pipeline Implementation

The system implements the following pipeline:

1. **User Query Input**
   - The user provides a natural language medical question through the interface.

2. **Query Embedding**
   - The query is converted into a dense vector representation using a SentenceTransformer model (`all-MiniLM-L6-v2`).
   - This embedding captures semantic meaning rather than keyword matching.

3. **Vector Similarity Search (FAISS)**
   - The query embedding is compared against a pre-built FAISS index containing embeddings of document chunks.
   - FAISS returns the top-K most similar chunks based on vector distance.

4. **Document Retrieval**
   - The system retrieves the most relevant text segments from the indexed medical documents.
   - These segments form the knowledge base for answering the query.

5. **Context Construction**
   - Retrieved chunks are concatenated into a structured context.
   - Context size is controlled to maintain model efficiency and avoid overload.

6. **Prompt Formation**
   - A prompt is constructed that includes:
     - The retrieved context
     - The user’s question
   - The model is guided to answer using only the provided context.

7. **Answer Generation (LLM)**
   - The prompt is sent to a local language model (`phi3`) via Ollama.
   - The model generates a natural, human-readable response grounded in the retrieved data.

8. **Response Delivery**
   - The system returns:
     - Generated answer
     - Confidence score (based on retrieval similarity)
     - Source references (document snippets)

---

## Explainability and Transparency

One of the core design goals of MediGuru AI is explainability.

### 1. Source Attribution
Each response is backed by actual document snippets retrieved during the search phase. This allows users to verify the origin of the information.

### 2. Confidence Score
A confidence score is computed from similarity distances returned by FAISS. This gives an indication of how relevant the retrieved context is to the query.

### 3. Reduced Hallucination
Because the model is conditioned on real retrieved data, it is less likely to generate unsupported or fabricated answers compared to standalone LLMs.

### 4. Context Visibility
The system exposes retrieved evidence, making the reasoning process more transparent.

---

## Key Design Decisions

- **Local LLM (Ollama)**  
  Ensures privacy, no API dependency, and zero per-query cost.

- **FAISS Vector Store**  
  Enables fast and scalable similarity search over large document collections.

- **Sentence Transformers for Embeddings**  
  Provides efficient semantic understanding of queries and documents.

- **Chunk-Based Document Processing**  
  Improves retrieval precision by indexing smaller, meaningful text segments.

- **Dynamic Prompting**  
  The system adapts responses based on query type rather than enforcing rigid templates.

---

## Features

- Natural language medical Q&A
- Retrieval-based grounding using real documents
- Confidence scoring for responses
- Source citation with evidence snippets
- Suggested queries for guided interaction
- Clean, user-focused interface
- Fully local execution (no external APIs)

---

## Project Structure


MediGuru/<br>
│<br>
├── streamlit_app.py # Main application (UI + pipeline integration)<br>
├── vector_store/<br>
│ ├── index.faiss # FAISS vector index<br>
│ └── documents.pkl # Stored document chunks<br>
│<br>
├── data/ # Source medical documents (PDFs)<br>
├── app.py<br>
├── rag_engine.py<br>
└── README.md<br>


---

## Setup Instructions

1. Clone the repository:

git clone https://github.com/yourusername/mediguru-ai.git
cd mediguru-ai


2. Install dependencies:

pip install -r requirements.txt


3. Install and run Ollama:

ollama run phi3


4. Start the application:

streamlit run streamlit_app.py


---

## Limitations

- The system depends on the quality and coverage of the document dataset.
- Not a substitute for professional medical diagnosis.
- Smaller local models may have limitations in reasoning depth.

---

## Future Improvements

- Conversational memory for multi-turn dialogue
- Deployment using Django/React architecture
- Voice interaction support
- Real-time streaming responses
- Integration with live medical databases
- Improved ranking and re-ranking models

---

## Disclaimer

This system is intended for informational purposes only.  
It does not provide medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare professional.

---

## Summary

MediGuru AI demonstrates how combining retrieval systems with language models results in more reliable, explainable, and practical AI applications. It reflects real-world design patterns used in modern AI systems and serves as a strong foundation for building production-grade intelligent assistants.
