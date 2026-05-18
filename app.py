import faiss
import pickle
import numpy as np
import ollama
from sentence_transformers import SentenceTransformer

print("🚀 Loading medical RAG system...")

# Load FAISS index
index = faiss.read_index("vector_store/index.faiss")

# Load documents
with open("vector_store/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Load embedding model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

print("✅ System ready")

while True:
    query = input("\n🔍 Ask a medical question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    # -----------------------------
    # STEP 1 — Query Embedding
    # -----------------------------
    query_embedding = embed_model.encode([query])
    faiss.normalize_L2(query_embedding)

    # -----------------------------
    # STEP 2 — Retrieval
    # -----------------------------
    k = 12
    distances, indices = index.search(np.array(query_embedding), k)

    scores = distances[0]

    retrieved_docs = [documents[i] for i in indices[0]]

    # -----------------------------
    # STEP 3 — Weighted Re-ranking
    # -----------------------------
    def score(doc, distance):
        keyword_score = len(set(query.lower().split()) & set(doc.page_content.lower().split()))
        return keyword_score + (1 / (distance + 1e-5))

    ranked = sorted(
        zip(retrieved_docs, scores),
        key=lambda x: score(x[0], x[1]),
        reverse=True
    )

    top_docs = [doc for doc, _ in ranked[:3]]

    # -----------------------------
    # STEP 4 — Context Creation
    # -----------------------------
    context = "\n\n".join([doc.page_content for doc in top_docs])

    # -----------------------------
    # STEP 5 — Confidence Score
    # -----------------------------
    confidence = float(np.mean(scores))

    # -----------------------------
    # STEP 6 — Prompt Engineering
    # -----------------------------
    prompt = f"""
You are a medical AI assistant.

STRICT RULES:
- Use ONLY the context
- Keep answers SIMPLE and CLEAR
- Do NOT use overly technical terms unless necessary

Format:

Symptoms:
- short bullet points only (1 line each)

Explanation:
- 2–3 simple sentences

When to see a doctor:
- short actionable points

If context is insufficient, say:
"I don't have enough information"

Context:
{context}

Question:
{query}
"""

    # -----------------------------
    # STEP 7 — LLM Response
    # -----------------------------
    response = ollama.chat(
        model='phi3',
        messages=[{'role': 'user', 'content': prompt}]
    )

    # -----------------------------
    # OUTPUT
    # -----------------------------
    print("\n🧠 Answer:\n")
    print(response['message']['content'])

    print(f"\n📊 Confidence Score: {confidence:.4f}")

    print("\n📚 Sources:")
    for doc in top_docs:
        print("-", doc.metadata.get("source"))

    print("\n🔍 Evidence Snippets:")
    for doc in top_docs:
        print("\n---")
        print(doc.page_content[:200])

    print("\n⚠️ This is not medical advice. Consult a licensed healthcare professional.")