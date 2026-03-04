import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Load Documents
# -----------------------------
with open("sample_docs.txt", "r") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

print("Indexed Documents:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc}")

# -----------------------------
# 1️⃣ Lexical Search (TF-IDF)
# -----------------------------
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# -----------------------------
# 2️⃣ Semantic Search (Simulated Embeddings)
# -----------------------------
# For simplicity, we reuse TF-IDF as embedding simulation
# In production, this would be neural embeddings (OpenAI, SBERT, etc.)

embedding_matrix = tfidf_matrix.copy()

# -----------------------------
# Query Input
# -----------------------------
query = input("\nEnter your search query: ")

query_tfidf = tfidf_vectorizer.transform([query])
query_embedding = query_tfidf.copy()

# -----------------------------
# Compute Similarities
# -----------------------------
lexical_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]
semantic_scores = cosine_similarity(query_embedding, embedding_matrix)[0]

# -----------------------------
# Normalize Scores
# -----------------------------
scaler = MinMaxScaler()

lexical_scores_norm = scaler.fit_transform(lexical_scores.reshape(-1,1)).flatten()
semantic_scores_norm = scaler.fit_transform(semantic_scores.reshape(-1,1)).flatten()

# -----------------------------
# Hybrid Score Fusion
# -----------------------------
alpha = 0.5  # weight for lexical
beta = 0.5   # weight for semantic

hybrid_scores = alpha * lexical_scores_norm + beta * semantic_scores_norm

# -----------------------------
# Ranking
# -----------------------------
ranked_indices = np.argsort(hybrid_scores)[::-1]

print("\n🔎 Hybrid Search Results:")
for rank, idx in enumerate(ranked_indices):
    print(f"\nRank {rank+1}")
    print("Document:", documents[idx])
    print("Lexical Score:", round(lexical_scores_norm[idx], 4))
    print("Semantic Score:", round(semantic_scores_norm[idx], 4))
    print("Hybrid Score:", round(hybrid_scores[idx], 4))
