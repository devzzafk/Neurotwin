import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# STEP 1: Load Documents
# ----------------------------

with open("sample_docs.txt", "r") as f:
    documents = [line.strip() for line in f.readlines() if line.strip()]

print("\nIndexed Documents:")
for i, doc in enumerate(documents):
    print(f"{i+1}. {doc}")

# ----------------------------
# STEP 2: Create TF-IDF Matrix (Lexical Search)
# ----------------------------

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# ----------------------------
# STEP 3: Take User Query
# ----------------------------

query = input("\nEnter your search query: ")

query_vector = vectorizer.transform([query])

# ----------------------------
# STEP 4: Compute Lexical Similarity
# ----------------------------

lexical_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

# ----------------------------
# STEP 5: Simulated Semantic Similarity
# (In real systems this would use embeddings)
# ----------------------------

semantic_scores = cosine_similarity(query_vector, tfidf_matrix)[0]

# ----------------------------
# STEP 6: Normalize Scores (0 to 1)
# ----------------------------

scaler = MinMaxScaler()

lexical_norm = scaler.fit_transform(lexical_scores.reshape(-1, 1)).flatten()
semantic_norm = scaler.fit_transform(semantic_scores.reshape(-1, 1)).flatten()

# ----------------------------
# STEP 7: Hybrid Score Fusion
# ----------------------------

alpha = 0.5  # weight for lexical search
beta = 0.5   # weight for semantic search

hybrid_scores = alpha * lexical_norm + beta * semantic_norm

# ----------------------------
# STEP 8: Rank Results
# ----------------------------

ranked_indices = np.argsort(hybrid_scores)[::-1]

print("\n--- Hybrid Search Results ---")

for rank, idx in enumerate(ranked_indices):
    print(f"\nRank {rank+1}")
    print("Document:", documents[idx])
    print("Lexical Score:", round(lexical_norm[idx], 4))
    print("Semantic Score:", round(semantic_norm[idx], 4))
    print("Hybrid Score:", round(hybrid_scores[idx], 4))
