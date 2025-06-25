from sentence_transformers import SentenceTransformer
import pickle

# Load your word list
with open("words/10KEnglishWords.txt", "r") as f:
    words = [line.strip() for line in f if line.strip()]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings
print("Generating embeddings for", len(words), "words...")
vectors = model.encode(words, show_progress_bar=True)

# Save to pickle
word_vectors = {word: vector.tolist() for word, vector in zip(words, vectors)}

with open("embeddings/word_vectors_real.pkl", "wb") as f:
    pickle.dump(word_vectors, f)

print("✅ Saved to embeddings/word_vectors_real.pkl")
