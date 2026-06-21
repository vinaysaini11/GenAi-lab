program 1
import gensim.downloader as api

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")   # smaller and faster
print("Model loaded!\n")

# Vector of a word
print("First 10 values of 'king':")
print(model["king"][:10])

# Similar words
print("\nWords similar to 'king':")
for word, score in model.most_similar("king", topn=5):
    print(word, ":", round(score, 4))

# Word analogy
print("\nking - man + woman = ?")
result = model.most_similar(
    positive=["king", "woman"],
    negative=["man"],
    topn=1
)

print(result[0][0], ":", round(result[0][1], 4))

# Similarity between two words
sim = model.similarity("king", "queen")
print("\nSimilarity between king and queen:", round(sim, 4))
