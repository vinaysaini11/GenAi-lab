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

program 2

!pip install gensim scikit-learn matplotlib

import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

print("Loading model...")
model = api.load("glove-wiki-gigaword-50")  # smaller model
print("Model loaded!\n")

# Similar words
print("Words similar to 'computer':")
for word, score in model.most_similar("computer", topn=5):
    print(word, ":", round(score, 4))

# Words to visualize
words = ["computer", "software", "hardware", "internet", "network"]

# Get vectors
vectors = [model[word] for word in words]

# Convert to 2D using PCA
pca = PCA(n_components=2)
points = pca.fit_transform(vectors)

# Plot
for i, word in enumerate(words):
    plt.scatter(points[i, 0], points[i, 1])
    plt.text(points[i, 0], points[i, 1], word)

plt.title("Word Embeddings")
plt.grid()
plt.show()

program 3

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Medical sentences
corpus = [
    ["patient", "diabetes", "hypertension"],
    ["mri", "brain", "tissue"],
    ["treatment", "antibiotics", "monitoring"],
    ["fever", "fatigue", "pain"],
    ["vaccine", "viral", "infection"]
]

# Train model
model = Word2Vec(corpus, vector_size=50, min_count=1)

# Similar words
print("Words similar to 'treatment':")
for word, score in model.wv.most_similar("treatment", topn=3):
    print(word, ":", round(score, 2))

# Words to plot
words = model.wv.index_to_key

# Get vectors
vectors = [model.wv[word] for word in words]

# PCA
pca = PCA(n_components=2)
points = pca.fit_transform(vectors)

# Plot
for i, word in enumerate(words):
    plt.scatter(points[i, 0], points[i, 1])
    plt.text(points[i, 0], points[i, 1], word)

plt.title("Medical Word Embeddings")
plt.grid()
plt.show()

program 4

import gensim.downloader as api
from transformers import pipeline

# Load pretrained Word2Vec
print("Loading Word2Vec model...")
wv = api.load("glove-wiki-gigaword-50")   # smaller and easier than google-news-300

# Load text generation model
print("Loading GPT model...")
generator = pipeline("text-generation", model="distilgpt2")

# Original prompt
prompt = "Explain the responsibilities of a king."

# Retrieve similar words
similar_words = [word for word, score in wv.most_similar("king", topn=5)]

# Enrich prompt
enriched_prompt = (
    prompt +
    " Include concepts related to " +
    ", ".join(similar_words)
)

print("\nSimilar Words:")
print(similar_words)

print("\nOriginal Prompt:")
print(prompt)

print("\nEnriched Prompt:")
print(enriched_prompt)

# Generate responses
response1 = generator(prompt)[0]["generated_text"]
response2 = generator(enriched_prompt)[0]["generated_text"]

print("\n----- Original Response -----")
print(response1)

print("\n----- Enriched Response -----")
print(response2)

print("\nComparison")
print("Original Length:", len(response1))
print("Enriched Length:", len(response2))

program 5

import gensim.downloader as api, random

print("Loading word vectors...")
wv = api.load("glove-wiki-gigaword-100")
print("Model loaded successfully!\n")

w = input("Enter a seed word: ")

try:
    words = [x for x, _ in wv.most_similar(w, 5)]
    print("\nSimilar words:", words)

    random.shuffle(words)

    para = (
        f"The word '{w}' often reminds people of {words[0]} and {words[1]}. "
        f"In many discussions, {words[2]} is closely connected with {w}. "
        f"People may also think about {words[3]} when talking about {w}. "
        f"Together, ideas like {words[4]} help explain the broader meaning of '{w}'."
    )

    print("\nGenerated Paragraph:\n")
    print(para)

except:
    print("Word not found in vocabulary. Please try another word.")


program 6

from transformers import pipeline

print("\nLoading Sentiment analysis model")

sa=pipeline("sentiment-analysis")

reviews=[
    "Product was good",
    "Worst Product ever"
]

print("Customer feedback: \n")

for r in reviews:
    res=sa(r)[0]
    print("Result : ",res)
    print("\nLabel: ",res['label'])
    print("\nScore: ",round(res['score'],4))


program 7

#!pip uninstall -y transformers
#!pip install transformers==4.41.2 torch sentencepiece


#// restart session


from transformers import pipeline

print("Loading Summarization Model/BART: ")

summ=pipeline("summarization",model="facebook/bart-large-cnn")

def summarize(t):
    t=" ".join(t.split())
    mx=min(len(t)//3,150)
    mn=max(30,mx//3)
    return summ(t,max_length=mx,min_length=mn,do_sample=False)[0]['summary_text']

text="""Artificial Intelligence (AI) is a rapidly evolving field of computer science..."""

res=summarize(text)

print("\n Original Text: ",text)
print("\n Summarize Text: ",res)


program 8


#!pip install langchain-cohere cohere

from langchain_cohere import ChatCohere

# Read text file
with open("teaching.txt", "r") as f:
    text = f.read()

# Model
llm = ChatCohere(
    cohere_api_key=input("Enter API Key: ")
)

# Query
res = llm.invoke(
    f"Summarize this text and give 3 key points + sentiment:\n{text}"
).content

print(res)


program 9

from pydantic import BaseModel
from typing import Optional
import wikipediaapi

class Inst(BaseModel):
    founder: Optional[str]
    founded: Optional[str]
    summary: Optional[str]

def get_data(name):
    wiki = wikipediaapi.Wikipedia(
        user_agent="GenAI-Lab/1.0",
        language="en"
    )

    page = wiki.page(name)

    if not page.exists():
        raise ValueError("Page not found!")

    text = page.text.lower().split('\n')

    founder = next((x for x in text if "founder" in x), None)
    founded = next((x for x in text if "founded" in x), None)
    summary = '. '.join(page.summary.split('. ')[:4])

    return Inst(
        founder=founder,
        founded=founded,
        summary=summary
    )

name = input("Enter Institution Name: ")

try:
    data = get_data(name)

    print("\nFounder :", data.founder or "N/A")
    print("Founded :", data.founded or "N/A")
    print("Summary :", data.summary)

except Exception as e:
    print("Error:", e)

program 10

from langchain_cohere import ChatCohere
import wikipediaapi

llm = ChatCohere(
    cohere_api_key=input("Enter your Cohere API Key: "),
    model="command-a-03-2025"
)

ipc = wikipediaapi.Wikipedia(
    user_agent="IPCChatbot", language="en"
).page("Indian Penal Code").summary

print("\nIPC Chatbot Ready! Type 'exit' to quit.\n")

while True:
    q = input("You: ")
    if q.lower() == "exit":
        print("Exiting chatbot...")
        break

    r = llm.invoke(f"{ipc}\n\nQuestion: {q}\n\nSection:\nExplanation:").content
    s = r.split("Explanation:")

    print("\nSection:", s[0].replace("Section:", "").strip() if len(s)>1 else "N/A")
    print("Explanation:", s[-1].strip(), "\n")
