import wikipedia
import re
import numpy as np
import pandas as pd
import gensim
from nltk.tokenize import word_tokenize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ========== CONFIGURATION ==========
TOPICS = ["University", "College", "Highschool", "Academy", "Institute"]
MAX_CHARS = 500
VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_WORD_COUNT = 1
LABELS = list(range(len(TOPICS)))


# tokenization and cleaning

def clean_and_tokenize(text):
    """Removes punctuation and lowercases, then tokenizes the input text."""
    cleaned = re.sub(r'[^\w\s]', '', text.lower())
    return word_tokenize(cleaned)


# data fetching from wikipedia

def fetch_wikipedia_documents(topics, max_chars=500):
    """Fetch and tokenize a fixed number of characters from Wikipedia summaries."""
    tokenized_docs = []
    raw_docs = []

    print("=== Wikipedia Page Fetch Results ===")
    for topic in topics:
        try:
            content = wikipedia.page(topic).content[:max_chars]
            tokens = clean_and_tokenize(content)

            tokenized_docs.append(tokens)
            raw_docs.append(content)
            print(f"✅ Fetched: {topic}")

        except wikipedia.exceptions.DisambiguationError as e:
            print(f"⚠️ Disambiguation for '{topic}', skipping. Options: {e.options}")
            tokenized_docs.append([])
            raw_docs.append("")
        except wikipedia.exceptions.PageError:
            print(f"❌ Page not found: {topic}")
            tokenized_docs.append([])
            raw_docs.append("")

    return tokenized_docs, raw_docs


# word2vec model training

def train_word2vec_model(tokenized_docs, vector_size=100, window=5, min_count=1):
    return gensim.models.Word2Vec(
        sentences=tokenized_docs,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4
    )


def get_average_vector(tokens, model):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)


def compute_doc_vectors(tokenized_docs, model):
    return np.array([get_average_vector(doc, model) for doc in tokenized_docs])


# classifcation and evaluation

def train_and_evaluate(doc_vectors, labels):
    model = LogisticRegression(max_iter=1000)
    model.fit(doc_vectors, labels)
    preds = model.predict(doc_vectors)

    print("\n=== Classification Report ===")
    print(classification_report(labels, preds, zero_division=1))

    acc = accuracy_score(labels, preds)
    conf_mat = confusion_matrix(labels, preds)
    
    print(f"Accuracy: {acc * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_mat)

    return model, preds


# entire pipeline

def run_pipeline():
    tokenized_docs, raw_docs = fetch_wikipedia_documents(TOPICS, MAX_CHARS)

    # successfully fetched documents
    print("\n=== Document Previews ===")
    for i, doc in enumerate(raw_docs):
        print(f"\nDocument {i+1} (Topic: {TOPICS[i]}):\n{doc[:300]}...\n")

    # fallback for empty documents
    valid_docs = [doc for doc in tokenized_docs if doc]
    if not valid_docs:
        print("❌ No valid documents to process.")
        return

    word2vec_model = train_word2vec_model(valid_docs, VECTOR_SIZE, WINDOW_SIZE, MIN_WORD_COUNT)
    doc_vectors = compute_doc_vectors(valid_docs, word2vec_model)

    # for skipped labels
    filtered_labels = [LABELS[i] for i, doc in enumerate(tokenized_docs) if doc]
    
    # train word2vec model and compute document vectors
    train_and_evaluate(doc_vectors, filtered_labels)


# main function to run the pipeline
if __name__ == "__main__":
    run_pipeline()
    print("\n=== Pipeline Completed ===")