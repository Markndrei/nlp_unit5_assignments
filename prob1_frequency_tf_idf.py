import wikipedia
import pandas as pd
from term_frequency import compute_tf
from tf_idf import compute_idf, compute_tfidf

# List of Wikipedia topics
topics = ["University", "College", "Highschool", "Academy", "Institute"]

# Fetch summaries from Wikipedia
documents = []
for topic in topics:
    try:
        summary = wikipedia.summary(topic)
        documents.append(summary)
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"DisambiguationError for '{topic}', using first option: {e.options[0]}")
        documents.append(wikipedia.summary(e.options[0]))
    except wikipedia.exceptions.PageError:
        print(f"PageError: The page '{topic}' does not exist on Wikipedia.")
        documents.append("")

print("Fetched Wikipedia Documents:")
for i, doc in enumerate(documents):
    print(f"\nDocument {i+1} (Topic: {topics[i]}):\n{doc[:300]}...")  # Print first 300 chars

# Tokenize and lowercase the documents
tokenized_docs = [doc.lower().split() for doc in documents]

# Create vocabulary
vocabulary = sorted(set(word for doc in tokenized_docs for word in doc))

# Compute term frequency vectors
tf_vectors = [compute_tf(doc, vocabulary) for doc in tokenized_docs]

# Compute inverse document frequency
idf = compute_idf(tokenized_docs, vocabulary)

# Compute TF-IDF vectors
tfidf_vectors = [compute_tfidf(tf, idf, vocabulary) for tf in tf_vectors]

# Convert TF vectors into a DataFrame (Term Frequency Matrix)
tf_df = pd.DataFrame(tf_vectors, columns=vocabulary, index=[f"Doc {i+1}" for i in range(len(documents))])
print("\n=== Term Frequency Document Matrix ===")
print(tf_df)

# Convert TF-IDF vectors into a DataFrame (TF-IDF Matrix)
tfidf_df = pd.DataFrame(tfidf_vectors, columns=vocabulary, index=[f"Doc {i+1}" for i in range(len(documents))])
print("\n=== TF-IDF Document Matrix ===")
print(tfidf_df)
