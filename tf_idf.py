from math import log

# Compute the Inverse Document Frequency

def compute_idf(tokenized_docs, vocab):
    N = len(tokenized_docs)
    idf_dict = {}
    for term in vocab:
        # Count the number of documents containing the term
        df = sum(term in doc for doc in tokenized_docs)
        # Compute IDF using the formula: idf(t) = lo(N / df(t))
        idf_dict[term] = log(N / (df or 1))
    return idf_dict

# Computer TF-IDF
def compute_tfidf(tf_vector, idf, vocab):
    # Compute the TF-IDF score for each term in the vocabulary
    # using the formula: tf-idf(t, d) = tf(t, d) * idf(t)
    return { term: tf_vector[term] * idf[term] for term in vocab}