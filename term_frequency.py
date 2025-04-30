from collections import Counter

# Computer Term Frequency
# Given a list of tokens and a vocabulary, compute the term frequency for each term in the vocabulary.
def compute_tf(tokens, vocab):
    count = Counter(tokens)
    total_terms = len(tokens)
    return { term: count[term] / total_terms for term in vocab }