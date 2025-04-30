import math

# Calculate the Cosine Similarity
def cosine_similarity(vec1, vec2, vocab):
    # Get the dot product of the two vectors
    dot_product = sum(vec1[term] * vec2[term] for term in vocab)
    vec1Length = math.sqrt(sum(vec1[term]**2 for term in vocab))
    vec2Length = math.sqrt(sum(vec2[term]**2 for term in vocab))

    if vec1Length == 0 or vec2Length == 0:
        return 0.0

    return dot_product / (vec1Length * vec2Length)