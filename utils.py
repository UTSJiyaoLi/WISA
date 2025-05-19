import numpy as np

def weighted_sampling(word_scores, sample_size=10, alpha=0.2):
    """
    Algorithm 2: Weighted Sampling.

    word_scores: list of (sent_id, word_idx, word, score), sorted descending.
    sample_size: number of words to sample.
    alpha: decay parameter.
    """
    n = len(word_scores)
    W = []

    for i in range(n):
        w = np.exp(-alpha * i)  # higher weight to important words
        W.append(w)

    W = np.array(W)
    W = W / W.sum()  # normalization

    indices = np.random.choice(n, size=min(sample_size, n), replace=False, p=W)

    sampled = [word_scores[i] for i in indices]

    return sampled
