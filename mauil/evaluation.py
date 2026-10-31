"""Paper Hit-precision (rank-weighted), and reproducible anchor splits."""

import numpy as np


def split_anchors(pairs, n_train=200, n_test=500, seed=0):
    """Legacy RandomState shuffle: first train pairs, last test pairs."""
    pairs = list(pairs.items()) if hasattr(pairs, "items") else list(pairs)
    if (n_train < 1 or n_test < 1 or n_train + n_test > len(pairs)):
        raise ValueError("Require positive, disjoint train/test sizes within anchors.")
    order = np.random.RandomState(seed).permutation(len(pairs))
    return ([pairs[i] for i in order[:n_train]],
            [pairs[i] for i in order[-n_test:]])


def hit_precision(ranked_nodes, true_targets, k=3):
    """Mean (k - rank + 1) / k for truths found in the first k results.

    rank is one-based. Missing truths score zero. This is NOT Hits@K or MRR.
    Supply one true target per query, in the same query order as rank().
    """
    ranked_nodes = np.asarray(ranked_nodes, dtype=object)
    true_targets = list(true_targets)
    if ranked_nodes.ndim != 2 or len(ranked_nodes) != len(true_targets):
        raise ValueError("One ranked row and one true target are required per query.")
    if not len(true_targets) or not 1 <= k <= ranked_nodes.shape[1]:
        raise ValueError("k must be within the returned ranking width; queries cannot be empty.")
    truth = np.empty(len(true_targets), dtype=object)
    truth[:] = true_targets
    matches = ranked_nodes[:, :k] == truth[:, None]
    weights = np.arange(k, 0, -1, dtype=float) / k
    return float(np.where(matches, weights, 0).max(axis=1).mean())
