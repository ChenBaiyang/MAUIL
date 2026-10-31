"""Small raw-input example. Replace the graphs, attributes and anchor lists."""

import networkx as nx
from mauil import MAUIL, hit_precision

import numpy as np
from gensim.models import KeyedVectors


def main():
    # Each network has its own ID namespace: both may contain the same IDs.
    source = nx.cycle_graph(24)
    target = nx.cycle_graph(24)
    # Already tokenized text avoids requiring downloaded NLTK resources here.
    attributes = {
        i: ("user" + chr(97 + i),
            "north university" if i < 12 else "south academy",
            "machine learning science" if i < 12 else "social network research")
        for i in source
    }
    train_pairs = [(i, i) for i in range(12)]
    test_pairs = [(i, i) for i in range(12, 24)]
    model = MAUIL(language='tokens', n_components=4, reg=1000,
                  embedding_dim=8, random_state=0, pretrained_language='en')
    # Tiny explicit vectors keep this example offline and fast. In a real English
    # dataset, omit word_vectors to download/cache the pinned Twitter-100 vectors.
    # For raw Chinese use language='zh', stop_words=[...] and Tencent-light-200.
    vectors = KeyedVectors(vector_size=8)
    words = ['north', 'university', 'south', 'academy']
    vectors.add_vectors(words, np.random.RandomState(0).normal(size=(len(words), 8)))
    model.fit(source, target, attributes, attributes, train_pairs, word_vectors=vectors)
    # Transductive: both full networks are embedded; only train_pairs train RCCA.
    queries = [u for u, _ in test_pairs]
    candidates = [v for _, v in test_pairs]
    ranked, distances = model.rank(queries, candidates, top_k=5)
    print('First query:', queries[0], 'candidates:', ranked[0].tolist())
    print('Distances:', distances[0].tolist())
    print('Hit-precision@3:', hit_precision(ranked, candidates, k=3))
    # Reuse the four raw blocks without retraining embeddings:
    cached = MAUIL(language='tokens', n_components=4, reg=1000, embedding_dim=8)
    cached.fit_embeddings(*model.embeddings_, model.source_nodes_,
                          model.target_nodes_, train_pairs)
    print('Ranking from cached embeddings:', cached.rank(queries, candidates, top_k=5)[0][0].tolist())


if __name__ == '__main__':
    main()
