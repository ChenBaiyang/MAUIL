"""Transductive MAUIL with paper presets."""

# Load Torch before SciPy/sklearn: their MKL DLLs can otherwise prevent Torch
# 2.11's c10.dll from initializing on Windows.
import torch
import numpy as np
from sklearn.neighbors import KDTree

from .alignment import rcca, standardize_blocks


PRESETS = {'wd': dict(language='zh', n_components=25, reg=100000.),
           'dblp': dict(language='en', n_components=80, reg=1000.)}


def _object_ids(nodes):
    # np.array(list_of_tuple_ids) would accidentally create a two-dimensional array.
    values = np.empty(len(nodes), dtype=object)
    values[:] = list(nodes)
    return values


class MAUIL:
    """Embed both complete networks, fit RCCA on training anchors, then rank.

    Custom datasets require language ('en', 'zh', or pretokenized 'tokens'),
    n_components and reg. Presets supply paper defaults; explicit values win.
    embedding_dim controls character/topic/LINE widths. The word block keeps
    the native pretrained or supplied vector width (English 100, Chinese 200).
    """

    def __init__(self, preset=None, *, language=None, n_components=None, reg=None,
                 embedding_dim=100, neighbor_weight=.1, random_state=0,
                 device='cpu', workers=1, stop_words=None, char_epochs=50,
                 line_epochs=50, batch_size=1024, word_vector_source='pretrained',
                 pretrained_language=None, word_vector_cache=None, local_files_only=False,
                 line_backend='auto', feature_cache=None):
        if preset is not None and preset not in PRESETS:
            raise ValueError("preset must be 'wd', 'dblp', or None.")
        defaults = PRESETS.get(preset, {})
        self.preset = preset
        self.language = language if language is not None else defaults.get('language')
        self.n_components = n_components if n_components is not None else defaults.get('n_components')
        self.reg = reg if reg is not None else defaults.get('reg')
        if self.language not in ('en', 'zh', 'tokens'):
            raise ValueError("Set language to 'en', 'zh', or 'tokens'.")
        if not isinstance(self.n_components, int) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if self.reg is None or not np.isfinite(self.reg) or self.reg <= 0:
            raise ValueError("reg must be finite and positive.")
        if not isinstance(embedding_dim, int) or embedding_dim < 2 or embedding_dim % 2:
            raise ValueError("embedding_dim must be a positive even integer for joint LINE.")
        if not np.isfinite(neighbor_weight) or not 0 <= neighbor_weight <= 1:
            raise ValueError("neighbor_weight must lie within [0, 1].")
        if any(not isinstance(v, int) or v < 1 for v in (workers, char_epochs, line_epochs, batch_size)):
            raise ValueError("workers, epochs and batch_size must be positive integers.")
        if not isinstance(random_state, int) or not 0 <= random_state < 2**32 - 4:
            raise ValueError("random_state must be an integer between 0 and 2**32-5.")
        self.embedding_dim, self.neighbor_weight = embedding_dim, neighbor_weight
        self.random_state, self.device, self.workers = random_state, device, workers
        self.stop_words = None if stop_words is None else tuple(stop_words)
        self.char_epochs, self.line_epochs, self.batch_size = char_epochs, line_epochs, batch_size
        if word_vector_source not in ('pretrained', 'local'):
            raise ValueError("word_vector_source must be 'pretrained' or 'local'.")
        if pretrained_language is not None and pretrained_language not in ('en', 'zh'):
            raise ValueError("pretrained_language must be 'en', 'zh', or None.")
        self.word_vector_source, self.pretrained_language = word_vector_source, pretrained_language
        self.word_vector_cache, self.local_files_only = word_vector_cache, local_files_only
        from .embedding import resolve_line_backend
        resolve_line_backend(line_backend, device)
        self.line_backend, self.feature_cache = line_backend, feature_cache

    def fit(self, source_graph, target_graph, source_attributes, target_attributes,
            train_pairs, word_vectors=None):
        """Fit on raw graphs and node -> (character, word, topic text) mappings.

        Defaults to pinned, cached pretrained vectors. An explicit word_vectors
        object overrides the selected source and keeps its native vector width.
        Local Word2Vec training is available only with word_vector_source='local'.
        """
        from .embedding import character_embedding, network_embedding, resolve_line_backend
        from .text import (character_counts, preprocess, topic_embedding,
                           train_word_vectors, word_embedding, word_coverage,
                           feature_memory, cached_feature)
        from .pretrained import load_pretrained_vectors
        graphs = source_graph, target_graph
        orders = [list(graph) for graph in graphs]
        if not all(orders):
            raise ValueError("Both graphs must contain nodes.")
        attrs = []
        for graph, nodes, attributes in zip(graphs, orders, (source_attributes, target_attributes)):
            if graph.is_multigraph():
                raise ValueError("Use a simple Graph or DiGraph, not a multigraph.")
            if any(data.get('weight', 1) != 1 for _, _, data in graph.edges(data=True)):
                raise ValueError("Graphs must be unweighted.")
            for node in nodes:
                values = attributes[node]
                if len(values) != 3 or any(not isinstance(value, str) for value in values):
                    raise ValueError("Every node must have three string attributes; use '' for missing text.")
                attrs.append(values)
        # Validate supervised input before performing expensive embedding work.
        pairs = self._validate_pairs(train_pairs, orders)
        if self.word_vector_source == 'local' and self.preset == 'wd' and word_vectors is None:
            raise ValueError("WD local mode requires externally trained word_vectors.")
        pretrained_language = self.pretrained_language or self.language
        if word_vectors is None and self.word_vector_source == 'pretrained' and pretrained_language == 'tokens':
            raise ValueError("Tokenized input needs pretrained_language='en'/'zh', explicit "
                             "word_vectors, or word_vector_source='local'.")
        memory = feature_memory(self.feature_cache)
        counts, self.character_vocabulary_ = cached_feature(memory, character_counts,
            [a[0] for a in attrs], self.language)
        word_docs = cached_feature(memory, preprocess, [a[1] for a in attrs], self.language, self.stop_words)
        topic_docs = cached_feature(memory, preprocess, [a[2] for a in attrs], self.language, self.stop_words)
        supplied = word_vectors is not None
        if word_vectors is None and self.word_vector_source == 'pretrained':
            word_vectors = load_pretrained_vectors(pretrained_language, self.word_vector_cache,
                                                  self.local_files_only)
        elif word_vectors is None:
            word_vectors = train_word_vectors(word_docs, self.embedding_dim,
                                              self.random_state + 1, self.workers)
        word_dim = getattr(word_vectors, 'vector_size', None)
        if word_dim is None:
            if not len(word_vectors):
                raise ValueError("An empty vector mapping needs a KeyedVectors object with vector_size.")
            vector = np.asarray(next(iter(word_vectors.values())))
            if vector.ndim != 1:
                raise ValueError("Word vector mapping values must be one-dimensional.")
            word_dim = len(vector)
        if not isinstance(word_dim, (int, np.integer)) or word_dim < 1:
            raise ValueError("Word vectors need a positive native vector size.")
        # Explicit local mode preserves the historical exact-token lookup.
        fallback = self.word_vector_source != 'local' and (self.language == 'en' or
                                                          pretrained_language == 'en')
        coverage = word_coverage(word_docs, word_vectors, fallback)
        if coverage['token_count'] and not coverage['matched_tokens']:
            raise ValueError("All word-level tokens are out of vocabulary; check the vector language and tokenizer.")
        metadata = dict(getattr(word_vectors, 'mauil_metadata',
                               {'mode': 'provided' if supplied else 'local', 'vector_size': int(word_dim)}))
        metadata['lookup_policy'] = 'phrase-fallback' if fallback else 'exact'
        char = character_embedding(counts, self.embedding_dim, self.random_state,
                                   self.device, self.char_epochs)
        word = word_embedding(word_docs, word_vectors, word_dim,
                              graphs, orders, self.neighbor_weight, fallback)
        topic = cached_feature(memory, topic_embedding, topic_docs, self.embedding_dim)
        self.line_backend_ = resolve_line_backend(self.line_backend, self.device)
        structure = [network_embedding(g, self.embedding_dim, self.random_state + i + 2,
                                       self.device, self.line_epochs, self.batch_size,
                                       backend=self.line_backend)
                     for i, g in enumerate(graphs)]
        n = len(orders[0])
        source = (char[:n], word[:n], topic[:n], structure[0])
        target = (char[n:], word[n:], topic[n:], structure[1])
        self.fit_embeddings(source, target, orders[0], orders[1], pairs)
        self.word_vector_metadata_, self.word_coverage_ = metadata, coverage
        return self

    @staticmethod
    def _validate_nodes(orders):
        for nodes in orders:
            if not len(nodes) or len(set(nodes)) != len(nodes):
                raise ValueError("Each network needs a nonempty sequence of unique hashable node IDs.")

    @staticmethod
    def _validate_pairs(train_pairs, orders):
        pairs = list(train_pairs.items()) if hasattr(train_pairs, 'items') else list(train_pairs)
        if not pairs or any(len(pair) != 2 for pair in pairs):
            raise ValueError("train_pairs must contain source/target ID pairs.")
        left, right = set(orders[0]), set(orders[1])
        if any(u not in left or v not in right for u, v in pairs):
            raise ValueError("Training anchors must reference nodes in the respective networks.")
        if len(set(u for u, _ in pairs)) != len(pairs) or len(set(v for _, v in pairs)) != len(pairs):
            raise ValueError("Training anchors must be one-to-one and unique.")
        return pairs

    def fit_embeddings(self, source_embeddings, target_embeddings, source_nodes,
                       target_nodes, train_pairs, *, standardized=False):
        """Fit RCCA using four blocks ordered (character, word, topic, structure).

        Each block has one row per corresponding node ID. Raw blocks are jointly
        standardized by default. Set standardized=True for blocks that have
        already been standardized jointly over both networks.
        """
        orders = [list(source_nodes), list(target_nodes)]
        self._validate_nodes(orders)
        blocks = []
        for embeddings, nodes in zip((source_embeddings, target_embeddings), orders):
            if len(embeddings) != 4:
                raise ValueError("Supply four blocks in character/word/topic/structure order.")
            arrays = [np.asarray(block) for block in embeddings]
            for array in arrays:
                if (array.ndim != 2 or array.shape[0] != len(nodes) or not array.shape[1]
                        or array.dtype.kind not in 'fi' or not np.isfinite(array).all()):
                    raise ValueError("Embedding blocks must be finite real matrices with one row per node.")
            blocks.append(arrays)
        if any(a.shape[1] != b.shape[1] for a, b in zip(*blocks)):
            raise ValueError("Corresponding source and target blocks must have equal widths.")
        self.embeddings_ = tuple(tuple(a.copy() for a in side) for side in blocks)
        if standardized:
            x, y = [np.concatenate(side, axis=1) for side in blocks]
            self.standardization_ = None
        else:
            x, y, self.standardization_ = standardize_blocks(*blocks)
        if self.n_components > min(x.shape[1], y.shape[1]):
            raise ValueError("n_components cannot exceed either combined feature width.")
        self.source_nodes_, self.target_nodes_ = map(_object_ids, orders)
        self._indexes = [{node: i for i, node in enumerate(nodes)} for nodes in orders]
        self._features = x, y
        self.word_embedding_dim_ = blocks[0][1].shape[1]
        return self.refit_alignment(train_pairs)

    def refit_alignment(self, train_pairs):
        """Refit RCCA on new anchors, reusing the fitted normalized features.

        Embeddings, node orders and transductive standardization are unchanged.
        Use this for multiple disjoint train/test splits of the same networks.
        """
        if not hasattr(self, '_features'):
            raise RuntimeError('Call fit or fit_embeddings before refit_alignment.')
        pairs = self._validate_pairs(train_pairs, (self.source_nodes_, self.target_nodes_))
        x, y = self._features
        ix = [self._indexes[0][u] for u, _ in pairs]
        iy = [self._indexes[1][v] for _, v in pairs]
        wx, wy = rcca(x[ix], y[iy], self.n_components, self.reg)
        zx, zy = x @ wx, y @ wy
        self.projections_, self.projected_ = (wx, wy), (zx, zy)
        self.train_pairs_ = tuple(tuple(pair) for pair in pairs)
        return self

    def rank(self, source_nodes, candidate_nodes, top_k=10):
        """Return (target IDs, Euclidean distances), both of shape (queries, k).

        Both sets must explicitly reference fitted nodes. k is capped at the
        candidate count. Ties use the original sparse-similarity ordering.
        """
        if not hasattr(self, 'projected_'):
            raise RuntimeError("Call fit or fit_embeddings before rank.")
        source_nodes, candidate_nodes = list(source_nodes), list(candidate_nodes)
        if not candidate_nodes or len(set(candidate_nodes)) != len(candidate_nodes):
            raise ValueError("Supply a nonempty set of unique candidate IDs.")
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer.")
        k = min(top_k, len(candidate_nodes))
        ix = [self._indexes[0][u] for u in source_nodes]
        iy = [self._indexes[1][v] for v in candidate_nodes]
        if not source_nodes:
            return np.empty((0, k), dtype=object), np.empty((0, k))
        distances, indices = KDTree(self.projected_[1][iy]).query(self.projected_[0][ix], k=k)
        # Legacy sparse.find returned candidate-column order before sorting
        # exp(-distance). Keep its returned-hit tie behavior, vectorized by row.
        columns = np.argsort(indices, axis=1)
        similarities = np.exp(-np.take_along_axis(distances, columns, axis=1))
        order = np.take_along_axis(columns, np.argsort(similarities, axis=1)[:, ::-1], axis=1)
        indices = np.take_along_axis(indices, order, axis=1)
        distances = np.take_along_axis(distances, order, axis=1)
        return _object_ids(candidate_nodes)[indices], distances
