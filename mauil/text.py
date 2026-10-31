"""Text preprocessing, sparse word aggregation, Word2Vec and LDA."""

import hashlib
import re
import string
import warnings
from collections import Counter
from functools import lru_cache

import numpy as np
from gensim import models, utils
from gensim.corpora import Dictionary
from gensim.matutils import corpus2csc
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx


def feature_memory(cache_dir):
    """Cache deterministic, label-free text features in a versioned namespace."""
    from joblib import Memory
    if cache_dir is None:
        return Memory(location=None, verbose=0)
    import json
    import platform
    from importlib.metadata import version
    from pathlib import Path
    from . import __version__
    versions = {name: version(name) for name in
                ('numpy', 'scipy', 'gensim', 'nltk', 'jieba', 'zhconv', 'pypinyin', 'joblib')}
    versions['python'] = platform.python_version()
    signature = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()[:12]
    return Memory(location=str(Path(cache_dir) / (__version__ + '-' + signature)), verbose=0)


def _compute_feature(function, docs, arguments, content_hash, code_hash):
    return function(docs, *arguments)


def cached_feature(memory, function, docs, *arguments):
    """Use Joblib content keys without persisting the full documents as metadata."""
    if memory.location is None:
        return function(docs, *arguments)
    import inspect
    from joblib import hash as content_hash
    return memory.cache(_compute_feature, ignore=['docs'])(
        function, docs, arguments, content_hash(docs), content_hash(inspect.getsource(function)))


def _phrases(docs):
    bigram = models.Phrases(docs, min_count=5, threshold=.1)
    trigram = models.Phrases(bigram[docs], threshold=.1)
    bigram, trigram = bigram.freeze(), trigram.freeze()
    docs = list(bigram[docs])
    # Intentional: preserve the original second bigram application.
    return list(trigram[bigram[docs]])


def preprocess(docs, language, stop_words=None):
    if language == 'tokens':
        return _phrases([doc.split() for doc in docs])
    if language == 'zh':
        if stop_words is None:
            raise ValueError("Chinese preprocessing requires explicit stop_words (use [] for none).")
        import jieba
        import zhconv
        stop = set(stop_words)
        return [[word for word in jieba.lcut(re.sub(
            '[^\u4e00-\u9fa5]', '', zhconv.convert(doc, 'zh-hans').strip()))
                 if word not in stop] for doc in docs]
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatize = lru_cache(maxsize=100000)(lemmatizer.lemmatize)
    try:
        stop = set(stopwords.words('english')) | {'from', 'subject', 're', 'edu', 'use'}
        docs = [[lemmatize(token) for token in utils.tokenize(
            doc, lower=True, deacc=True, errors='ignore') if 2 <= len(token) <= 15]
                for doc in docs]
    except LookupError as error:
        raise LookupError("Prepare NLTK resources explicitly: python -m nltk.downloader "
                          "wordnet stopwords. No resources are downloaded by MAUIL.") from error
    return _phrases([[word for word in doc if word not in stop] for doc in docs])


def character_counts(docs, language):
    if language == 'zh':
        from pypinyin import lazy_pinyin
        docs = [''.join(lazy_pinyin(doc)).lower() for doc in docs]
    docs = _phrases([[c for c in doc if not c.isnumeric() and not c.isspace()
                      and c not in string.punctuation] for doc in docs])
    vocabulary = Dictionary(docs)
    data = corpus2csc((vocabulary.doc2bow(doc) for doc in docs),
                      num_terms=len(vocabulary), num_docs=len(docs), dtype=np.float64).T.toarray()
    return data, vocabulary


def _stable_hash(word):
    return int.from_bytes(hashlib.sha256(word.encode('utf-8')).digest()[:4], 'little')


def train_word_vectors(docs, dim, seed=0, workers=1):
    model = models.Word2Vec(vector_size=dim, window=5, min_count=5, workers=workers,
                           sg=0, hs=0, negative=5, ns_exponent=.75, cbow_mean=1,
                           alpha=.025, min_alpha=.0001, sample=.001, epochs=5,
                           sorted_vocab=1, seed=seed, hashfxn=_stable_hash)
    model.build_vocab(docs, keep_raw_vocab=True)
    if len(model.wv):
        # Gensim 4 changed both tie ordering and initialization. Retain Gensim
        # 3's frequency-stable vocabulary and per-token RandomState vectors.
        keys = sorted((w for w in model.raw_vocab if w in model.wv),
                      key=lambda w: model.raw_vocab[w], reverse=True)
        order = [model.wv.key_to_index[word] for word in keys]
        model.wv.expandos = {key: values[order] for key, values in model.wv.expandos.items()}
        model.wv.index_to_key = keys
        model.wv.key_to_index = {word: i for i, word in enumerate(keys)}
        for i, word in enumerate(keys):
            rng = np.random.RandomState(_stable_hash(word + str(seed)))
            model.wv.vectors[i] = (rng.rand(dim) - .5) / dim
        model.make_cum_table()
        model.raw_vocab.clear()
        model.train(docs, total_examples=model.corpus_count, epochs=5)
    else:
        warnings.warn("No words survive Word2Vec min_count=5; word features have no signal.",
                      RuntimeWarning)
    return model.wv


def _word_lookup(word, vectors, phrase_fallback=False):
    if word in vectors:
        return [word]
    if phrase_fallback and '_' in word:
        return [part for part in word.split('_') if part in vectors]
    return []


def word_coverage(docs, vectors, phrase_fallback=False):
    """Report vocabulary and occurrence coverage using the actual lookup policy."""
    counts = Counter(word for doc in docs for word in doc)
    matched = {word for word in counts if _word_lookup(word, vectors, phrase_fallback)}
    total = sum(counts.values())
    hits = sum(counts[word] for word in matched)
    return dict(vocabulary_size=len(counts), matched_vocabulary=len(matched),
                missing_vocabulary=len(counts) - len(matched),
                vocabulary_coverage=len(matched) / len(counts) if counts else 0.,
                token_count=total, matched_tokens=hits, missing_tokens=total - hits,
                token_coverage=hits / total if total else 0.,
                phrase_fallback_tokens=sum(counts[w] for w in matched if w not in vectors),
                phrase_fallback=phrase_fallback)


def word_embedding(docs, vectors, dim, graphs, node_orders, neighbor_weight,
                   phrase_fallback=False):
    if not len(vectors) or not any(docs):
        return np.zeros((len(docs), dim), dtype=np.float64)
    terms = dict.fromkeys(word for doc in docs for word in doc)
    vocabulary = {word: i for i, word in enumerate(terms)}
    # Resolve each native vector once, including repeated fallback components.
    lookup = {word: _word_lookup(word, vectors, phrase_fallback) for word in terms}
    keys = dict.fromkeys(key for matches in lookup.values() for key in matches)
    native = {}
    for key in keys:
        vector = np.asarray(vectors[key], dtype=np.float64)
        if vector.shape != (dim,) or not np.isfinite(vector).all():
            raise ValueError("Word vectors must be finite and match their native vector size.")
        native[key] = vector
    table = np.zeros((len(terms), dim), dtype=np.float64)
    for word, matches in lookup.items():
        if matches:
            table[vocabulary[word]] = np.sum([native[key] for key in matches], axis=0)
    counts = CountVectorizer(analyzer=lambda doc: doc, vocabulary=vocabulary,
                             dtype=np.float64).fit_transform(docs)
    original = np.asarray(counts @ table) + 1e-13
    result = np.empty_like(original)
    offset = 0
    for graph, nodes in zip(graphs, node_orders):
        if not nodes:
            continue
        adjacency = nx.to_scipy_sparse_array(graph, nodelist=nodes, weight=None,
                                             dtype=np.float64, format='csr')
        side = original[offset:offset + len(nodes)]
        result[offset:offset + len(nodes)] = ((1 - neighbor_weight) * side
                                              + neighbor_weight * (adjacency @ side + 1e-13))
        offset += len(nodes)
    return result


def topic_embedding(docs, dim):
    vocabulary = Dictionary(docs)
    vocabulary.filter_extremes(no_below=10, no_above=.5)
    if not len(vocabulary):
        warnings.warn("No topics survive dictionary filtering; topic features are uniform.",
                      RuntimeWarning)
        return np.full((len(docs), dim), 1 / dim, dtype=np.float64)
    corpus = [vocabulary.doc2bow(doc) for doc in docs]
    model = models.LdaModel(corpus=corpus, id2word=vocabulary, chunksize=256,
                            passes=4, iterations=100, num_topics=dim,
                            alpha='symmetric', eta=None, eval_every=0,
                            minimum_probability=1e-13, random_state=0)
    result = np.full((len(docs), dim), 1e-13, dtype=np.float64)
    for row, topics in zip(result, model.get_document_topics(corpus)):
        for index, probability in topics:
            row[index] = probability
    return result
