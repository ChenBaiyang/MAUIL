"""Pinned static word vectors, downloaded once into the Hugging Face cache."""

import hashlib
from pathlib import Path

from gensim.models import KeyedVectors


_RESOURCES = {
    'en': {
        'repo_id': 'fse/glove-twitter-100',
        'revision': '23ea42d022c3a0a525890bfc08395c2adbac23e2',
        'vector_size': 100, 'format': 'gensim', 'license': 'PDDL',
        'files': {
            'glove-twitter-100.model': {
                'size': 68268001,
                'sha256': 'be82099660c80b42c654e276f8b664edd790c15700a434ff9fc8693d9b7401b7'},
            'glove-twitter-100.model.vectors.npy': {
                'size': 477405728,
                'sha256': 'aed6fc2e861ac971a7f5256b902932b148bd6d88d0dd601622ea4ea2582faf7d'},
        },
    },
    'zh': {
        'repo_id': 'shibing624/text2vec-word2vec-tencent-chinese',
        'revision': 'b7b9fccfd5dd34cfc340607c58986ac2970e3a13',
        'vector_size': 200, 'format': 'word2vec-binary', 'license': 'Apache-2.0 (model card)',
        'files': {
            'light_Tencent_AILab_ChineseEmbedding.bin': {
                'size': 116021049,
                'sha256': '5515923c7e67cdc7eb42996546e0bad273c8452f3bfad6db0794e51c848d151b'},
        },
    },
}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pretrained_metadata(language):
    """JSON-compatible immutable resource identity, also used by experiment caches."""
    import copy
    if language not in _RESOURCES:
        raise ValueError("Pretrained vectors require language='en' or 'zh'.")
    return dict(copy.deepcopy(_RESOURCES[language]), mode='pretrained', language=language)


def _verify(path, expected):
    if Path(path).stat().st_size != expected['size'] or _sha256(path) != expected['sha256']:
        raise ValueError("Pretrained vector checksum mismatch: %s. Remove the damaged "
                         "file from the cache and download it again." % path)


def load_pretrained_vectors(language, cache_dir=None, local_files_only=False):
    """Load English Twitter-100 or Chinese Tencent-light-200 KeyedVectors.

    The first online call downloads the fixed, checksummed HF snapshot; later
    calls use the cache. local_files_only=True never contacts the network.
    The Chinese resource is general-domain, not a Weibo-specific embedding.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.utils import LocalEntryNotFoundError

    metadata = pretrained_metadata(language)
    paths = []
    for filename, expected in metadata['files'].items():
        arguments = dict(repo_id=metadata['repo_id'], revision=metadata['revision'],
                         filename=filename, cache_dir=cache_dir)
        try:
            path = hf_hub_download(**arguments, local_files_only=True)
        except LocalEntryNotFoundError:
            if local_files_only:
                raise FileNotFoundError("Pretrained vectors are not cached: %s / %s. "
                                        "Run once online or supply local word_vectors." %
                                        (metadata['repo_id'], filename)) from None
            print('Downloading %s / %s (%.1f MB) into %s' %
                  (metadata['repo_id'], filename, expected['size'] / 1e6,
                   cache_dir if cache_dir is not None else HF_HUB_CACHE), flush=True)
            path = hf_hub_download(**arguments)
        _verify(path, expected)
        paths.append(path)
    if metadata['format'] == 'gensim':
        vectors = KeyedVectors.load(paths[0], mmap='r')
    else:
        vectors = KeyedVectors.load_word2vec_format(paths[0], binary=True)
    vectors.mauil_metadata = metadata
    return vectors


def load_word_vectors(path, format='gensim'):
    """Load a trusted local .kv model, or explicit word2vec text/binary format.

    Gensim models use pickle; load only trusted local models. Text input may
    have the standard count/dimension header or be headerless GloVe-style text.
    """
    path = Path(path)
    if format == 'gensim':
        vectors = KeyedVectors.load(str(path), mmap='r')
    elif format in ('word2vec-text', 'word2vec-binary'):
        arguments = dict(binary=format == 'word2vec-binary')
        if not arguments['binary']:
            with path.open('r', encoding='utf-8') as stream:
                first = stream.readline().split()
            arguments['no_header'] = not (len(first) == 2 and all(x.isdigit() for x in first))
        vectors = KeyedVectors.load_word2vec_format(str(path), **arguments)
    else:
        raise ValueError("format must be 'gensim', 'word2vec-text', or 'word2vec-binary'.")
    # Gensim may store large matrices in sidecar files; include those in identity.
    files = [path] + sorted(path.parent.glob(path.name + '.*.npy')) if format == 'gensim' else [path]
    vectors.mauil_metadata = dict(mode='provided', format=format, vector_size=vectors.vector_size,
                                 files={p.name: {'size': p.stat().st_size, 'sha256': _sha256(p)}
                                        for p in files})
    return vectors
