"""Read the two original, trusted dataset archives without unpacking them."""

import json
import pickle
from pathlib import Path
from zipfile import ZipFile


def load_paper_dataset(archive, dataset):
    """Return graphs, separate attribute mappings, anchors, and Chinese stopwords.

    Archives contain Python pickles; only use the original trusted archives.
    Node and anchor insertion order is retained for historical split identities.
    """
    if dataset not in ('wd', 'dblp'):
        raise ValueError("dataset must be 'wd' or 'dblp'.")
    with ZipFile(Path(archive)) as data:
        source, target = pickle.loads(data.read(dataset + '/networks'))
        attrs = pickle.loads(data.read(dataset + '/attrs'))
        pairs = json.loads(data.read(dataset + '/anchors.txt'))
        stop = (pickle.loads(data.read('wd/stop_words_cn.pkl')) if dataset == 'wd' else None)
    return (source, target, {u: attrs[u] for u in source},
            {u: attrs[u] for u in target}, pairs, stop)
