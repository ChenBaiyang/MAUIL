# MAUIL v2.0

Implementation of **MAUIL: Multilevel attribute embedding for semisupervised
user identity linkage**, Information Sciences 593 (2022), 527–545.
[Paper](https://doi.org/10.1016/j.ins.2022.02.023).

MAUIL combines character, word, topic and network embeddings, aligns two networks
using labeled user pairs and RCCA, then ranks candidate users by Euclidean distance.

## Changes from v1.0 to v2.0

- **PyTorch replaces TensorFlow/Keras** for the character autoencoder and LINE.
- **Separate English and Chinese pretrained word vectors** are downloaded and
  cached automatically. The default workflow needs no Wikipedia corpus or local
  Word2Vec training.
- Framework Adam, SciPy sampling, sparse feature aggregation and a faster RCCA
  solver improve the implementation. RCCA's objective, regularization scale and
  projection normalization are unchanged.
- A unified `MAUIL` API supports custom datasets, reusable embeddings and repeated
  anchor splits, with explicit seeds and versioned caches.

Pretrained vectors and the framework optimizer change the training trajectory;
v2.0 results are reported below alongside the paper's results.

## Installation

Python 3.10 is recommended. Install in a dedicated environment:

```bash
conda create -n mauil python=3.10 -y
conda activate mauil
python -m pip install -c requirements.txt .
python example.py
```

`requirements.txt` pins the recommended dependencies. Python 3.8 / PyTorch 1.8
is also supported; use `python -m pip install .` in an existing environment
without these Python 3.10 constraints.

CPU is the default. For a GPU, install a CUDA build using the
[PyTorch installer](https://pytorch.org/get-started/locally/) and pass
`device='cuda'` to `MAUIL`, or `--device cuda` to the experiment script.
PyTorch handles the character and LINE training; text processing and RCCA use CPU.

English raw-text preprocessing needs two NLTK resources, prepared once:

```bash
python -m nltk.downloader wordnet stopwords
```

## Using MAUIL

`example.py` is a complete, offline example with small graphs and explicit word
vectors. Replace its graphs, attributes and training pairs with your own data:

```python
from mauil import MAUIL

model = MAUIL('dblp', random_state=0)
model.fit(source_graph, target_graph, source_attributes, target_attributes,
          train_pairs)
target_ids, distances = model.rank(query_nodes, candidate_nodes, top_k=10)
```

- Graphs are unweighted NetworkX `Graph` or `DiGraph` objects. Node IDs must be
  unique within each graph; the two graphs may use the same IDs.
- Attributes map each node ID to three strings: `(character_text, word_text,
  topic_text)`. Use `''` for missing text. Missing attributes do not remove users.
- Training pairs are one-to-one `(source_id, target_id)` anchors. Test labels are
  used only for evaluation.
- Queries and candidates must be fitted nodes. Both returned arrays have shape
  `(number_of_queries, min(top_k, number_of_candidates))`; smaller distances rank
  first.

This is a **transductive** method: both complete graphs and all attributes are
used for embedding and standardization; only training anchors fit RCCA.
It does not embed unseen nodes at retrieval time.

| Preset | Language | Character / word / topic / LINE widths | RCCA dimensions | Regularization |
|---|---|---|---:|---:|
| `wd` | Chinese | 100 / 200 / 100 / 100 | 25 | 100000 |
| `dblp` | English | 100 / 100 / 100 / 100 | 80 | 1000 |

Explicit parameters override presets. Custom datasets must set `language`
(`'en'`, `'zh'` or whitespace-tokenized `'tokens'`), `n_components` and `reg`.
Chinese preprocessing also requires `stop_words` (use `[]` for none).
For `'tokens'`, supply vectors, set `pretrained_language='en'` or `'zh'`, or
select local Word2Vec training. `embedding_dim` controls the character, topic and
total LINE widths; the word branch keeps its vectors' native width.

Reuse embeddings or refit only the alignment:

```python
model.refit_alignment(other_train_pairs)

other = MAUIL('dblp')
other.fit_embeddings(*model.embeddings_, model.source_nodes_, model.target_nodes_,
                     other_train_pairs)
```

The four raw blocks are ordered **character, word, topic, structure**. Each row
must match the supplied node order. Standardization uses both networks, with
constant columns set to zero. Word features retain neighbor summation with
weight `0.1`. RCCA uses `X.T @ X + reg * I`, without additional anchor centering
or division by sample count. Cholesky/SVD solves the same RCCA problem, with a
generalized eigensolver for degenerate projections.

## Word vectors and caches

| Language | Default resource | Dimensions | Download | Resource license |
|---|---|---:|---:|---|
| English | [fse/glove-twitter-100](https://huggingface.co/fse/glove-twitter-100), Twitter text | 100 | ~546 MB | PDDL |
| Chinese | [shibing624/text2vec-word2vec-tencent-chinese](https://huggingface.co/shibing624/text2vec-word2vec-tencent-chinese), light Tencent vectors | 200 | ~116 MB | Apache-2.0, per model card |

The Chinese resource is general-purpose, not Weibo-specific. Downloads use pinned
Hugging Face revisions and SHA256 checksums. English phrases use a complete
entry when available, otherwise known underscore-separated words are summed.
Coverage is available in `model.word_coverage_`; entirely unmatched nonempty text
raises an error.

```python
from mauil import load_pretrained_vectors, load_word_vectors

# Download once; later calls reuse this cache.
vectors = load_pretrained_vectors('zh', cache_dir='models')
# Offline: cached files must already exist.
vectors = load_pretrained_vectors('zh', cache_dir='models', local_files_only=True)
# Alternatively load your own vectors and pass word_vectors=vectors to fit().
vectors = load_word_vectors('custom.bin', format='word2vec-binary')
```

Explicit `word_vectors` override automatic loading. Supported local formats are
`gensim`, `word2vec-text` (including headerless GloVe) and `word2vec-binary`.
Use only trusted Gensim models and Joblib caches, which contain pickle data.

`MAUIL(..., feature_cache='embeddings/text')` caches text preprocessing and LDA.
Character and LINE training retain each run's seed. The experiment script also
caches all four embedding blocks. Cache keys include input content, resource
identity, configuration and software versions; incompatible caches are not reused.
`line_backend='auto'` selects fused Adam on supported PyTorch 2.11+ CPUs;
`'torch'` selects ordinary Adam for LINE.

## Paper experiments

The original `data/wd.zip` and `data/dblp.zip` are read directly, without unpacking.
Each default run uses 200 training anchors, 500 test pairs and exactly the 500
target test users as candidates. The seed controls a shuffle: training pairs are
selected from its beginning and test pairs from its end.

```bash
# One embedding repeat and one split; results are never overwritten.
python reproducing_paper_results.py --dataset dblp --output results/dblp.csv
python reproducing_paper_results.py --dataset wd --output results/wd.csv

# Three embedding repeats, ten splits each. Use --dataset wd for WD.
python reproducing_paper_results.py --dataset dblp --embedding-repeats 3 --split-seeds 0 1 2 3 4 5 6 7 8 9 --output results/dblp-full.csv
```

CSV files contain every split, per-embedding means and an overall mean, together
with seeds, parameters, word coverage and resource checksums. The metric is the
paper's **rank-weighted Hit-precision**, not Hits@K: a true match at rank `r <= k`
scores `(k - r + 1) / k`; otherwise it scores zero.

| Dataset | Paper HP@1 / @3 / @5 | v2.0 HP@1 / @3 / @5 |
|---|---|---|
| WD | 0.313 / 0.373 / 0.408 | 0.306533 / 0.374000 / 0.414787 |
| DBLP | 0.660 / 0.702 / 0.725 | 0.697533 / 0.740800 / 0.763800 |

The v2.0 means use all 3 × 10 runs on Python 3.10 / PyTorch 2.11 CPU. All six
values lie within −2 to +5 percentage points of the paper. GPU results have not
been verified. Table 4 is not supported because its negative-pair and prediction
decision protocol could not be recovered.

Additional options (see `--help`):

- Offline downloads: `--word-vector-cache models --local-files-only`.
- Custom vectors: `--word-vectors custom.bin --word-vector-format word2vec-binary`.
- Local Word2Vec: `--word-vector-source local`. DBLP uses word attributes; WD
  additionally requires `--word-corpus tokenized-wiki.txt`, one document per line,
  or supplied `--word-vectors`.

## Citation and license

```bibtex
@article{Chen2022MAUIL,
  title = {{MAUIL}: Multilevel attribute embedding for semisupervised user identity linkage},
  author = {Baiyang Chen and Xiaoliang Chen},
  journal = {Information Sciences},
  volume = {593},
  pages = {527--545},
  year = {2022},
  doi = {10.1016/j.ins.2022.02.023}
}
```

Project code uses the MIT license. [NOTICE](NOTICE) and
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) contain GraphEmbedding's MIT
notice and Pyrcca's educational/research/non-profit license terms.
Pretrained vectors have their own licenses listed above.

The v1.0 → v2.0 upgrade preserves the paper's computational workflow while using
modern implementations and a more user-friendly API to improve runtime efficiency
and reproducibility.
