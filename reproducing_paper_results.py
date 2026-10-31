"""Run MAUIL's Table 3 ranking experiment on WD or DBLP."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np

from mauil import (MAUIL, __version__, hit_precision, split_anchors,
                   load_pretrained_vectors, load_word_vectors)
from mauil.data import load_paper_dataset
from mauil.pretrained import pretrained_metadata
from mauil.embedding import resolve_line_backend

ROOT = Path(__file__).resolve().parent


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', choices=['wd', 'dblp'], default='dblp')
    cli.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    cli.add_argument('--cache-dir', type=Path, default=ROOT / 'embeddings')
    cli.add_argument('--output', type=Path, help='Default: results/<dataset>.csv next to this script')
    cli.add_argument('--word-vector-source', choices=['pretrained', 'local'], default='pretrained')
    cli.add_argument('--word-vectors', type=Path, help='Explicit local vectors override the default resource')
    cli.add_argument('--word-vector-format', choices=['gensim', 'word2vec-text', 'word2vec-binary'], default='gensim')
    cli.add_argument('--word-vector-cache', type=Path, help='Hugging Face cache directory')
    cli.add_argument('--local-files-only', action='store_true', help='Read pretrained files from cache without networking')
    cli.add_argument('--word-corpus', type=Path, help='Historical local mode only: tokenized external corpus, one document per line')
    cli.add_argument('--embedding-repeats', type=int, default=1)
    cli.add_argument('--embedding-seed', type=int, default=0)
    cli.add_argument('--split-seeds', type=int, nargs='+', default=[0])
    cli.add_argument('--n-train', type=int, default=200)
    cli.add_argument('--n-test', type=int, default=500)
    cli.add_argument('--embedding-dim', type=int, default=100)
    cli.add_argument('--n-components', type=int)
    cli.add_argument('--reg', type=float)
    cli.add_argument('--neighbor-weight', type=float, default=.1)
    cli.add_argument('--char-epochs', type=int, default=50)
    cli.add_argument('--line-epochs', type=int, default=50)
    cli.add_argument('--batch-size', type=int, default=1024)
    cli.add_argument('--workers', type=int, default=1)
    cli.add_argument('--device', default='cpu')
    cli.add_argument('--line-backend', choices=['auto', 'torch'], default='auto',
                     help='auto: fused Adam when supported; torch: ordinary Adam')
    cli.add_argument('--torch-threads', type=int, default=4)
    cli.add_argument('--no-feature-cache', action='store_true',
                     help='Disable deterministic text/LDA caching under cache-dir/text')
    return cli


def _run(args):
    if args.output is None:
        args.output = ROOT / 'results' / (args.dataset + '.csv')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite results: %s' % args.output)
    if args.embedding_repeats < 1 or args.n_test < 5:
        raise ValueError('Require at least one embedding repeat and five test pairs.')
    if len(set(args.split_seeds)) != len(args.split_seeds):
        raise ValueError('Split seeds must be unique.')
    line_backend = resolve_line_backend(args.line_backend, args.device)
    archive = args.data_dir / (args.dataset + '.zip')
    source, target, attr_s, attr_t, anchors, stop_words = load_paper_dataset(archive, args.dataset)
    for seed in args.split_seeds:
        split_anchors(anchors, args.n_train, args.n_test, seed)
    if args.word_corpus is not None and (args.word_vector_source != 'local' or args.word_vectors is not None):
        raise ValueError('--word-corpus requires --word-vector-source local and cannot be combined with --word-vectors.')
    if args.dataset == 'wd' and args.word_vector_source == 'local' and args.word_corpus is None and args.word_vectors is None:
        raise ValueError('Historical WD local mode requires --word-corpus or --word-vectors.')
    corpus_hash = _digest(args.word_corpus) if args.word_corpus is not None else None
    shared_vectors = None
    language = 'zh' if args.dataset == 'wd' else 'en'
    if args.word_vectors is not None:
        shared_vectors = load_word_vectors(args.word_vectors, args.word_vector_format)
        word_source = dict(shared_vectors.mauil_metadata)
    elif args.word_vector_source == 'pretrained':
        word_source = pretrained_metadata(language)
    else:
        word_source = dict(mode='local', corpus_sha256=corpus_hash,
                           corpus='external' if args.word_corpus else 'dataset-word-attributes',
                           vector_size=args.embedding_dim, seed_policy='embedding_seed+1')
    word_source['lookup_policy'] = ('phrase-fallback' if language == 'en' and
                                    args.word_vector_source != 'local' else 'exact')
    dependencies = {name: importlib.metadata.version(name) for name in
                    ['torch', 'numpy', 'scipy', 'gensim', 'networkx', 'scikit-learn',
                     'nltk', 'jieba', 'zhconv', 'pypinyin', 'huggingface_hub', 'joblib',
                     'threadpoolctl']}
    config = dict(dataset=args.dataset, archive_sha256=_digest(archive),
                  corpus_sha256=corpus_hash, version=__version__, dependencies=dependencies,
                  python=platform.python_version(), platform=platform.platform(),
                  embedding_dim=args.embedding_dim, neighbor_weight=args.neighbor_weight,
                  char_epochs=args.char_epochs, line_epochs=args.line_epochs,
                  batch_size=args.batch_size, workers=args.workers, device=args.device,
                  word_vector_source=word_source, word_lookup_mode=args.word_vector_source,
                  line_backend=line_backend, torch_threads=args.torch_threads, blas_threads=1,
                  optimizer=dict(name='torch.optim.Adam', betas=[.9, .999], weight_decay=0,
                                 character=dict(lr=.01, epsilon=1e-8,
                                                backend=resolve_line_backend('auto', args.device)),
                                 line=dict(lr=.001, epsilon=1e-7, backend=line_backend)),
                  sampler=dict(name='scipy.stats.sampling.DiscreteAliasUrn', degree_power=.75,
                               negative_ratio=5, urn_factor=2, random_state='local RandomState'),
                  phrases='gensim4-freeze-double-bigram', lda_eval_every=0,
                  word_aggregation='csr', rcca='cholesky-svd-with-degenerate-eigh',
                  standardized=False, cache_format=5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    columns = ['dataset', 'kind', 'embedding_repeat', 'embedding_seed', 'split_seed',
               'n_train', 'n_test', 'candidate_count', 'embedding_dim', 'n_components',
               'reg', 'word_vector_source', 'word_embedding_dim', 'word_token_coverage',
               'word_vocabulary_coverage', 'word_missing_tokens', 'word_vector_metadata',
               'hp1', 'hp3', 'hp5', 'configuration']
    # Reserve the result before any expensive training; never overwrite a run.
    with args.output.open('x', newline='', encoding='utf-8') as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        all_scores = []
        for repeat in range(args.embedding_repeats):
            seed = args.embedding_seed + repeat
            model = MAUIL(args.dataset, n_components=args.n_components, reg=args.reg,
                          embedding_dim=args.embedding_dim, neighbor_weight=args.neighbor_weight,
                          random_state=seed, device=args.device, workers=args.workers,
                          stop_words=stop_words, char_epochs=args.char_epochs,
                          line_epochs=args.line_epochs, batch_size=args.batch_size,
                          word_vector_source=args.word_vector_source,
                          word_vector_cache=args.word_vector_cache,
                          local_files_only=args.local_files_only, line_backend=args.line_backend,
                          feature_cache=None if args.no_feature_cache else args.cache_dir / 'text' / args.dataset)
            identity = dict(config, embedding_seed=seed)
            serialized = json.dumps(identity, sort_keys=True)
            cache_key = hashlib.sha256(serialized.encode()).hexdigest()[:24]
            cache_path = args.cache_dir / (args.dataset + '-' + cache_key + '.npz')
            first_train, _ = split_anchors(anchors, args.n_train, args.n_test, args.split_seeds[0])
            if cache_path.exists():
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached['configuration']) != serialized:
                        raise ValueError('Embedding cache configuration does not match this run.')
                    blocks = tuple(tuple(cached['%s_%d' % (side, i)] for i in range(4))
                                   for side in ['source', 'target'])
                    coverage = json.loads(str(cached['word_coverage']))
                    vector_metadata = json.loads(str(cached['word_vector_metadata']))
                model.fit_embeddings(*blocks, list(source), list(target), first_train)
                model.word_coverage_, model.word_vector_metadata_ = coverage, vector_metadata
                print('Loaded embeddings:', cache_path, flush=True)
            else:
                vectors = shared_vectors
                if args.word_vectors is None and args.word_vector_source == 'pretrained' and vectors is None:
                    shared_vectors = load_pretrained_vectors(language, args.word_vector_cache,
                                                            args.local_files_only)
                    vectors = shared_vectors
                if args.word_corpus is not None:
                    from gensim.models.word2vec import LineSentence
                    from mauil.text import train_word_vectors
                    vectors = train_word_vectors(LineSentence(str(args.word_corpus)),
                                                 args.embedding_dim, seed + 1, args.workers)
                    vectors.mauil_metadata = dict(word_source, training_seed=seed + 1)
                print('Training', args.dataset, 'embedding repeat', repeat, 'seed', seed,
                      'LINE backend', line_backend, flush=True)
                model.fit(source, target, attr_s, attr_t, first_train, word_vectors=vectors)
                arrays = {'configuration': np.array(serialized), 'source_nodes': np.asarray(list(source)),
                          'target_nodes': np.asarray(list(target)),
                          'word_coverage': np.array(json.dumps(model.word_coverage_, sort_keys=True)),
                          'word_vector_metadata': np.array(json.dumps(model.word_vector_metadata_, sort_keys=True))}
                arrays.update({'%s_%d' % (side, i): block for side, blocks in
                               zip(['source', 'target'], model.embeddings_) for i, block in enumerate(blocks)})
                with cache_path.open('xb') as cache:
                    np.savez_compressed(cache, **arrays)
                del arrays
            record_config = json.dumps(dict(identity, n_components=model.n_components,
                                            reg=model.reg, word_coverage=model.word_coverage_,
                                            actual_word_vectors=model.word_vector_metadata_), sort_keys=True)
            base = dict(dataset=args.dataset, n_train=args.n_train, n_test=args.n_test,
                        candidate_count=args.n_test, embedding_dim=args.embedding_dim,
                        n_components=model.n_components, reg=model.reg, configuration=record_config,
                        word_vector_source=word_source['mode'], word_embedding_dim=model.word_embedding_dim_,
                        word_token_coverage=model.word_coverage_['token_coverage'],
                        word_vocabulary_coverage=model.word_coverage_['vocabulary_coverage'],
                        word_missing_tokens=model.word_coverage_['missing_tokens'],
                        word_vector_metadata=json.dumps(model.word_vector_metadata_, sort_keys=True))
            print('Word vectors: %s, width=%d, token coverage=%.2f%%, vocabulary coverage=%.2f%%, missing tokens=%d' %
                  (word_source.get('repo_id', word_source['mode']), model.word_embedding_dim_,
                   100 * model.word_coverage_['token_coverage'],
                   100 * model.word_coverage_['vocabulary_coverage'],
                   model.word_coverage_['missing_tokens']), flush=True)
            scores = []
            for split_seed in args.split_seeds:
                train, test = split_anchors(anchors, args.n_train, args.n_test, split_seed)
                if split_seed != args.split_seeds[0]:
                    model.refit_alignment(train)
                truth = [v for _, v in test]
                ranked, _ = model.rank([u for u, _ in test], truth, top_k=10)
                values = [hit_precision(ranked, truth, k) for k in [1, 3, 5]]
                scores.append(values)
                writer.writerow(dict(base, kind='split', embedding_repeat=repeat,
                                     embedding_seed=seed, split_seed=split_seed,
                                     **dict(zip(['hp1', 'hp3', 'hp5'], values))))
                output.flush()
                print('repeat=%d split=%d train=%d candidates=%d K=%d R=%g HP=%s' %
                      (repeat, split_seed, len(train), len(truth), model.n_components,
                       model.reg, values), flush=True)
            all_scores.extend(scores)
            writer.writerow(dict(base, kind='embedding_mean', embedding_repeat=repeat,
                                 embedding_seed=seed, **dict(zip(['hp1', 'hp3', 'hp5'],
                                                               np.mean(scores, axis=0)))))
            output.flush()
        writer.writerow(dict(base, kind='overall_mean', **dict(zip(['hp1', 'hp3', 'hp5'],
                                                                  np.mean(all_scores, axis=0)))))
    print('Saved:', args.output)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.torch_threads < 1:
        raise ValueError('--torch-threads must be positive.')
    import torch
    from threadpoolctl import threadpool_limits
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.torch_threads)
        with threadpool_limits(limits=1, user_api='blas'):
            _run(args)
    finally:
        torch.set_num_threads(previous_threads)


if __name__ == '__main__':
    main()
