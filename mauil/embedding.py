"""PyTorch character autoencoder and the original first+second order LINE."""

import math
import warnings
import inspect

import numpy as np
import torch
from torch import nn
from scipy.stats.sampling import DiscreteAliasUrn


def _adam(parameters, lr, epsilon, backend='auto', device='cpu'):
    """Standard framework Adam; epsilon differs from the historical TF update."""
    options = {'fused': True} if resolve_line_backend(backend, device) == 'torch-fused' else {}
    return torch.optim.Adam(parameters, lr=lr, betas=(.9, .999), eps=epsilon,
                            weight_decay=0, amsgrad=False, **options)


class _Autoencoder(nn.Module):
    def __init__(self, input_dim, dim, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.encoder = nn.Linear(input_dim, dim).double()
            self.decoder = nn.Linear(dim, input_dim).double()
        rng = np.random.RandomState(seed)
        with torch.no_grad():
            for layer in (self.encoder, self.decoder):
                bound = math.sqrt(6 / (layer.in_features + layer.out_features))
                layer.weight.copy_(torch.from_numpy(rng.uniform(
                    -bound, bound, (layer.out_features, layer.in_features))))
                layer.bias.zero_()

    def forward(self, data):
        hidden = self.encoder(data)
        return hidden, self.decoder(hidden)


def character_embedding(data, dim, seed=0, device='cpu', epochs=50):
    if data.shape[1] == 0:
        return np.zeros((len(data), dim), dtype=np.float64)
    model = _Autoencoder(data.shape[1], dim, seed).to(device)
    optimizer = _adam(model.parameters(), lr=.01, epsilon=1e-8, device=device)
    x = torch.as_tensor(data, dtype=torch.float64, device=device)
    for _ in range(epochs):
        optimizer.zero_grad()
        hidden, output = model(x)
        loss = (output - x).square().mean()
        # TF sess.run([train_op, hidden, loss]) returns the pre-update hidden.
        if _ == epochs - 1:
            embedding = hidden.detach().cpu().numpy().copy()
        loss.backward()
        optimizer.step()
    return embedding


def _line_batches(graph, nodes, batch_size, negative_ratio, seed):
    index = {node: i for i, node in enumerate(nodes)}
    edges = np.asarray([(index[u], index[v]) for u, v in graph.edges()], dtype=np.int64)
    degree = np.bincount(edges[:, 0], minlength=len(nodes)).astype(float) ** .75
    rng = np.random.RandomState(seed)
    # Extra urn capacity also avoids SciPy 1.10's initialization failure for
    # some exactly uniform distributions (e.g. regular/cycle graphs).
    sampler = DiscreteAliasUrn(degree, urn_factor=2, random_state=rng)
    while True:
        order = rng.permutation(len(edges))
        for start in range(0, len(edges), batch_size):
            h, t = edges[order[start:start + batch_size]].T
            yield h, t, 1
            for _ in range(negative_ratio):
                yield h, sampler.rvs(size=len(h)), -1


class _LINE(nn.Module):
    def __init__(self, n_nodes, dim, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.first = nn.Embedding(n_nodes, dim // 2).float()
            self.second = nn.Embedding(n_nodes, dim // 2).float()
            self.context = nn.Embedding(n_nodes, dim // 2).float()
        rng = np.random.RandomState(seed)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.from_numpy(rng.uniform(-.05, .05,
                                                             tuple(parameter.shape)).astype('float32')))

    def forward(self, heads, tails, sign):
        # One lookup/backward avoids a second full first-order gradient matrix.
        first_pair = self.first(torch.stack((heads, tails)))
        first = (first_pair[0] * first_pair[1]).sum(-1)
        second = (self.second(heads) * self.context(tails)).sum(-1)
        return sum(-torch.log(torch.sigmoid(sign * score) + 1e-8).mean()
                   for score in (first, second))


def resolve_line_backend(backend, device='cpu'):
    if backend not in ('auto', 'torch'):
        raise ValueError("line_backend must be 'auto' or 'torch'.")
    # CPU fused Adam requires Torch 2.11; other versions/devices use ordinary Adam.
    release = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
    fused = (torch.device(device).type == 'cpu' and release >= (2, 11)
             and 'fused' in inspect.signature(torch.optim.Adam).parameters)
    return 'torch-fused' if backend == 'auto' and fused else 'torch'


def network_embedding(graph, dim, seed=0, device='cpu', epochs=50,
                      batch_size=1024, negative_ratio=5, backend='auto'):
    """Unweighted LINE with the legacy continuous generator/update budget."""
    graph = graph if graph.is_directed() else graph.to_directed()
    nodes = list(graph)
    model = _LINE(len(nodes), dim, seed).to(device)
    if graph.number_of_edges():
        optimizer = _adam(model.parameters(), lr=.001, epsilon=1e-7,
                          backend=backend, device=device)
        batches = _line_batches(graph, nodes, batch_size, negative_ratio, seed)
        steps = (graph.number_of_edges() * (1 + negative_ratio) - 1) // batch_size + 1
        for _ in range(epochs * steps):
            h, t, sign = next(batches)
            optimizer.zero_grad()
            loss = model(torch.as_tensor(h, device=device),
                         torch.as_tensor(t, device=device), sign)
            loss.backward()
            optimizer.step()
    else:
        warnings.warn("No graph edges; LINE returns its initialized vectors.", RuntimeWarning)
    return np.concatenate((model.first.weight.detach().cpu().numpy(),
                           model.second.weight.detach().cpu().numpy()), axis=1)
