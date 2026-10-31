"""The original, unnormalized RCCA generalized eigenvalue problem."""

import numpy as np
from scipy.linalg import cholesky, eigh, solve_triangular, svd


def standardize_blocks(source, target):
    """Standardize each block over BOTH networks, with population std (ddof=0)."""
    left, right, statistics = [], [], []
    for x, y in zip(source, target):
        # Preserve the incoming dtype: legacy LINE standardization used float32.
        data = np.concatenate((x, y), axis=0)
        mean, scale = data.mean(axis=0), data.std(axis=0)
        constant = np.all(data == data[0], axis=0)
        scale = np.where(constant | (scale == 0), 1, scale)
        normalized = (data - mean) / scale
        normalized[:, constant] = 0
        left.append(normalized[:len(x)])
        right.append(normalized[len(x):])
        statistics.append((mean, scale))
    return np.concatenate(left, axis=1), np.concatenate(right, axis=1), statistics


def rcca(x, y, n_components, reg):
    """Return paired projections; no extra centering or division by sample count."""
    dx, dy = x.shape[1], y.shape[1]
    xx = x.T @ x + reg * np.eye(dx)
    yy = y.T @ y + reg * np.eye(dy)
    cross = x.T @ y
    lx = cholesky(xx, lower=True)
    ly = cholesky(yy, lower=True)
    whitened = solve_triangular(lx, cross, lower=True)
    whitened = solve_triangular(ly, whitened.T, lower=True).T
    u, values, vh = svd(whitened, full_matrices=False)
    tolerance = np.finfo(values.dtype).eps * max(dx, dy) * values[0]
    # The null-space eigenvectors of the doubled problem need not split their
    # norm equally across networks. Preserve its historical treatment there.
    tied_cutoff = (n_components < len(values) and
                   abs(values[n_components - 1] - values[n_components]) <= tolerance)
    if values[n_components - 1] <= tolerance or tied_cutoff:
        return _generalized_rcca(x, y, n_components, reg)
    wx = solve_triangular(lx.T, u[:, :n_components], lower=False) / np.sqrt(2)
    wy = solve_triangular(ly.T, vh.T[:, :n_components], lower=False) / np.sqrt(2)
    return wx, wy


def _generalized_rcca(x, y, n_components, reg):
    """Original generalized solve for degenerate/null-space projections."""
    dx, dy = x.shape[1], y.shape[1]
    cross = x.T @ y
    lhs = np.zeros((dx + dy, dx + dy), dtype=np.float64)
    rhs = np.zeros_like(lhs)
    lhs[:dx, dx:], lhs[dx:, :dx] = cross, cross.T
    rhs[:dx, :dx] = x.T @ x + reg * np.eye(dx)
    rhs[dx:, dx:] = y.T @ y + reg * np.eye(dy)
    lhs, rhs = (lhs + lhs.T) / 2, (rhs + rhs.T) / 2
    # subset_by_index replaces SciPy's removed eigvals argument. Keep the same
    # generalized subset driver and B-normalized eigenvectors as the old code.
    values, vectors = eigh(lhs, rhs, subset_by_index=(dx + dy - n_components,
                                                   dx + dy - 1), driver="gvx")
    vectors = vectors[:, np.argsort(values)[::-1]]
    return vectors[:dx], vectors[dx:]
