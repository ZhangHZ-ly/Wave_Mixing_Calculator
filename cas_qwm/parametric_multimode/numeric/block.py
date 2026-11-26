"""Grid of chain excitations, sector parameters, Hamiltonian and operator factors in a block."""
import numpy as np
from scipy.sparse import coo_matrix, diags, identity
from scipy.special import jv

__all__ = ['Grid', 'grid_size', 'expi']

_PROCESSES = {'twm': 1, 'fwm': 2}


def expi(H, v, ts, tol=1e-15, chunk=64):
    """exp(-i t H) v for the times ts (leading axis), H Hermitian sparse, v one or more vectors
    (Chebyshev expansion)."""
    d = H.diagonal().real
    r = np.asarray(abs(H).sum(1)).ravel() - abs(d)
    lo, hi = (d - r).min(), (d + r).max()
    c, h = (hi + lo) / 2, max((hi - lo) / 2, 1e-300)
    zs = ts * h
    zmax = abs(zs).max()
    ks = np.arange(int(zmax + 12 * zmax ** (1 / 3) + 30))
    J = jv(ks, zs[:, None])
    K = int(np.nonzero((abs(J) > tol).any(0))[0].max()) + 1
    coef = 2 * (-1j) ** ks[:K] * J[:, :K]
    coef[:, 0] /= 2
    A = ((H - c * identity(H.shape[0], format='csr')) * (1 / h)).astype(complex).tocsr()
    res = np.zeros((len(ts),) + v.shape, complex)
    buf = np.empty((min(chunk, K),) + v.shape, complex)
    prev, cur = None, v.astype(complex)
    for k in range(K):
        if k:
            nxt = A @ cur
            if k > 1:
                nxt *= 2
                nxt -= prev
            prev, cur = cur, nxt
        buf[k % chunk] = cur
        if k % chunk == chunk - 1 or k == K - 1:
            k0 = k - k % chunk
            res += np.tensordot(coef[:, k0:k + 1], buf[:k - k0 + 1], 1)
    return np.exp(-1j * ts * c).reshape((-1,) + (1,) * v.ndim) * res


def grid_size(lims, total):
    """Number of states of Grid(chains, lims, total)."""
    n = np.zeros(int(total) + 1)
    n[0] = 1
    for lim in lims:
        cs = np.cumsum(n)
        n = cs - np.concatenate([np.zeros(int(lim) + 1), cs])[:len(n)]
    return int(round(n.sum()))


def _grid(lims, total):
    states = np.zeros((1, 0), dtype=np.int64)
    for lim in lims:
        reps = np.minimum(lim, total - states.sum(1)) + 1
        idx = np.repeat(np.arange(len(states)), reps)
        new = np.arange(len(idx)) - np.repeat(np.cumsum(reps) - reps, reps)
        states = np.column_stack([states[idx], new])
    return states


class Grid:
    """
    Pair excitations m = (m_1, ..., m_C) of the chains with m_c <= lims[c] and Σ m <= total,
    shared by all sectors. A sector d = ((d_1, d_2) per chain) gives the photons beyond the
    pairs, which fix the Bargmann indices; a block is a sector with a value of L = n_0 + r Σ m,
    conserved by H (n_0 the drive photons, r = 1 for TWM down-conversion, 2 for FWM; L may be
    real).
    """

    def __init__(self, chains, lims, total=None, process='twm'):
        self.chains, self.r = tuple(chains), _PROCESSES[process]
        self.lims = tuple(int(x) for x in lims)
        self.total = sum(self.lims) if total is None else min(int(total), sum(self.lims))
        self.states = _grid(self.lims, self.total)
        self.dim, self.m_sum = len(self.states), self.states.sum(1)
        self._strides = (max(self.lims) + 1) ** np.arange(len(self.chains) - 1, -1, -1)
        self._codes = self.states @ self._strides
        self.up = []
        for c, s in enumerate(self._strides):
            src = np.nonzero((self.m_sum < self.total) & (self.states[:, c] < self.lims[c]))[0]
            self.up.append((src, np.searchsorted(self._codes, self._codes[src] + s)))

    def index(self, m):
        return int(np.searchsorted(self._codes, np.asarray(m) @ self._strides))

    def k(self, d):
        """Bargmann indices of sector d."""
        return np.array([c.n / 2 if c.kind == 'collective' else
                         (1 + dc[0] + dc[1]) / 2 if c.kind == 'pair' else 0.25 + 0.5 * dc[0]
                         for c, dc in zip(self.chains, d)])

    def drive(self, L):
        """Drive photon numbers n_0 of the grid states in a block L."""
        return L - self.r * self.m_sum

    def numbers(self, d, c, field=0):
        """Photon numbers N_1 (field 0) or N_2 (field 1) of chain c (N of a degenerate mode)."""
        m = self.states[:, c]
        return (2 * m if self.chains[c].kind == 'degenerate' else m) + d[c][field]

    def hamiltonian(self, d, L, nlc):
        """
        Sparse H in block (d, L), in the frames of the solvers' `hamiltonian`:
        TWM: χ (a_0† A + a_0 A†) + Σ_c Δk_c K_0c;
        FWM: χ (n_0 (n_0 + 1) / 2 + 4 n_0 Σ_c K_0c + a_0†² A + a_0² A†) - Σ_c Δk_c K_0c.
        """
        k = self.k(d)
        k0 = self.states + k
        n0 = self.drive(L)
        diag = k0 @ np.array([c.mismatch for c in self.chains], float)
        if self.r == 2:
            diag = nlc * (n0 * (n0 + 1) / 2 + 4 * n0 * k0.sum(1)) - diag
        n0 = np.maximum(n0, 0)
        lower = np.sqrt(n0 if self.r == 1 else n0 * np.maximum(n0 - 1, 0))
        rows, cols, vals = (np.concatenate(x) for x in zip(*(
            (dst, src, nlc * lower[src] * np.sqrt((self.states[src, c] + 1) * (self.states[src, c] + 2 * k[c])))
            for c, (src, dst) in enumerate(self.up))))
        off = coo_matrix((vals, (rows, cols)), shape=(self.dim, self.dim))
        return (off + off.T + diags(diag)).tocsr()

    def _shift(self, c, v, up, amp):
        src, dst = self.up[c]
        out = np.zeros_like(v)
        if up:
            out[..., dst] = amp[src] * v[..., src]
        else:
            out[..., src] = amp[dst] * v[..., dst]
        return out

    def act(self, name, c, d, L, v):
        """A factor of an Op (of chain c) on the vector(s) v (last axis) of block (d, L):
        (sector, L, image)."""
        r = self.r
        if name == 'fn':
            return d, L, c(self, d, L) * v
        if name == 'a0':
            return d, L - 1, np.sqrt(np.maximum(self.drive(L), 0)) * v
        if name == 'a0d':
            return d, L + 1, np.sqrt(np.maximum(self.drive(L) + 1, 0)) * v
        if name == 'N0':
            return d, L, self.drive(L) * v
        m = self.states[:, c]
        if name in ('K+', 'K-'):
            k = self.k(d)[c]
            if name == 'K+':
                return d, L + r, self._shift(c, v, True, np.sqrt((m + 1) * (m + 2 * k)))
            return d, L - r, self._shift(c, v, False, np.sqrt(m * (m - 1 + 2 * k)))
        if name == 'K0':
            return d, L, (m + self.k(d)[c]) * v
        if name in ('N1', 'N2'):
            return d, L, self.numbers(d, c, name == 'N2') * v
        d1, d2 = d[c]
        if self.chains[c].kind == 'degenerate':
            if name.endswith('d') != bool(d1):
                return _set(d, c, (1 - d1,) * 2), L, np.sqrt(2 * m + 1) * v
            if not name.endswith('d'):
                return _set(d, c, (1, 1)), L - r, self._shift(c, v, False, np.sqrt(2 * m))
            return _set(d, c, (0, 0)), L + r, self._shift(c, v, True, np.sqrt(2 * m + 2))
        f = name[1] == '2'
        own, other = (d2, d1) if f else (d1, d2)
        if name.endswith('d'):
            if other:
                return _set(d, c, _pair(f, own, other - 1)), L + r, self._shift(c, v, True, np.sqrt(m + 1.0))
            return _set(d, c, _pair(f, own + 1, other)), L, np.sqrt(m + own + 1.0) * v
        if own:
            return _set(d, c, _pair(f, own - 1, other)), L, np.sqrt(m + own + 0.0) * v
        return _set(d, c, _pair(f, own, other + 1)), L - r, self._shift(c, v, False, np.sqrt(m + 0.0))


def _pair(f, own, other):
    return (other, own) if f else (own, other)


def _set(d, c, dc):
    return d[:c] + (dc,) + d[c + 1:]
