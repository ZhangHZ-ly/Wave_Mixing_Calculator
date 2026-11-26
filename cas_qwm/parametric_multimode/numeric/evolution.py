"""Exact evolution in the blocks of fixed sector and conserved L, with Fock or coherent drive and seeds."""
from collections import defaultdict
from itertools import product
from math import factorial, lgamma
import numpy as np
from scipy.special import gammaln
from scipy.stats import nbinom, poisson
from .block import Grid, grid_size, expi as _expi
from . import frame as _frame
from .operators import Op

__all__ = ['Evolution', 'charlier_nodes', 'coherent']


def charlier_nodes(lam, q):
    """Gauss-Charlier nodes and weights of the Poisson distribution of mean lam (exact for
    polynomials of degree < 2q)."""
    a = np.arange(q) + lam
    b = np.sqrt(np.arange(1, q) * lam)
    x, V = np.linalg.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
    return x, V[0] ** 2


def coherent(beta, tol=1e-12):
    """{n: <n|beta>} of a coherent state, truncated where the remaining weight is below tol."""
    p, n, out = abs(beta) ** 2, 0, {}
    left = 1.0
    while left > tol or n <= p:
        a = np.exp(-p / 2) * beta ** n / np.sqrt(float(factorial(n)))
        out[n] = a
        left -= abs(a) ** 2
        n += 1
    return out


def _rule(lam, nodes, small):
    return ('poisson' if lam <= small else 6) if nodes is None else nodes


def _number_nodes(lam, nodes):
    if nodes != 'poisson':
        return charlier_nodes(lam, nodes)
    sd = np.sqrt(lam)
    Ms = np.arange(max(0, int(lam - 10 * sd - 10)), int(lam + 10 * sd + 10) + 1)
    return Ms, poisson.pmf(Ms, lam)


def _coh(x, s, lam, phase):
    """c*_{x+s} c_x / P(x) of a coherent state of mean number lam and phase `phase`."""
    return np.exp(0.5 * (lgamma(x + 1) - lgamma(x + s + 1) + s * np.log(lam)) - 1j * s * phase)


class Evolution:
    """
    Exact evolution exp(-i H t) of the drive and the weak-field chains (see Chain) under the
    TWM down-conversion or FWM Hamiltonian of the solvers. The frames are those of
    TWMSolver.get_sol and FWMSolver.sol_expr (with the self-phase modulation). Only χ t and Δk t
    enter; t may be an array of times (the results then have a leading time axis, at the cost
    of the longest time).

    initial: input state of the explicit chains (the collective ones are in vacuum), as
    {(occ_1, ..., occ_E): amplitude} over the explicit chains in order (occ = (n_1, n_2) of a
    pair, n of a degenerate mode), or as a list of (n_0, occs, amplitude) with drive photon
    numbers n_0 added to the drive state of expect (N=0 for a drive given in full); default:
    the product of the chains' Fock inputs.
    seeds: coherent seeds {(chain, field): beta} of explicit chains (field 0: signal, 1: idler),
    whose photons are added to the input's. Like the drive, a seed of a pair is averaged over its
    photon number, which only shifts the sector: by a Gauss-Charlier quadrature with seed_nodes
    nodes over real photon numbers (default for |beta|^2 > 25: 6), or exactly (seed_nodes=
    'poisson', default below; always for a degenerate mode, two seeds on one pair, or a partner
    field likely to hold more photons than the seed) by expanding the input into Fock terms.
    cutoff: pair excitations kept beyond the largest initial ones, in each chain and in their sum
    (an integer, or one per chain followed by the one of the sum); None grows them separately
    until each outermost excitation shell holds a population below tol (see edge_weight) in every
    block used. max_dim: largest grid allowed (ValueError beyond it).
    method: 'exact' (Fock basis of the pair excitations), 'frame' (frame co-moving with the
    classical motion of the chains, see frame.py: cost independent of the drive and seed
    amplitudes; polynomial observables, ODE relative tolerance rtol) or 'auto' (frame when the
    exact grid would exceed 10^4 states with a barely depleted drive, decided at the first
    evaluation, exact for non-polynomial observables).
    """

    def __init__(self, chains, nlc, t, initial=None, seeds=None, cutoff=None, process='twm', tol=1e-12,
                 max_dim=4_000_000, seed_nodes=None, method='auto', rtol=1e-10):
        self.chains, self.process, self.tol, self.max_dim = tuple(chains), process, tol, max_dim
        self.method, self.rtol, self._frames, self._auto_method = method, rtol, {}, method == 'auto'
        self.nlc = float(nlc)
        self.times = np.atleast_1d(np.asarray(t, float))
        self._scalar = np.ndim(t) == 0
        explicit = [i for i, c in enumerate(self.chains) if c.explicit]
        if initial is None:
            initial = {tuple(self.chains[i].occ for i in explicit): 1}
        if isinstance(initial, dict):
            initial = [(0, occs, amp) for occs, amp in initial.items()]
        comps = [(n0, {i: list(np.atleast_1d(o)) for i, o in zip(explicit, occs)}, amp) for n0, occs, amp in initial]
        seeds = {k: complex(b) for k, b in (seeds or {}).items() if b}
        self._quad = []
        for (c, f), beta in seeds.items():
            if not self.chains[c].explicit:
                raise ValueError('Only explicit chains can be seeded')
            rule = _rule(abs(beta) ** 2, seed_nodes, 25)
            if rule != 'poisson' and self.chains[c].kind == 'pair' and (c, 1 - f) not in seeds and poisson.cdf(
                    max(o[c][1 - f] - o[c][f] for _, o, _ in comps) - 1, abs(beta) ** 2) <= tol:
                self._quad.append((c, f, abs(beta) ** 2, np.angle(beta)) + _number_nodes(abs(beta) ** 2, rule))
                continue
            new = []
            for n0, occ, amp in comps:
                for n, a in coherent(beta, tol).items():
                    o = {i: list(x) for i, x in occ.items()}
                    o[c][f] += n
                    new.append((n0, o, amp * a))
            comps = new
        self._seed_grid = [(tuple(x for x, _ in xs), np.prod([w for _, w in xs]))
                           for xs in product(*(list(zip(q[4], q[5])) for q in self._quad))]
        r = 1 if process == 'twm' else 2
        quad = {q[0]: q for q in self._quad}
        self._comps, n_eff = [], []
        for n0, occ, amp in comps:
            secs = []
            for i, ch in enumerate(self.chains):
                if i in quad:
                    _, f, _, _, x, _ = quad[i]
                    own, other = occ[i][f], occ[i][1 - f]
                    if own - other + x.min() < 0:
                        raise ValueError('Seed nodes below the idler photons: use seed_nodes="poisson"')
                    dc = [0, 0]
                    dc[f] = own - other
                    secs.append(((1 + own - other + x.max()) / 2, other, tuple(dc)))
                else:
                    secs.append(ch.sector((tuple(occ[i]) if ch.kind == 'pair' else occ[i][0]) if ch.explicit else None))
            m = tuple(s[1] for s in secs)
            self._comps.append((tuple(s[2] for s in secs), n0 + r * sum(m), m, amp))
            n_eff.append([2 * s[0] + 2 * s[1] for s in secs] + [sum(2 * s[0] + 2 * s[1] for s in secs)])
        w = np.abs([c[3] for c in self._comps]) ** 2
        w, n_eff = w / w.sum(), np.array(n_eff, float)
        mean = w @ n_eff
        self._n_eff = np.maximum(mean + 6 * np.sqrt(w @ (n_eff - mean) ** 2), 1e-9)
        self._m0 = np.append(np.max([c[2] for c in self._comps], 0), max(sum(c[2]) for c in self._comps))
        self._e_max = max(c[1] for c in self._comps)
        self._n0_max = max(n0 for n0, _, _ in initial)
        self._index = defaultdict(list)
        for fam in {(d, e) for d, e, _, _ in self._comps}:
            self._index[self._mask(fam[0])].append(fam)
        self._auto, self._fitted = cutoff is None, -np.inf
        cuts = self._guess() if cutoff is None else np.broadcast_to(np.asarray(cutoff, int), (len(self.chains) + 1,))
        self._rebuild(cuts, self._cap(0))

    def _mask(self, d):
        d = list(d)
        for c, f, *_ in self._quad:
            d[c] = (None, d[c][1]) if f == 0 else (d[c][0], None)
        return tuple(d)

    def _place(self, d, S):
        d = list(d)
        for (c, f, *_), x in zip(self._quad, S):
            d[c] = (d[c][0] + x, d[c][1]) if f == 0 else (d[c][0], d[c][1] + x)
        return tuple(d)

    def _cap(self, N):
        return int(np.floor((max(N, 0) + self._e_max) / (1 if self.process == 'twm' else 2))) + 10

    def _rebuild(self, cuts, cap):
        lims = self._m0 + np.asarray(cuts, int)
        total = lims[-1] if self.method == 'frame' else min(lims[-1], cap)
        size = grid_size(lims[:-1], total)
        if size > self.max_dim:
            raise ValueError(f'Cutoffs {tuple(int(x) for x in cuts)} need a grid of {size} states '
                             f'(max_dim {self.max_dim}): reduce the chains, the drive or the time')
        self.cutoff, self._cap_now = tuple(int(x) for x in cuts), cap
        self.grid, self._states, self._lad = Grid(self.chains, lims[:-1], total, self.process), {}, {}
        self._init = defaultdict(lambda: np.zeros(self.grid.dim, complex))
        for d, e, m, amp in self._comps:
            self._init[d, e][self.grid.index(m)] += amp

    def _guess(self, N=0):
        if self.method == 'frame':
            return np.full(len(self.chains) + 1, 6)
        r = 1 if self.process == 'twm' else 2
        N = max(N + self._n0_max, 0)
        g = self.nlc * abs(self.times).max() * (np.sqrt(N) if r == 1 else N)
        q = np.tanh(min(g, 20.0)) ** 2
        return np.minimum(nbinom.isf(self.tol, self._n_eff, 1 - q) + 5, N / r + 10).astype(int)

    def _out(self, x):
        return x[0] if self._scalar else x

    def _ladders(self, d):
        if d not in self._lad:
            self._lad[d] = _frame.Ladders(self.grid, d)
        return self._lad[d]

    def _ensure(self, keys):
        todo = list(dict.fromkeys((fam, float(M), S) for fam, M, S in keys if (fam, float(M), S) not in self._states))
        if self.method != 'frame':
            for fam, M, S in todo:
                H = self.grid.hamiltonian(self._place(fam[0], S), M + fam[1], self.nlc)
                self._states[fam, M, S] = _expi(H, self._init[fam], self.times)
            return
        items = [(self._ladders(self._place(fam[0], S)), M + fam[1], self._init[fam]) for fam, M, S in todo]
        for key, (phi, fr) in zip(todo, _frame.evolve(self, items) if items else ()):
            self._states[key], self._frames[key] = phi, fr

    def _state(self, fam, M, S=()):
        self._ensure([(fam, M, S)])
        return self._states[fam, float(M), S]

    def _fock(self, runs, N):
        """Grid of the pair excitations and {(family, M, S): states} of the runs in it (frames expanded)."""
        keys = [(fam, float(M), S) for fam, M, S, *_ in runs]
        self._ensure(keys)
        if self.method != 'frame':
            self.fock_grid = self.grid
            return self.grid, {k: self._states[k] for k in keys}
        grid, states = _frame.fock_states(self, keys, self._cap(N))
        self.fock_grid = grid
        return grid, states

    def _edge(self, M):
        g, C = self.grid, len(self.chains)
        tops = [(c, g.states[:, c] == g.lims[c]) for c in range(C) if g.lims[c] < g.total]
        if g.total < min(sum(g.lims), np.inf if self.method == 'frame' else self._cap_now):
            tops.append((C, g.m_sum == g.total))
        out = np.zeros(C + 1)
        self._ensure([(fam, M, S) for fam in self._init for S, _ in self._seed_grid])
        for fam in list(self._init):
            for S, _ in self._seed_grid:
                p = abs(self._state(fam, M, S)) ** 2
                for c, top in tops:
                    out[c] = max(out[c], p[:, top].sum(1).max())
        return out

    def _choose(self, N):
        """'frame' if the exact grid would be large and the drive barely depleted, else 'exact'."""
        if not self._auto:
            return 'exact'
        r, n0 = (1 if self.process == 'twm' else 2), max(N + self._n0_max, 1)
        cuts = self._guess(N)
        size = grid_size(self._m0[:-1] + cuts[:-1], min(self._m0[-1] + cuts[-1], self._cap(N)))
        g = self.nlc * abs(self.times).max() * (np.sqrt(n0) if r == 1 else n0)
        pairs = self._n_eff[-1] * np.sinh(min(g, 20.0)) ** 2
        return 'frame' if size > 10_000 and pairs < 0.05 * n0 / r else 'exact'

    def _to_exact(self):
        self.method, self._fitted, self._frames = 'exact', -np.inf, {}
        self._rebuild(self._guess(), self._cap(0))

    def _fit(self, M):
        if M <= self._fitted:
            return
        if self.method == 'auto':
            self.method = self._choose(M)
            if self.method == 'frame':
                self._rebuild(self._guess(M), self._cap(M))
        cuts = np.maximum(self.cutoff, self._guess(M)) if self._auto else np.array(self.cutoff)
        cap = max(self._cap(M), self._cap_now)
        if cap > self._cap_now or tuple(cuts) != self.cutoff:
            self._rebuild(cuts, cap)
        while self._auto:
            bad = self._edge(M) > self.tol
            if not bad.any():
                break
            self._rebuild(np.where(bad, np.array(self.cutoff) * 3 // 2 + 8, self.cutoff), cap)
        self._fitted = M

    def _nodes(self, N, alpha, nodes):
        if alpha is None:
            self._fit(N)
            return [N], [1.0], None
        Ms, ws = _number_nodes(abs(alpha) ** 2, _rule(abs(alpha) ** 2, nodes, 100))
        self._fit(max(Ms) + 2 * self.grid.r)
        return Ms, ws, alpha

    def _runs(self, Ms, ws):
        for fam in list(self._init):
            for M, w in zip(Ms, ws):
                for S, v in self._seed_grid:
                    yield fam, M, S, w * v

    def edge_weight(self, N=0):
        """Largest population (over chains, blocks and times) of the outermost excitation shell of
        a chain with the drive in |N>, which bounds the truncation error of the cutoffs."""
        self._fit(N)
        return self._edge(N).max()

    def blocks(self, N=0):
        """{(sector, L): state vector(s)} with the drive in |N> (N=0 for an input given in full),
        at the nodes of the seeds averaged by quadrature."""
        self._fit(N)
        runs = [(fam, N, S) for fam in self._init for S, _ in self._seed_grid]
        _, states = self._fock(runs, N)
        return {(self._place(fam[0], S), N + fam[1]): self._out(states[fam, float(N), S]) for fam, _, S in runs}

    def _bras(self, M, S, d2, L2, alpha):
        """Blocks (family, M, S) holding the image (d2, L2) of block (M, S), with their coherent weights."""
        lam = abs(alpha or 0) ** 2
        for fam2 in self._index.get(self._mask(d2), ()):
            s = L2 - fam2[1] - M
            if (s if alpha is None else M + s < 0):
                continue
            f, S2 = 1 if alpha is None else _coh(M, s, lam, np.angle(alpha)), []
            for (c, fl, lam_s, phase, *_), x in zip(self._quad, S):
                sig = round(d2[c][fl] - fam2[0][c][fl] - x)
                f, S2 = f * _coh(x, sig, lam_s, phase), S2 + [x + sig]
            yield fam2, M + s, tuple(S2), f

    def expect(self, op: Op, N=0, alpha=None, nodes=None):
        """
        <op>(t) with the drive in the Fock state |N> or the coherent state |alpha> (added to the
        drive numbers of the input). With alpha, the average over the drive photon number is an
        exact Poisson sum over integer blocks (nodes='poisson') or a Gauss-Charlier quadrature
        with `nodes` nodes over real ones, which needs block results smooth in N (large
        |alpha|^2); default: the former for |alpha|^2 <= 100, else 6 nodes.
        """
        if self._auto_method and self.method != 'exact' and any(n == 'fn' for _, f in op.terms for n, _ in f):
            self._to_exact()
        Ms, ws, alpha = self._nodes(N, alpha, nodes)
        runs, frame, res = list(self._runs(Ms, ws)), self.method == 'frame', 0
        if frame:
            self._ensure([(fam, M, S) for fam, M, S, _ in runs])
            zero = np.zeros(self.grid.dim, complex)
            self._ensure([b[:3] for fam, M, S, _ in runs
                          for d2, L2 in op.apply(self.grid, self._place(fam[0], S), M + fam[1], zero)
                          for b in self._bras(M, S, d2, L2, alpha)])
        for fam, M, S, w in runs:
            v, d = self._state(fam, M, S), self._place(fam[0], S)
            src = self._frames[fam, float(M), S] if frame else None
            grid = _frame.Rotated(self, src[0], src[3], src[4], self.grid.k(d).sum()) if frame else self.grid
            for (d2, L2), img in op.apply(grid, d, M + fam[1], v).items():
                for fam2, M2, S2, f in self._bras(M, S, d2, L2, alpha):
                    bra = self._state(fam2, M2, S2)
                    ket = _frame.relative(self._ladders(d2), self._frames[fam2, float(M2), S2], src, img) if frame else img
                    res = res + w * f * (bra.conj() * ket).sum(-1)
        return self._out(res if np.ndim(res) else np.zeros(len(self.times), complex))

    def distribution(self, chain=None, N=0, alpha=None, nodes=None, single=False):
        """
        Distribution of the signal photon number N_1 of `chain` (of the pairs Σ m over all chains
        if None), for a Fock or coherent drive (the latter averaged as in expect). single=True
        gives that of one mode of a collective chain (exact marginal of its n modes).
        """
        if any(c == chain and f == 0 for c, f, *_ in self._quad):
            raise ValueError('The signal of a seed averaged by quadrature has no distribution (seed_nodes="poisson")')
        Ms, ws, _ = self._nodes(N, alpha, nodes)
        runs = list(self._runs(Ms, ws))
        g, states = self._fock(runs, max(Ms))
        parts = [(w, (g.m_sum if chain is None else g.numbers(self._place(fam[0], S), chain)).astype(int),
                  abs(states[fam, float(M), S]) ** 2) for fam, M, S, w in runs]
        size = max(q.max() for _, q, _ in parts) + 1
        res = sum(w * np.stack([np.bincount(q, x, minlength=size) for x in p]) for w, q, p in parts)
        if single:
            res = res @ _single_marginal(self.chains[chain].n, size)
        return self._out(res)


def _single_marginal(n, size):
    """P(l | m): pairs l of one of the n modes of a collective chain holding m pairs."""
    if n == 1:
        return np.eye(size)
    m, l = np.arange(size)[:, None], np.arange(size)[None, :]
    j = np.maximum(m - l, 0)
    with np.errstate(invalid='ignore'):
        logp = (gammaln(m + 1) - gammaln(j + 1) + gammaln(n - 1 + j) - gammaln(n - 1)
                - gammaln(n + m) + gammaln(n))
    return np.where(l <= m, np.exp(logp), 0)
