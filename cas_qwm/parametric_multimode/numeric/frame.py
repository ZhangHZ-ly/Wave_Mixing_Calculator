"""
Co-moving frame of a block: U(t) = e^{-iθ} Π_c D(G_c), with G_c in SU(1,1) following the classical
(mean-field) motion of chain c, so that the state left in the frame only holds the quantum
fluctuations around it.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigh_tridiagonal
from scipy.special import binom, gammaln
from scipy.stats import nbinom
from .block import Grid, grid_size

_EPS = 1e-17


class Ladders:
    """Block generators K+, K-, K0 (L kept) of the chains of sector d on vectors (last axis)."""

    def __init__(self, grid, d):
        self.grid, self.k = grid, grid.k(d)
        m = grid.states
        self.k0 = [m[:, c] + self.k[c] for c in range(len(self.k))]
        self.up = [np.sqrt((m[:, c] + 1) * (m[:, c] + 2 * self.k[c])) for c in range(len(self.k))]
        self.dn = [np.sqrt(m[:, c] * (m[:, c] - 1 + 2 * self.k[c])) for c in range(len(self.k))]
        rows, cols, vals, col = [], [], [], []
        for c, (src, dst) in enumerate(grid.up):
            for r_, c_, v_, j in ((dst, src, self.up[c][src], 3 * c), (src, dst, self.up[c][src], 3 * c + 1),
                                  (np.arange(grid.dim), np.arange(grid.dim), self.k0[c], 3 * c + 2)):
                rows.append(r_), cols.append(c_), vals.append(v_), col.append(np.full(len(r_), j))
        rows, cols, vals, col = (np.concatenate(x) for x in (rows, cols, vals, col))
        keys = rows * grid.dim + cols
        uniq, pos = np.unique(keys, return_inverse=True)
        self._flat = uniq
        self._map = np.zeros((len(uniq), 3 * len(self.k)))
        np.add.at(self._map, (pos, col), vals)

    def lin(self, c, cp, cm, c0, v):
        """(cp K+ + cm K- + c0 K0) v of chain c."""
        return (cp * self.grid._shift(c, v, True, self.up[c]) + cm * self.grid._shift(c, v, False, self.dn[c])
                + c0 * self.k0[c] * v)

    def rot(self, name, c, u, v, x):
        """Generator `name` of chain c in the frame of the Bogoliubov coefficients (u, v)."""
        if name == 'K-':
            return self.lin(c, v * v, u * u, 2 * u * v, x)
        if name == 'K+':
            return self.lin(c, np.conj(u * u), np.conj(v * v), 2 * np.conj(u * v), x)
        return self.lin(c, np.conj(u) * v, u * np.conj(v), abs(u) ** 2 + abs(v) ** 2, x)

    def bound(self, u, v):
        """Upper bound of the norm of Σ_c (K̃0_c - k_c) beyond its classical value, on the grid."""
        lims = np.array(self.grid.lims)
        return float(np.sum((abs(u) ** 2 + abs(v) ** 2) * lims
                            + 2 * abs(u * v) * np.sqrt((lims + 1) * (lims + 2 * self.k))))


def _sqrt_series(a, b, n):
    """Taylor coefficients of sqrt(a + b x) at x = 0, up to x^n."""
    j = np.arange(n + 1)
    return np.sqrt(a + 0j).real * binom(0.5, j) * (b / a) ** j


def _series(process, L, M, n):
    """Taylor coefficients (blocks, n + 1) in X of the drive factor R(M + X) of the off-diagonal elements."""
    j = np.arange(n + 1)
    sq = lambda a, b: np.sqrt(a)[:, None] * binom(0.5, j) * (b / a)[:, None] ** j
    A = L - M if process == 'twm' else L - 2 * M - 1
    if (A <= 0).any():
        raise ValueError('Drive depleted: the co-moving frame does not apply')
    if process == 'twm':
        return sq(A, -1.0)
    s1, s2 = sq(A + 1, -2.0), sq(A, -2.0)
    return np.stack([(s1[:, :i + 1] * s2[:, i::-1]).sum(1) for i in range(n + 1)], 1)


def _order(ratio):
    if ratio >= 0.5:
        raise ValueError('Depletion fluctuations too large for the co-moving frame (use method="exact")')
    return 1 if ratio == 0 else int(np.ceil(np.log(_EPS) / np.log(ratio))) + 1


def _horner(coefs, X, x):
    """Σ_n coefs[n] X^n x, coefs of shape (n,) or (n, T, 1)."""
    y = coefs[-1] * x
    for c in coefs[-2::-1]:
        y = X(y) + c * x
    return y


def classical(G, k0, km):
    """Classical K0, K- of the chains from their group elements and initial expectations."""
    u, v = G[..., 0, 0], G[..., 0, 1]
    K0 = (abs(u) ** 2 + abs(v) ** 2) * k0 + 2 * (np.conj(u) * v * np.conj(km)).real
    Km = u * u * km + v * v * np.conj(km) + 2 * u * v * k0
    return K0, Km


def _batch(lads, coefs, shift=None):
    """Dense Σ_c generators of the blocks (blocks, dim, dim) from their coefficients (blocks, 3C)."""
    nb, n = len(coefs), lads[0].grid.dim
    A = np.zeros((nb, n * n), complex)
    A[:, lads[0]._flat] = np.einsum('bnc,bc->bn', np.stack([lad._map for lad in lads]), coefs)
    A = A.reshape(nb, n, n)
    if shift is not None:
        A[:, np.arange(n), np.arange(n)] += shift[:, None]
    return A


def evolve(ev, items, chunk=4_000_000):
    """
    Frames and states of blocks [(Ladders of the sector, L, initial vector)], integrated together:
    [(phi (T, dim), (G (T, C, 2, 2), eta (T, C), theta (T,), k0, km))].
    """
    dim = ev.grid.dim
    step = max(1, chunk // dim ** 2)
    if len(items) > step:
        return [x for i in range(0, len(items), step) for x in evolve(ev, items[i:i + step], chunk)]
    C, nlc, process, nb = len(ev.chains), ev.nlc, ev.process, len(items)
    dk = np.array([c.mismatch for c in ev.chains], float)
    lads = [x[0] for x in items]
    L = np.array([x[1] for x in items], float)
    phi0 = np.stack([x[2] for x in items]).astype(complex)
    kk = np.stack([lad.k for lad in lads])
    ksum, lims = kk.sum(1), np.array(ev.grid.lims)
    norm = (abs(phi0) ** 2).sum(1)
    k0 = np.array([[np.vdot(p, lad.k0[c] * p).real for c in range(C)] for lad, p in zip(lads, phi0)]) / norm[:, None]
    km = np.array([[np.vdot(p, lad.lin(c, 0, 1, 0, p)) for c in range(C)] for lad, p in zip(lads, phi0)]) / norm[:, None]
    o1, o2 = 4 * C * nb, 5 * C * nb
    mv = lambda A, x: (A @ x[..., None])[..., 0]

    def rhs(t, y):
        G = y[:o1].reshape(nb, C, 2, 2)
        phi = y[o2 + nb:].reshape(nb, dim)
        u, v = G[..., 0, 0], G[..., 0, 1]
        K0c, Kmc = classical(G, k0, km)
        P, S0 = 2 * Kmc.real.sum(1), K0c.sum(1)
        M = S0 - ksum
        bound = ((abs(u) ** 2 + abs(v) ** 2) * lims + 2 * abs(u * v) * np.sqrt((lims + 1) * (lims + 2 * kk))).sum(1)
        ratio = bound / (L - M) if process == 'twm' else 2 * bound / (L - 2 * M - 1)
        n = _order(ratio.max())
        cs = _series(process, L, M, n)
        w, Rp = nlc * cs[:, 0], cs[:, 1]
        if process == 'twm':
            dF, Fcl = np.broadcast_to(dk, (nb, C)), K0c @ dk
        else:
            n0 = L - 2 * M
            dF = (nlc * (2 * n0 - 1 - 8 * S0))[:, None] - dk
            Fcl = nlc * (n0 * (n0 + 1) / 2 + 4 * n0 * S0) - K0c @ dk
        delta = dF + (nlc * Rp * P)[:, None]
        c0 = Fcl - (dF * K0c).sum(1) - nlc * Rp * P * S0
        h = np.zeros((nb, C, 2, 2), complex)
        h[..., 0, 0], h[..., 1, 1], h[..., 0, 1], h[..., 1, 0] = delta / 2, -delta / 2, w[:, None], -w[:, None]
        dG = -1j * h @ G
        deta = -1j * delta - 2j * w[:, None] * G[..., 0, 1] / G[..., 1, 1]
        X = _batch(lads, np.stack((np.conj(u) * v, u * np.conj(v), abs(u) ** 2 + abs(v) ** 2), -1).reshape(nb, -1), -S0)
        Ap = _batch(lads, np.stack((np.conj(u * u), np.conj(v * v), 2 * np.conj(u * v)), -1).reshape(nb, -1))

        def D(x):
            y = cs[:, -1, None] * x
            for i in range(n - 1, 0, -1):
                y = mv(X, y) + cs[:, i, None] * x
            return mv(X, y)

        Xphi = mv(X, phi)
        V = nlc * (mv(Ap, D(phi)) + D(mv(Ap.conj().swapaxes(1, 2), phi))) - (nlc * Rp * P)[:, None] * Xphi
        if process == 'fwm':
            V = V - 6 * nlc * mv(X, Xphi)
        return np.concatenate([dG.ravel(), deta.ravel(), c0, -1j * V.ravel()])

    y0 = np.concatenate([np.tile(np.eye(2, dtype=complex), (nb, C, 1, 1)).ravel(), np.zeros(C * nb + nb, complex),
                         phi0.ravel()])
    atol = np.concatenate([np.full(o2 + nb, ev.rtol * 1e-3), np.repeat(ev.rtol * 1e-3 * np.sqrt(norm), dim)])
    ts = np.unique(np.concatenate([[0.0], ev.times]))
    sol = solve_ivp(rhs, (0, ts[-1]), y0, method='DOP853', t_eval=ts, rtol=ev.rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y.T[np.searchsorted(ts, ev.times)]
    T = len(Y)
    G, eta = Y[:, :o1].reshape(T, nb, C, 2, 2), Y[:, o1:o2].reshape(T, nb, C)
    theta, phi = Y[:, o2:o2 + nb], Y[:, o2 + nb:].reshape(T, nb, dim)
    return [(phi[:, b], (G[:, b], eta[:, b], theta[:, b].copy(), k0[b], km[b])) for b in range(nb)]


def chain_matrix(k, G, eta, J, tol):
    """
    <k, m| D |k, j> (m along axis 0, j <= J) of the frame D = e^{zK+} e^{ηK0} e^{yK-} of a chain:
    the columns are the eigenvectors of the tridiagonal D K0 D^-1 (eigenvalues k + j), with the
    phase of column 0 from the coherent state D|0> and the others from D K+ D^-1.
    """
    z = G[0, 1] / G[1, 1]
    q = min(abs(z) ** 2, 1 - 1e-15)
    M = (int(nbinom.isf(tol, 2 * k, 1 - q)) if q > 0 else 0) + 2 * J + 10
    Gi = np.array([[G[1, 1], -G[0, 1]], [-G[1, 0], G[0, 0]]]) / (G[0, 0] * G[1, 1] - G[0, 1] * G[1, 0])
    u, v = Gi[0, 0], Gi[0, 1]
    A, B = abs(u) ** 2 + abs(v) ** 2, np.conj(u) * v
    while True:
        m = np.arange(M + 1)
        up = np.sqrt((m[:-1] + 1) * (m[:-1] + 2 * k))
        _, V = eigh_tridiagonal(A * (m + k), abs(B) * up, select='i', select_range=(0, J))
        if (V[-J - 5:] ** 2).sum(0).max() <= tol:
            break
        M = int(M * 1.5) + 10
    D = V * np.exp(1j * np.angle(B) * m)[:, None]
    p = int(np.argmax(abs(D[:, 0])))
    ref = eta * k + (p * np.log(z) if p else 0)
    D[:, 0] *= np.exp(1j * (ref.imag - np.angle(D[p, 0])))
    cp, cm, c0 = np.conj(u * u), np.conj(v * v), 2 * np.conj(u * v)
    for j in range(J):
        x = D[:, j]
        y = c0 * (m + k) * x
        y[1:] += cp * up * x[:-1]
        y[:-1] += cm * up * x[1:]
        ov = np.vdot(D[:, j + 1], y)
        D[:, j + 1] *= ov / abs(ov)
    return D


def fock_states(ev, keys, cap):
    """Grid of the pair excitations and the states {key: (T, dim)} of framed blocks in it."""
    C, g, T = len(ev.chains), ev.grid, len(ev.times)
    boxes = {}
    for key in keys:
        phi, (G, eta, theta, *_) = ev._states[key], ev._frames[key]
        k = g.k(ev._place(key[0][0], key[2]))
        out = []
        for t in range(T):
            psi = np.zeros(tuple(l + 1 for l in g.lims), complex)
            psi[tuple(g.states.T)] = phi[t]
            for c in range(C):
                psi = np.moveaxis(np.tensordot(chain_matrix(k[c], G[t, c], eta[t, c], g.lims[c], ev.tol), psi,
                                               axes=([1], [c])), 0, c)
            out.append(np.exp(-1j * theta[t]) * psi)
        boxes[key] = out
    lims = [max(b.shape[c] for bs in boxes.values() for b in bs) - 1 for c in range(C)]
    if grid_size(lims, cap) > ev.max_dim:
        raise ValueError(f'The Fock states need a grid of {grid_size(lims, cap)} states (max_dim {ev.max_dim})')
    grid = Grid(ev.chains, lims, cap, ev.process)
    res = {}
    for key, bs in boxes.items():
        vec = np.zeros((T, grid.dim), complex)
        for t, b in enumerate(bs):
            inside = np.all(grid.states < np.array(b.shape), 1)
            vec[t, inside] = b[tuple(grid.states[inside].T)]
        res[key] = vec
    return grid, res


def _exp(op, coef, x):
    out, term, n = x, x, 1
    while True:
        term = coef * op(term) / n
        out = out + term
        if abs(term).max() <= _EPS * abs(out).max():
            return out
        n += 1
        if n > 1000:
            raise ValueError('Frames of the blocks too far apart (use method="exact")')


def relative(lad, bra, src, x):
    """U_bra^† U_src x on the sector of lad, for the frames (G, eta, theta, ...) of two blocks."""
    (Gb, eb, tb, *_), (Gs, es, ts, *_) = bra, src
    H = np.linalg.inv(Gb) @ Gs
    a, c_ = H[..., 0, 1] / H[..., 1, 1], -H[..., 1, 0] / H[..., 1, 1]
    b = -2 * np.log(H[..., 1, 1])
    b = b + 4j * np.pi * np.round(((es - eb) - b).imag / (4 * np.pi))
    for c in range(len(lad.k)):
        x = _exp(lambda z: lad.lin(c, 0, 1, 0, z), c_[:, c, None], x)
        x = np.exp(b[:, c, None] * lad.k0[c]) * x
        x = _exp(lambda z: lad.lin(c, 1, 0, 0, z), a[:, c, None], x)
    return np.exp(1j * (tb - ts))[:, None] * x


class Rotated:
    """Factors of an Op in the frame of a source block (the grid argument of Op.apply)."""

    _SINGLE = {'a1': ((0, 'a1', 0), (1, 'a2d', -1)), 'a2': ((0, 'a2', 0), (1, 'a1d', -1)),
               'a1d': ((2, 'a1d', 0), (3, 'a2', 1)), 'a2d': ((2, 'a2d', 0), (3, 'a1', 1))}
    _DEG = {'a1': ((0, 'a1', 0), (1, 'a1d', -1)), 'a1d': ((2, 'a1d', 0), (3, 'a1', 1))}

    def __init__(self, ev, G, k0, km, ksum):
        self.ev, self.grid, self.r = ev, ev.grid, ev.grid.r
        self.G, self.u, self.v = G, G[:, :, 0, 0, None], G[:, :, 0, 1, None]
        self.ref = classical(G, k0, km)[0].sum(-1) - ksum

    def _M(self, lad):
        u, v = self.u, self.v
        return lambda x: sum(lad.rot('K0', c, u[:, c], v[:, c], x) for c in range(len(lad.k))) - lad.k.sum() * x

    def _drive(self, lad, L, x, sq):
        M = self._M(lad)
        if not sq:
            return L * x - self.r * M(x)
        ref, r = self.ref, self.r
        b = max(lad.bound(g[:, 0, 0], g[:, 0, 1]) for g in self.G)
        n = _order(r * b / max((L - r * ref).min(), 1e-300))
        cs = np.stack([_sqrt_series(L - r * m, -float(r), n) for m in ref], 1)[..., None]
        return _horner(cs, lambda y: M(y) - ref[:, None] * y, x)

    def act(self, name, c, d, L, x):
        ev, r = self.ev, self.r
        lad = ev._ladders(d)
        if name in ('K+', 'K-', 'K0'):
            out = lad.rot(name, c, self.u[:, c], self.v[:, c], x)
            return d, L + (r if name == 'K+' else -r if name == 'K-' else 0), out
        if name in ('N1', 'N2'):
            deg = ev.chains[c].kind == 'degenerate'
            y = lad.rot('K0', c, self.u[:, c], self.v[:, c], x) - lad.k[c] * x
            return d, L, (2 if deg else 1) * y + d[c][name == 'N2'] * x
        if name in self._SINGLE:
            coef = (self.u[:, c], self.v[:, c], np.conj(self.u[:, c]), np.conj(self.v[:, c]))
            table = self._DEG if ev.chains[c].kind == 'degenerate' else self._SINGLE
            res = None
            for i, f, s in table[name.replace('a2', 'a1') if ev.chains[c].kind == 'degenerate' else name]:
                d2, L2, y = self.grid.act(f, c, d, L, x)
                res = (d2, L2 + s * r, coef[i] * y) if res is None else (res[0], res[1], res[2] + coef[i] * y)
            return res
        if name == 'N0':
            return d, L, self._drive(lad, L, x, False)
        if name == 'a0':
            return d, L - 1, self._drive(lad, L, x, True)
        if name == 'a0d':
            return d, L + 1, self._drive(lad, L + 1, x, True)
        raise ValueError(f'The factor {name} is not available in the co-moving frame (use method="exact")')
