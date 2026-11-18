"""
Single-mode three-wave mixing with two strong (stimulated) inputs and one weak input.

The Hamiltonian is H = ħχ (a_1 a_2 a_0† + a_1† a_2† a_0) with pump a_0, signal a_1 and idler a_2
(a_p, a_s, a_i). Two of the three inputs are strong (perturbative order -1) and one is weak
(order 0); χ is of order 1, so that the elliptic argument u = χ t sqrt(L) is of order 0.

Method (see the "Two strong inputs" section of cas_qwm_documentation.md):

1. Conservation laws. M_s = N_s + N_p, M_i = N_i + N_p and H' are conserved, so every field obeys
   a closed second-order Heisenberg equation in which only conserved operators and the field
   itself appear:
       d²a_s/dt² = χ² [(2N_s + N_p - N_i) a_s - 2 a_s† a_s a_s](t),
       d²a_i/dt² = χ² [(2N_i + N_p - N_s) a_i - 2 a_i† a_i a_i](t),
       d²a_p/dt² = χ² [-(N_s + N_i + 2N_p + 1) a_p + 2 a_p† a_p a_p](t),
   where the linear coefficients are conserved combinations placed on the left of the field.
2. Forward recursion. The solution is expanded in perturbative orders; the leading order is the
   classical Jacobi elliptic solution, and every further order solves the variational equation
   with a source given by the residual of the lower orders. The variational equation decouples
   into Lamé equations of degree 1 and 2.

This module only applies the algebra of the jacobi package to the quantum evolution:
- jacobi.elliptic_algebra: the coefficients as functions of the number operators (EllipticTail in
  PF, shifted exactly by NumShift) and their expansion by orders (EllipticAlgebra, registered with
  PE.register_algebra), with which the operator products of the residual are evaluated;
- jacobi.dsolve_lame: the ring Q(m)[n, u, E(u), sn, cn, dn] and the closed-form Lamé solver
  (explicit fundamental solutions, recursive integration as in dsolve_pm).

Coefficients are represented in conserved variables: with the weak mode w, two conserved
"strong" quantities L_X, L_Y and the weak number n = N_w,
    weak pump:   L_X = N_s + N_p, L_Y = N_i + N_p,  m =  L_X / L_Y,
    weak signal: L_X = N_p + N_s, L_Y = N_i - N_s,  m = -L_X / L_Y,
    weak idler:  L_X = N_p + N_i, L_Y = N_s - N_i,  m = -L_X / L_Y,
and u = χ t sqrt(L_Y), λ = L_Y^(-1/2).
"""
from __future__ import annotations
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.singleton import S
from sympy.physics.quantum.dagger import Dagger
from .op_pattern.boson_pattern import BosonNum
from .pattern_perturb import PE, PF, pf_add
from .jacobi.dsolve_lame import R, red, mul, diff_u, conj, solve_lame
from .jacobi.elliptic_algebra import EllipticAlgebra, ls_add, ls_scale, ls_map
from .wm_variables import a_p, a_s, a_i, t_, nlc

__all__ = ['TWMDualSolver', 'exact_fock_apply']

_n, _u, _E, _s, _c, _d, _j = R.gens
_OPS = (a_p, a_s, a_i)
_NOPS = tuple(BosonNum(op.name) for op in _OPS)
_MODES = {'p': 0, 's': 1, 'i': 2, 0: 0, 1: 1, 2: 2, a_p.name: 0, a_s.name: 1, a_i.name: 2}


def _mono(K):
    """Normally ordered monomial of the signed exponents K of (a_p, a_s, a_i)."""
    return Mul(*[_OPS[k]**e if e > 0 else Dagger(_OPS[k])**(-e) for k, e in enumerate(K) if e])


def _exps(key):
    """Signed exponents (a^e: e > 0, (a†)^e: e < 0) of (a_p, a_s, a_i) of a PF table key."""
    K = [0, 0, 0]
    for b in key[0]:
        K[_MODES[b.name]] -= int(b.parity_diff)
    return tuple(K)


# target mode -> (coefficients of (N_0, N_1, N_2) and constant of the conserved linear
# coefficient c, cubic coefficient κ, exponents of the partner product in ȧ(0) = -iχ partner)
_TARGET = {
    0: ((-2, -1, -1), -1, 2, (0, 1, 1)),
    1: ((1, 2, -1), 0, -2, (1, 0, -1)),
    2: ((1, -1, 2), 0, -2, (1, -1, 0)),
}


class TWMDualSolver:
    """
    Perturbative Heisenberg solution of single-mode three-wave mixing with two strong inputs.

    Parameters
    ==========
    weak: the weak input mode, 'p' (sum-frequency generation from strong signal and idler,
        the default), 's' or 'i' (difference-frequency generation from a strong pump and the
        other strong lower-frequency field).

    The solution of mode `target` (0/'p', 1/'s', 2/'i') is a PF whose coefficients are
    EllipticTails of `alg`; the leading term (order -1) is the classical Jacobi elliptic solution,
    and order k contains the corrections of relative order k + 1 (the notes' "first order" is
    order 0). Orders are counted by `pe`: strong modes -1, weak mode 0, λ = L_Y^(-1/2) 1.
    """

    def __init__(self, weak='p'):
        w = self.weak = _MODES[weak]
        if w == 0:
            strong = (1, 2)
            lx, ly, ln, eps = (1, 1, 0), (1, 0, 1), (1, 0, 0), 1
        else:
            strong = (0, 3 - w)
            lx, ly, ln, eps = [0] * 3, [0] * 3, [0] * 3, -1
            lx[0] = lx[w] = ly[3 - w] = ln[w] = 1
            ly[w] = -1
        self.X = strong[0]
        self.alg = EllipticAlgebra(_NOPS, lx, ly, ln, eps)
        self.pe = PE()
        self.pe.update({_OPS[k].name: -1 for k in strong})
        self.pe.register_algebra(self.alg)
        self._sols = {}
        self._data = {}

    # ----------------------------------- PF helpers ---------------------------------------
    def _term(self, op, f):
        """PF of the operator monomial op times the coefficient LSeries f."""
        (k,) = PF.from_val(op).table
        return PF({k: self.alg.tail(f)})

    def _orders(self, P, n):
        """Terms of P expanded up to order n, grouped by order: {order: PF}."""
        out = {}
        for term, o in self.pe.expand_pf(P, n, with_order=True):
            out.setdefault(int(o), []).append(term)
        return {o: pf_add(*ts) for o, ts in out.items()}

    def _korder(self, key):
        return int(sum(self.pe.leading_order(b)[1] for b in key[0]))

    def _ring(self, P, key, p):
        """λ^p coefficient (ring element) of the key in P."""
        t = P.table.get(key)
        return R.zero if t is None else self.alg.series(t).get(p, R.zero)

    # --------------------------------------------------------------------------------------
    def _target_data(self, t):
        if t in self._data:
            return self._data[t]
        alg = self.alg
        coeffs, const, kappa, partner = _TARGET[t]
        if t == self.weak:
            T, F, pF = _mono(partner), -_j * _s, 1
        else:
            T, F, pF = _OPS[t], (_c if t == self.X else _d), 0
        TP = PF.from_val(T)
        (nt,) = (TP.adjoint() * TP).table.values()
        NT = alg.to_ls(nt, alg._lo(nt))
        (pNT, NTc), = NT.items()
        if any(NTc.degrees()):
            raise RuntimeError('leading normalization depends on the weak number operator')
        c_ls = ls_add({0: R(const)}, *[ls_scale(Nm, R(w)) for w, Nm in zip(coeffs, alg.N) if w])
        assert 2 + 2 * pF + pNT == 0 and min(c_ls) == -2
        F2, aF2 = mul(F, F), mul(F, conj(F))
        # variational potentials of P, Q = g ± J[g]: λ²(c + 2κ|F|²N_T ± κF²N_T)
        alpha = c_ls[-2] + 2 * kappa * mul(aF2, NTc)
        gamma = kappa * mul(F2, NTc)
        dat = dict(c=Add(const, *[w * N for w, N in zip(coeffs, _NOPS)]), kappa=kappa,
                   TT=TP * TP, NT=(pNT, R(1 / NTc.LC)),
                   VP=red(alpha + gamma), VQ=red(alpha - gamma),
                   lead=self._term(T, {pF: F}),
                   B0=self._term(_OPS[t], {0: R.one}),      # a(0) = a
                   B1=self._term(_mono(partner), {1: -_j}),  # a'(0) = -iλ partner (u units)
                   jkey={})
        self._data[t] = dat
        return dat

    def _J(self, dat, P, k):
        """
        The antilinear involution J[X] = T² X† / N_T (leading normalization) on the terms of
        order k; it pairs the key K with T² K† and conjugates its coefficient, with the exact
        weak-number shift from NumShift.
        """
        pNT, inv = dat['NT']
        Y = self.pe.expand_pf(dat['TT'] * P.adjoint(), k + pNT)
        return Y.apply(lambda tl: self.alg.map(tl, lambda f: ls_scale(f, inv, -pNT)))

    def _jkey(self, dat, key):
        """The key paired with `key` by J."""
        if key not in dat['jkey']:
            (dat['jkey'][key],) = (dat['TT'] * PF({key: S.One}).adjoint()).table
        return dat['jkey'][key]

    def residual(self, target, A, cap):
        """b'' - λ²[c b + κ b† b b] of a PF b (u units), expanded up to order cap."""
        dat = self._target_data(_MODES[target])
        pe, alg = self.pe, self.alg
        aa = pe.expand_pf(A * A, cap - 1)
        rhs = pe.expand_pf(PF.from_val(dat['c']) * A + dat['kappa'] * (A.adjoint() * aa), cap - 2)
        d2 = A.apply(lambda tl: alg.map(tl, lambda f: ls_map(f, lambda r: diff_u(diff_u(r)))))
        l2 = rhs.apply(lambda tl: alg.map(tl, lambda f: ls_scale(f, R.one, 2)))
        return pe.expand_pf(d2 - l2, cap)

    def _step(self, t, k):
        """
        Order k: g'' = V g + γ J[g] + S splits into P = g + J[g] and Q = g - J[g], which obey
        independent Lamé equations with the potentials V_P, V_Q; each pair {K, J(K)} is solved
        once, at a representative key K: g_K = (P + Q)_K / 2 and g_J(K) = J[(P - Q)_K / 2].
        """
        dat = self._target_data(t)
        res = self._orders(self.residual(t, pf_add(*self._sols[t]), k), k)
        if any(o < k for o in res):
            raise RuntimeError(f'nonzero residual at order {min(res)} while solving order {k}')
        src = -res.get(k, PF())
        b0 = self._orders(dat['B0'], k).get(k, PF())
        b1 = self._orders(dat['B1'], k).get(k, PF())
        ins = (src, b0, b1)
        jins = tuple(self._J(dat, X, k) for X in ins)
        keys = set()
        for X in ins + jins:
            keys.update(X.table)
        out, done = [], set()
        for key in sorted(keys, key=str):
            if key in done:
                continue
            kp = self._jkey(dat, key)
            done.update((key, kp))
            p = k - self._korder(key)
            xs = [self._ring(X, key, p) for X in ins]
            js = [self._ring(X, key, p) for X in jins]
            P = solve_lame(dat['VP'], *[red(x + y) for x, y in zip(xs, js)])
            Q = solve_lame(dat['VQ'], *[red(x - y) for x, y in zip(xs, js)])
            g = red((P + Q) * Rational(1, 2))
            if g:
                out.append(PF({key: self.alg.tail({p: g})}))
            if kp != key:
                h = red((P - Q) * Rational(1, 2))
                if h:
                    out.append(self._J(dat, PF({key: self.alg.tail({p: h})}), k))
        return pf_add(*out) if out else PF()

    def get_sol(self, target, order):
        """List of the PF terms of orders -1, ..., order of mode `target`."""
        t = _MODES[target]
        if t not in self._sols:
            self._sols[t] = [self._target_data(t)['lead']]
        sols = self._sols[t]
        while len(sols) - 2 < order:
            sols.append(self._step(t, len(sols) - 1))
        return sols[:order + 2]

    def sol_pf(self, target, order):
        """The solution accurate up to `order` as one PF."""
        return pf_add(*self.get_sol(target, order))

    # ----------------------------------- output -------------------------------------------
    def symbols(self):
        """SymPy expressions of (L_X, L_Y, m, n) in number operators, and u in t."""
        return self.alg.symbols(nlc * t_)

    def sol_expr(self, target, order, explicit=True):
        """
        SymPy expression of the solution accurate up to `order`. With explicit=False the result
        is written with the symbols u, m, λ, n and sn, cn, dn, E standing for the Jacobi
        functions of (u|m); with explicit=True they are expressed in number operators and t.
        """
        chi_t = nlc * t_ if explicit else None
        P = self.sol_pf(target, order)
        return P.apply(lambda tl: self.alg.to_expr(self.alg.series(tl), chi_t)).expr()

    def apply(self, target, order, fock, chi_t):
        """
        Numerical action of the solution accurate up to `order` on the Fock state |n_p, n_s, n_i>
        at time t (chi_t = χ t). Returns {(n_p, n_s, n_i): amplitude}.
        """
        import mpmath
        nf = list(fock)
        ev = self.alg.evaluator(nf, chi_t)
        out = {}
        for key, tl in self.sol_pf(target, order).table.items():
            amp, nn = 1, list(nf)
            for k, e in enumerate(_exps(key)):
                for _ in range(abs(e)):
                    if e > 0:
                        amp *= mpmath.sqrt(nn[k])
                        nn[k] -= 1
                    else:
                        nn[k] += 1
                        amp *= mpmath.sqrt(nn[k])
            if amp == 0:
                continue
            out[tuple(nn)] = out.get(tuple(nn), 0) + amp * ev(self.alg.series(tl))
        return out


# ============================================================================================
# Exact reference (pure Python): Chebyshev propagation in the conserved blocks
# ============================================================================================

def _bessel_seq(x):
    """J_0(x), ..., J_K(x) up to K where they are negligible (Miller's backward recurrence)."""
    K = int(x + 20 + 10 * x**(1 / 3))
    start = K + 20 + int((40 * K)**0.5)
    vals = [0.0] * (start + 1)
    jp1, j = 0.0, 1e-280
    vals[start] = j
    for k in range(start, 0, -1):
        jp1, j = j, 2 * k / x * j - jp1
        vals[k - 1] = j
        if abs(j) > 1e250:
            for i in range(k - 1, start + 1):
                vals[i] *= 1e-250
            j *= 1e-250
            jp1 *= 1e-250
    norm = vals[0] + 2 * sum(vals[2::2])
    return [v / norm for v in vals[:K + 1]]


def _cheb_expi(beta, v, z):
    """exp(i z H) v for the real symmetric tridiagonal H with zero diagonal and off-diagonal beta."""
    d = len(v)
    if d == 1 or z == 0:
        return list(v)
    a = max((beta[k - 1] if k else 0) + (beta[k] if k < d - 1 else 0) for k in range(d)) * 1.0001
    b = [x / a for x in beta]
    J = _bessel_seq(abs(z) * a)
    ph = 1j if z > 0 else -1j

    def mv(w):
        o = [b[0] * w[1]] + [b[k - 1] * w[k - 1] + b[k] * w[k + 1] for k in range(1, d - 1)]
        return o + [b[d - 2] * w[d - 2]]

    # Jacobi–Anger: exp(i x cos θ) = J_0(x) + 2 Σ_k i^k J_k(x) cos kθ
    t0, t1 = list(v), mv(v)
    phk = ph
    res = [J[0] * x + 2 * phk * J[1] * y for x, y in zip(t0, t1)]
    for k in range(2, len(J)):
        t0, t1 = t1, [2 * x - y for x, y in zip(mv(t1), t0)]
        phk *= ph
        ck = 2 * phk * J[k]
        res = [r + ck * y for r, y in zip(res, t1)]
    return res


def exact_fock_apply(target, fock, chi_t):
    """
    Exact action of the Heisenberg operator a_target(t) on |n_p, n_s, n_i> (single mode). The
    state is propagated in the conserved blocks M_s = N_s + N_p, M_i = N_i + N_p, where H' is a
    tridiagonal matrix in n_p, by a Chebyshev expansion (pure Python, double precision).
    Returns {(n_p, n_s, n_i): amplitude}.
    """
    t = _MODES[target]
    n_p, n_s, n_i = fock
    Ms, Mi = n_s + n_p, n_i + n_p
    chi_t = float(chi_t)

    def beta(Ms, Mi):
        return [((Ms - k) * (Mi - k) * (k + 1.0))**0.5 for k in range(min(Ms, Mi))]

    v = [0j] * (min(Ms, Mi) + 1)
    v[n_p] = 1
    v = _cheb_expi(beta(Ms, Mi), v, -chi_t)
    dMs, dMi = {0: (-1, -1), 1: (-1, 0), 2: (0, -1)}[t]
    Ms1, Mi1 = Ms + dMs, Mi + dMi
    d1 = min(Ms1, Mi1)
    w = [0j] * (d1 + 1)
    for k, x in enumerate(v):
        if t == 0:
            if 1 <= k <= d1 + 1:
                w[k - 1] += k**0.5 * x
        elif k <= d1:
            w[k] += ((Ms if t == 1 else Mi) - k)**0.5 * x
    w = _cheb_expi(beta(Ms1, Mi1), w, chi_t)
    return {(k, Ms1 - k, Mi1 - k): x for k, x in enumerate(w)}
