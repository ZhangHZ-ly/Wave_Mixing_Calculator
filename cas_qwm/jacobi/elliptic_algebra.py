"""
λ-series of elliptic polynomials as functions of number operators, and their coefficient algebra
for pattern forms (PF) and the perturbative expander (PE).

Three number operators (N_1, N_2, N_3) are mapped linearly onto two large quantities L_X, L_Y and
one parameter n,
    L_X = Σ x_k N_k,  L_Y = Σ y_k N_k,  n = Σ z_k N_k,
with m = ε L_X / L_Y, λ = L_Y^(-1/2) and the elliptic argument u = χ t sqrt(L_Y). A function of
the number operators is an LSeries {p: f_p} = Σ λ^p f_p(u, m, n) with f_p in the ring of
dsolve_lame. As a PF coefficient it is an EllipticTail(id, N_1, N_2, N_3), whose arguments NumShift
shifts exactly; EllipticAlgebra, registered with PE.register_algebra, expands the shifted tails by
orders: exactly in n, and as Taylor series in λ² on (u, m, λ) through the closed-form derivatives
of the Jacobi functions.
"""
from __future__ import annotations
from fractions import Fraction
from sympy.core.add import Add
from sympy.core.function import Function
from sympy.core.numbers import I, Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices import Matrix
from sympy.polys.domains import QQ
from ..op_pattern.boson_pattern import BosonNum
from .jacobi import jacobi_sn, jacobi_cn, jacobi_dn, jacobi_epsilon
from .dsolve_lame import m as _m, R, M, mul, conj, key, diff_um, evaluate

__all__ = ['ls_add', 'ls_mul', 'ls_scale', 'ls_map', 'EllipticTail', 'EllipticAlgebra']

_n, _u, _E, _s, _c, _d, _j = R.gens


# ============================================================================================
# λ-series: {p: ring element} = Σ λ^p f_p
# ============================================================================================

def ls_add(*fs):
    """Sum of LSeries."""
    out = {}
    for f in fs:
        for p, r in f.items():
            v = out.get(p, R.zero) + r
            if v:
                out[p] = v
            else:
                out.pop(p, None)
    return out


def ls_mul(f, g, maxp):
    """Product of LSeries, truncated at λ^maxp."""
    out = {}
    for p, r in f.items():
        for q, h in g.items():
            if p + q > maxp:
                continue
            v = out.get(p + q, R.zero) + mul(r, h)
            if v:
                out[p + q] = v
            else:
                out.pop(p + q, None)
    return out


def ls_scale(f, r, dp=0):
    """λ^dp r f for a ring element r."""
    out = {}
    for p, v in f.items():
        w = mul(v, r)
        if w:
            out[p + dp] = w
    return out


def ls_map(f, fn):
    """fn applied to every coefficient."""
    out = {}
    for p, v in f.items():
        w = fn(v)
        if w:
            out[p] = w
    return out


def _ser_mul(a, b, K):
    """Truncated product of power series in x (lists of ring elements, index = power)."""
    out = [R.zero] * (K + 1)
    for i, x in enumerate(a):
        if not x:
            continue
        for k, y in enumerate(b[:K + 1 - i]):
            if y:
                out[i + k] += mul(x, y)
    return out


def _binom(alpha, k):
    r = Fraction(1)
    for i in range(k):
        r *= (alpha - i) / Fraction(i + 1)
    return Rational(r.numerator, r.denominator)


# ============================================================================================
# PF coefficients
# ============================================================================================

_TAILS = []        # id -> (algebra signature, LSeries)
_TAIL_IDS = {}


def _tail_id(sig, f):
    k = (sig, tuple(sorted((p, key(r)) for p, r in f.items())))
    if k not in _TAIL_IDS:
        _TAIL_IDS[k] = len(_TAILS)
        _TAILS.append((sig, dict(f)))
    return _TAIL_IDS[k]


class EllipticTail(Function):
    """
    EllipticTail(id, N_1, N_2, N_3): the function of the number operators given by the registered
    LSeries `id`, with λ, u, m, n expressed in its arguments by the EllipticAlgebra that created
    it. Shifted arguments (e.g. N_2 - 1) represent the exactly shifted function.
    """

    @classmethod
    def eval(cls, *args):
        return None

    def _eval_adjoint(self):
        sig, f = _TAILS[int(self.args[0])]
        return self.func(Integer(_tail_id(sig, ls_map(f, conj))), *self.args[1:])

    def _sympystr(self, printer):
        return 'F%s(%s)' % (self.args[0], ', '.join(printer._print(a) for a in self.args[1:]))


class EllipticAlgebra:
    """
    Coefficient algebra of LSeries in the number operators `nops` (three BosonNum objects), with
    L_X, L_Y, n given by the integer coefficient tuples `lx`, `ly`, `ln` over `nops` and
    m = eps L_X / L_Y. Implements the PE coefficient-algebra interface (match, leading_order,
    expand) for polynomials in the number operators and the EllipticTails of this algebra.
    """

    def __init__(self, nops, lx, ly, ln, eps=1):
        self.nops = tuple(nops)
        self.lx, self.ly, self.ln = tuple(lx), tuple(ly), tuple(ln)
        self.eps = eps
        self.sig = (self.nops, self.lx, self.ly, self.ln, eps)
        self.nmode = {N: k for k, N in enumerate(self.nops)}
        # number operators in (L_X, L_Y, n) = (ε m λ^-2, λ^-2, n)
        inv = Matrix([self.lx, self.ly, self.ln]).inv()
        self.N = []
        for k in range(3):
            a, b, c = (QQ.convert(inv[k, i]) for i in range(3))
            self.N.append({p: r for p, r in ((-2, R(a * eps) * M + R(b)), (0, c * _n)) if r})
        self._shift_cache = {}
        self._ls_cache = {}
        self._pow_cache = {}

    def lshift(self, dN):
        """Number shift dN -> (dL_X, dL_Y, dn)."""
        return tuple(sum(c * d for c, d in zip(row, dN)) for row in (self.lx, self.ly, self.ln))

    # ------------------------------------- shifts -----------------------------------------
    def _dpow(self, dLX, dLY, K, a, b):
        """
        Series in x = λ² (lists of length K + 1) of Δu^a Δm^b for a shift (dL_X, dL_Y) at fixed t,
        where Δu = u[(1 + dL_Y x)^(1/2) - 1], Δm = x (ε dL_X - m dL_Y) / (1 + dL_Y x);
        with a = 'A': (1 + dL_Y x)^(-b/2). None if the series vanishes.
        """
        ck = (dLX, dLY, K, a, b)
        if ck in self._pow_cache:
            return self._pow_cache[ck]
        if a == 'A':
            res = [R(_binom(Fraction(-b, 2), k) * dLY**k) for k in range(K + 1)]
        elif a == 0 and b == 0:
            res = [R.one] + [R.zero] * K
        elif b == 0:
            prev = self._dpow(dLX, dLY, K, a - 1, 0)
            du = [R.zero] + [_u * _binom(Fraction(1, 2), k) * dLY**k for k in range(1, K + 1)]
            res = None if prev is None else _ser_mul(prev, du, K)
        else:
            prev = self._dpow(dLX, dLY, K, a, b - 1)
            c0 = self.eps * dLX - M * dLY
            dm = [R.zero] + [R(c0 * (-dLY)**(k - 1)) for k in range(1, K + 1)]
            res = None if prev is None else _ser_mul(prev, dm, K)
        if res is not None and not any(res):
            res = None
        self._pow_cache[ck] = res
        return res

    def shift_ring(self, r, p, dLX, dLY, dn, K):
        """λ^p r(u, m, n) at shifted (L_X, L_Y, n), as {power: ring} up to λ^(p + 2K)."""
        ck = (key(r), p, dLX, dLY, dn, K)
        if ck in self._shift_cache:
            return self._shift_cache[ck]
        if dLX == 0 and dLY == 0:
            res = {p: r.compose(_n, _n + dn) if dn else r}
        else:
            ser = [R.zero] * (K + 1)
            fa = 1
            for a in range(K + 1):
                fa *= max(a, 1)
                if not diff_um(r, a, 0):
                    break
                fb = 1
                for b in range(K + 1 - a):
                    fb *= max(b, 1)
                    dr = diff_um(r, a, b)
                    if not dr:
                        break
                    pw = self._dpow(dLX, dLY, K, a, b)
                    if pw is None:
                        break
                    coef = Rational(1, fa * fb)
                    for k, x in enumerate(pw):
                        if x:
                            ser[k] += mul(x, dr) * coef
            ser = _ser_mul(self._dpow(dLX, dLY, K, 'A', p), ser, K)
            res = {}
            for k, x in enumerate(ser):
                if x:
                    res[p + 2 * k] = x.compose(_n, _n + dn) if dn else x
        self._shift_cache[ck] = res
        return res

    def shift(self, f, dN, maxp):
        """f(N + dN), truncated at λ^maxp."""
        dLX, dLY, dn = self.lshift(dN)
        out = {}
        for p, r in f.items():
            if p > maxp:
                continue
            for q, v in self.shift_ring(r, p, dLX, dLY, dn, (maxp - p) // 2).items():
                w = out.get(q, R.zero) + v
                if w:
                    out[q] = w
                else:
                    out.pop(q, None)
        return out

    # ------------------------------ PF coefficients ---------------------------------------
    def tail(self, f):
        """The PF coefficient of an LSeries f (1 and 0 stay numbers)."""
        if not f:
            return S.Zero
        if f == {0: R.one}:
            return S.One
        return EllipticTail(Integer(_tail_id(self.sig, f)), *self.nops)

    def series(self, e):
        """Exact LSeries of a canonical coefficient: a number, an unshifted tail or a sum of them."""
        if e.is_Add:
            return ls_add(*(self.series(x) for x in e.args))
        if isinstance(e, EllipticTail) and e.args[1:] == self.nops:
            return _TAILS[int(e.args[0])][1]
        if e.is_number:
            return self.to_ls(e, 0)
        if e.is_Mul:
            c, rest = e.as_coeff_mul()
            if c != 1 and len(rest) == 1:
                return ls_scale(self.series(rest[0]), self.to_ls(c, 0)[0])
        raise ValueError(f'not a canonical coefficient: {e}')

    def map(self, e, fn):
        """The coefficient fn(series(e)) for a function fn of LSeries."""
        return self.tail(fn(self.series(e)))

    def _lo(self, e):
        """Leading λ power of a recognized expression (a lower bound if terms cancel)."""
        if e.is_number:
            return 0
        if isinstance(e, EllipticTail):
            return min(_TAILS[int(e.args[0])][1])
        if isinstance(e, BosonNum):
            return min(self.N[self.nmode[e]])
        if e.is_Add:
            return min(self._lo(x) for x in e.args)
        if e.is_Mul:
            return sum(self._lo(x) for x in e.args)
        return int(e.exp) * self._lo(e.base)

    def to_ls(self, e, maxp):
        """LSeries of a recognized expression, truncated at λ^maxp."""
        ck = (e, maxp)
        if ck in self._ls_cache:
            return self._ls_cache[ck]
        if e.is_number:
            re, im = e.as_real_imag()
            r = R(QQ.convert(re)) + R(QQ.convert(im)) * _j
            res = {0: r} if r and maxp >= 0 else {}
        elif isinstance(e, EllipticTail):
            f = _TAILS[int(e.args[0])][1]
            dN = tuple(int(x - n0) for x, n0 in zip(e.args[1:], self.nops))
            res = self.shift(f, dN, maxp)
        elif isinstance(e, BosonNum):
            res = {p: v for p, v in self.N[self.nmode[e]].items() if p <= maxp}
        elif e.is_Add:
            res = ls_add(*(self.to_ls(x, maxp) for x in e.args))
        else:
            fs = e.args if e.is_Mul else (e.base,) * int(e.exp)
            los = [self._lo(x) for x in fs]
            tot = sum(los)
            res, rest = {0: R.one}, tot
            for x, lo in zip(fs, los):
                rest -= lo
                res = ls_mul(res, self.to_ls(x, maxp - tot + lo), maxp - rest)
                if not res:
                    break
        self._ls_cache[ck] = res
        return res

    # ------------------------------ PE interface ------------------------------------------
    def match(self, e):
        """Whether e is a polynomial in the number operators and the EllipticTails of this algebra."""
        if e.is_number:
            return False
        ck = ('m', e)
        if ck not in self._ls_cache:
            self._ls_cache[ck] = self._match(e)
        return self._ls_cache[ck]

    def _match(self, e):
        if e.is_number:
            return True
        if isinstance(e, EllipticTail):
            return _TAILS[int(e.args[0])][0] == self.sig
        if isinstance(e, BosonNum):
            return e in self.nmode
        if e.is_Add or e.is_Mul:
            return all(self._match(x) for x in e.args)
        if e.is_Pow:
            return e.exp.is_Integer and e.exp >= 0 and self._match(e.base)
        return False

    def leading_order(self, e):
        return e, Integer(self._lo(e))

    def expand(self, e, n):
        for p, r in sorted(self.to_ls(e, int(n)).items()):
            yield self.tail({p: r}), Integer(p)

    # ------------------------------------ output ------------------------------------------
    def symbols(self, chi_t=None):
        """SymPy expressions of L_X, L_Y, m, n in the number operators, and u = chi_t sqrt(L_Y)."""
        LX, LY, n = (Add(*[c * N for c, N in zip(row, self.nops)])
                     for row in (self.lx, self.ly, self.ln))
        out = dict(LX=LX, LY=LY, m=self.eps * LX / LY, n=n)
        if chi_t is not None:
            out['u'] = chi_t * sqrt(LY)
        return out

    def to_expr(self, f, chi_t=None):
        """
        SymPy expression of an LSeries: in the number operators with u = chi_t sqrt(L_Y) if chi_t
        is given, otherwise with the symbols λ, m, n, u and sn, cn, dn, E for the Jacobi functions.
        """
        gens = [Symbol(str(g)) for g in R.gens]
        if chi_t is not None:
            sy = self.symbols(chi_t)
            uu, mm = sy['u'], sy['m']
            vals = [sy['n'], uu, jacobi_epsilon(uu, mm), jacobi_sn(uu, mm), jacobi_cn(uu, mm),
                    jacobi_dn(uu, mm), I]
            lamv, mv = sy['LY']**Rational(-1, 2), mm
        else:
            vals = [Symbol(x) for x in ('n', 'u', 'E', 'sn', 'cn', 'dn')] + [I]
            lamv, mv = Symbol('λ'), Symbol('m')
        sub = dict(zip(gens, vals))
        sub[_m] = mv
        return Add(*[lamv**p * r.as_expr().xreplace(sub) for p, r in f.items()])

    def evaluator(self, nvals, chi_t):
        """Function returning the numerical value (mpmath) of an LSeries at N = nvals, χt = chi_t."""
        import mpmath
        LX, LY, n = (sum(c * v for c, v in zip(row, nvals)) for row in (self.lx, self.ly, self.ln))
        mval = mpmath.mpf(self.eps * LX) / LY
        uval = chi_t * mpmath.sqrt(LY)
        lam = 1 / mpmath.sqrt(LY)
        gv = [n, uval, jacobi_epsilon._mp(uval, mval)] + \
             [mpmath.ellipfun(k, uval, m=mval) for k in ('sn', 'cn', 'dn')] + [mpmath.mpc(0, 1)]
        return lambda f: sum(lam**p * evaluate(r, mval, gv) for p, r in f.items())
