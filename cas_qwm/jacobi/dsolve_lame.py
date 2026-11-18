"""
Elliptic-polynomial algebra and closed-form Lamé solver.

The ring R = Q(m)[n, u, E, s, c, d, j] with the relations c² = 1 - s², d² = 1 - m s², j² = -1, where
s, c, d = sn, cn, dn(u|m), E = E(u|m) is Jacobi's epsilon function, u the elliptic argument, n a
parameter (e.g. a number operator) and j the imaginary unit. The ring is closed under ∂/∂u and
∂/∂m (closed-form m-derivatives of the Jacobi functions, see jacobi.py).

Like dsolve_pm for the hyperbolic case, everything is analytic and recursive:
- integration in u: ∫ sn^(2k) by its recurrence, everything else by parts; an antiderivative
  outside R (e.g. ∫E du) raises ObstructionError;
- the Lamé equations y'' = (ν(ν+1) m sn² - h) y of degree 1 and 2 that occur in three-wave mixing
  have tabulated closed-form fundamental solutions (LAME), and y'' - V y = S is solved by
  variation of constants (solve_lame).
"""
from __future__ import annotations
from functools import lru_cache
from sympy.core.numbers import Rational
from sympy.core.symbol import Symbol
from sympy.polys.domains import QQ
from sympy.polys.rings import ring

__all__ = ['m', 'R', 'M', 'red', 'mul', 'diff_u', 'diff_m', 'diff_um', 'conj', 'at0',
           'key', 'ObstructionError', 'antideriv', 'integrate0', 'LAME', 'lame', 'solve_lame', 'evaluate']

# ============================================================================================
# Elliptic polynomial ring Q(m)[n, u, E, s, c, d, j] with c² = 1 - s², d² = 1 - m s², j² = -1
# (n: weak number operator, u: elliptic argument, E = E(u|m), s, c, d = sn, cn, dn(u|m),
#  j: imaginary unit)
# ============================================================================================
m = Symbol('m')
_K = QQ.frac_field(m)
R, _n, _u, _E, _s, _c, _d, _j = ring('n u E s c d j', _K)
M = _K.convert(m)
_mgen = _K.gens[0]
_U, _EE, _S, _C, _D, _J = range(1, 7)


@lru_cache(maxsize=None)
def _cd_pow(kc, kd):
    """(1 - s²)^kc (1 - m s²)^kd."""
    return (1 - _s**2)**kc * (1 - M * _s**2)**kd


def red(f):
    """Reduces c², d², j² in a polynomial."""
    if all(mon[_C] < 2 and mon[_D] < 2 and mon[_J] < 2 for mon in f.itermonoms()):
        return f
    out = {}
    extra = R.zero
    for mon, co in f.iterterms():
        ec, ed, ej = mon[_C], mon[_D], mon[_J]
        if ec < 2 and ed < 2 and ej < 2:
            out[mon] = out.get(mon, 0) + co
            continue
        base = list(mon)
        base[_C], base[_D], base[_J] = ec & 1, ed & 1, ej & 1
        if ej & 2:
            co = -co
        extra += R({tuple(base): co}) * _cd_pow(ec >> 1, ed >> 1)
    return red(R(out) + extra) if extra else R(out)


def mul(f, g):
    return red(f * g)


def diff_u(f):
    """∂/∂u at fixed m."""
    if not f:
        return f
    r = f.diff(_u) + f.diff(_E) * (1 - M * _s**2) + f.diff(_s) * _c * _d \
        - f.diff(_c) * _s * _d - f.diff(_d) * M * _s * _c
    return red(r)


def _m_derivs():
    m = M
    X = (_u * (1 - m) - _E) * (1 / (2 * m * (1 - m)))
    dphi = -_d * _E * (1 / (2 * m * (1 - m))) + _d * _u * (1 / (2 * m)) + _s * _c * (1 / (2 * (1 - m)))
    return {
        _S: red(_c * _d * X + _s * _c**2 * (1 / (2 * (1 - m)))),
        _C: red(-_s * dphi),
        _D: red(-_s**2 * _d * (1 / (2 * (1 - m))) - m * _s * _c * X),
        _EE: red((_E - _u) * (1 / (2 * m)) + _d * dphi),
    }


_DM = _m_derivs()
_GEN = {_S: _s, _C: _c, _D: _d, _EE: _E}


def diff_m(f):
    """∂/∂m at fixed u."""
    if not f:
        return f
    r = R({mon: co.diff(_mgen) for mon, co in f.iterterms()})
    for k, g in _GEN.items():
        df = f.diff(g)
        if df:
            r += df * _DM[k]
    return red(r)


def conj(f):
    """Complex conjugate (j -> -j; every other generator is real)."""
    return R({mon: (-co if mon[_J] else co) for mon, co in f.iterterms()})


def at0(f):
    """Value at u = 0 (E = s = 0, c = d = 1)."""
    out = {}
    for mon, co in f.iterterms():
        if mon[_U] or mon[_EE] or mon[_S]:
            continue
        b = list(mon)
        b[_C] = b[_D] = 0
        b = tuple(b)
        out[b] = out.get(b, 0) + co
    return R(out)


def key(f):
    return tuple(sorted(f.iterterms()))


_deriv_cache = {}


def diff_um(f, a, b):
    """∂_u^a ∂m^b f (cached)."""
    if a == 0 and b == 0:
        return f
    k = (key(f), a, b)
    if k not in _deriv_cache:
        _deriv_cache[k] = diff_u(diff_um(f, a - 1, b)) if a else diff_m(diff_um(f, 0, b - 1))
    return _deriv_cache[k]


# ----------------------------------- integration in u ---------------------------------------

class ObstructionError(ArithmeticError):
    """The antiderivative is not an elliptic polynomial (e.g. ∫E du = log θ4 + ...)."""


@lru_cache(maxsize=None)
def _even_red(k):
    """s^(2k) = Q' + α + β d² with Q in family A; returns (Q, α, β)."""
    m = M
    if k == 0:
        return R.zero, _K.one, _K.zero
    if k == 1:
        return R.zero, 1 / m, -1 / m
    # (s^(2k-3) c d)' = (2k-3) s^(2k-4) - (2k-2)(1+m) s^(2k-2) + (2k-1) m s^(2k)
    q1, a1, b1 = _even_red(k - 1)
    q2, a2, b2 = _even_red(k - 2)
    f = 1 / ((2 * k - 1) * m)
    q = (_s**(2 * k - 3) * _c * _d - (2 * k - 3) * q2 + (2 * k - 2) * (1 + m) * q1) * f
    return red(q), (-(2 * k - 3) * a2 + (2 * k - 2) * (1 + m) * a1) * f, \
        (-(2 * k - 3) * b2 + (2 * k - 2) * (1 + m) * b1) * f


def _ell_red(P):
    """For P free of transcendental generators and of family A: P = Q' + α + β d²."""
    Q, al, be = R.zero, R.zero, R.zero
    for mon, co in P.iterterms():
        ks, kc, kd = mon[_S], mon[_C], mon[_D]
        rest = list(mon)
        rest[_S] = rest[_C] = rest[_D] = 0
        coef = R({tuple(rest): co})
        if kc == 0 and kd == 0 and ks % 2 == 0:
            q, a, b = _even_red(ks // 2)
            Q += coef * q
            al += coef * a
            be += coef * b
        elif kc == 1 and kd == 1 and ks % 2 == 1:
            Q += coef * _s**(ks + 1) * Rational(1, ks + 1)
        else:
            raise ObstructionError(f'integrand term s^{ks} c^{kc} d^{kd} is not of family A')
    return red(Q), al, be


def _split_uE(f):
    """Groups f by powers of (u, E): {(a, b): elliptic polynomial}."""
    out = {}
    for mon, co in f.iterterms():
        a, b = mon[_U], mon[_EE]
        mm = list(mon)
        mm[_U] = mm[_EE] = 0
        out.setdefault((a, b), {})[tuple(mm)] = co
    return {k: R(v) for k, v in out.items()}


def antideriv(f):
    """
    F with dF/du = f, for f in Q(m)[n, j][u, E] ⊗ (family A), by recursion on the total degree in
    (u, E): u^a E^b R = (u^a E^b Q)' - (lower degree) + α u^a E^b + β u^a E^b d², and the pure
    terms are reduced with (u^a E^b)' = a u^(a-1) E^b + b u^a E^(b-1) d². Raises ObstructionError
    if a pure term u^a E^b with b > 0 remains (its integral is not an elliptic polynomial).
    """
    res = R.zero
    groups = _split_uE(f)
    while groups:
        dmax = max(a + b for a, b in groups)
        pureJ, pureK = {}, {}   # coefficients of u^a E^b and of u^a E^b d²
        for (a, b) in [k for k in groups if sum(k) == dmax]:
            Q, al, be = _ell_red(groups.pop((a, b)))
            mono = _u**a * _E**b
            if Q:
                res += mul(mono, Q)
                low = R.zero
                if a:
                    low += a * _u**(a - 1) * _E**b * Q
                if b:
                    low += b * _u**a * _E**(b - 1) * mul(_d**2, Q)
                for kk, v in _split_uE(red(-low)).items():
                    groups[kk] = groups.get(kk, R.zero) + v
            if al:
                pureJ[(a, b)] = pureJ.get((a, b), R.zero) + al
            if be:
                pureK[(a, b)] = pureK.get((a, b), R.zero) + be
        # u^a E^b d² = [ (u^a E^(b+1))' - a u^(a-1) E^(b+1) ] / (b+1)
        for (a, b), v in pureK.items():
            res += v * _u**a * _E**(b + 1) * Rational(1, b + 1)
            if a:
                pureJ[(a - 1, b + 1)] = pureJ.get((a - 1, b + 1), R.zero) - v * Rational(a, b + 1)
        for (a, b), v in pureJ.items():
            if not v:
                continue
            if b:
                raise ObstructionError(f'non-elementary term u^{a} E^{b}')
            res += v * _u**(a + 1) * Rational(1, a + 1)
        groups = {k: v for k, v in groups.items() if v}
    return red(res)


def integrate0(f):
    """∫_0^u f du."""
    F = antideriv(f)
    return F - at0(F)


# --------------------------------- Lamé fundamental solutions -------------------------------

def _lame_table():
    """
    Fundamental solutions of the Lamé equations y'' = (n(n+1) m sn² - h) y that occur, keyed by
    (n, h), normalized to y1(0) = 1, y1'(0) = 0, y2(0) = 0, y2'(0) = 1 (Wronskian 1). One solution
    is a Jacobi function (or a derivative of one); the other contains u and E(u).
    """
    m, u, E, s, c, d = M, _u, _E, _s, _c, _d
    one = _K.one
    T = {
        (1, one): (c, (c * ((1 - m) * u - E) + s * d) * (1 / (1 - m))),
        (1, m): (d, (E * d - m * s * c) * (1 / (1 - m))),
        (1, 1 + m): (c * d - s * (u - E), s),
        (2, 1 + m): (c * d, ((1 - m) * u * c * d - (1 + m) * E * c * d + (1 + m**2) * s
                             - m * (1 + m) * s**3) * (1 / (1 - m)**2)),
        (2, 1 + 4 * m): (c - u * s * d + ((2 * m - 1) * E * s * d + m * (1 - 2 * m) * c * s**2)
                         * (1 / (m - 1)), s * d),
        (2, 4 + m): (d - 2 * u * s * c + ((m - 2) * E * s * c + (2 - m) * d * s**2) * (1 / (m - 1)),
                     s * c),
    }
    return {k: (red(y1), red(y2)) for k, (y1, y2) in T.items()}


LAME = _lame_table()


def lame(V):
    """(y1, y2) for a potential V = n(n+1) m s² - h."""
    A = V.coeff(_s**2)
    h = -V.coeff(1)
    if V != A * _s**2 - h or A not in (2 * M, 6 * M):
        raise ValueError(f'not a Lamé potential of degree 1 or 2: {V.as_expr()}')
    key = (1 if A == 2 * M else 2, h)
    if key not in LAME:
        raise ValueError(f'no closed-form Lamé solutions tabulated for {V.as_expr()}')
    return LAME[key]


def solve_lame(V, src, y0, dy0):
    """
    Solves y'' - V y = src with y(0) = y0, y'(0) = dy0 by variation of constants:
    y = y0 y1 + dy0 y2 + y2 ∫_0^u y1 src - y1 ∫_0^u y2 src.
    """
    y1, y2 = lame(V)
    out = mul(y1, y0) + mul(y2, dy0)
    if src:
        out += mul(y2, integrate0(mul(y1, src))) - mul(y1, integrate0(mul(y2, src)))
    return red(out)


# ------------------------------------- numerical value --------------------------------------

def evaluate(f, mval, gvals):
    """
    Numerical value of f at the parameter mval and the generator values gvals = (n, u, E, s, c, d, j)
    (numbers of any type supporting arithmetic, e.g. mpmath numbers).
    """
    def ev(p):
        return sum(mval**e[0] * int(c.numerator) / int(c.denominator) for e, c in p.iterterms())

    out = 0
    for mon, co in f.iterterms():
        v = ev(co.numer) / ev(co.denom)
        for e, g in zip(mon, gvals):
            if e:
                v *= g**e
        out += v
    return out
