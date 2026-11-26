"""Operators on the chain blocks, as sums of products of ladder, single-field and number operators."""
from collections import defaultdict
import numpy as np

__all__ = ['Op', 'K_plus', 'K_minus', 'K_0', 'N_1', 'N_2', 'a_1', 'a_2', 'a_0', 'a_0_dag', 'N_0',
           'identity', 'diagonal']

_ADJ = {'K+': 'K-', 'K-': 'K+', 'a0': 'a0d', 'a0d': 'a0', 'a1': 'a1d', 'a1d': 'a1', 'a2': 'a2d', 'a2d': 'a2'}


class Op:
    """
    Linear combination of products of factors: K_+, K_-, K_0, N_1, N_2 of a chain, a_1, a_2
    (and adjoints) of an explicit chain, a_0, a_0†, N_0 of the drive, and diagonal functions;
    the rightmost factor acts first.
    """
    __slots__ = ('terms',)

    def __init__(self, terms):
        self.terms = tuple((complex(c), tuple(f)) for c, f in terms if c != 0)

    def __add__(self, other):
        return Op(self.terms + (other if isinstance(other, Op) else identity * other).terms)

    __radd__ = __add__

    def __sub__(self, other):
        return self + other * -1

    def __rsub__(self, other):
        return self * -1 + other

    def __mul__(self, other):
        if isinstance(other, Op):
            return Op([(c1 * c2, f1 + f2) for c1, f1 in self.terms for c2, f2 in other.terms])
        return Op([(c * other, f) for c, f in self.terms])

    __rmul__ = __mul__

    def __pow__(self, e):
        res = identity
        for _ in range(int(e)):
            res = res * self
        return res

    def adjoint(self):
        return Op([(np.conj(c), tuple(_adj(n, a) for n, a in reversed(f))) for c, f in self.terms])

    def apply(self, grid, d, L, v):
        """{(sector, L): image} of the vector(s) v of block (d, L)."""
        out = defaultdict(lambda: 0)
        for c, f in self.terms:
            dd, ll, w = d, L, v
            for name, arg in reversed(f):
                dd, ll, w = grid.act(name, arg, dd, ll, w)
            out[dd, ll] = out[dd, ll] + c * w
        return out


def _adj(name, arg):
    if name == 'fn':
        return name, lambda grid, d, L, f=arg: np.conj(f(grid, d, L))
    return _ADJ.get(name, name), arg


def _op(name, arg=None):
    return Op([(1, ((name, arg),))])


identity = Op([(1, ())])
a_0, a_0_dag, N_0 = _op('a0'), _op('a0d'), _op('N0')


def K_plus(c):
    """Pair creation of chain c (Σ a_1† a_2†, or a_1†² / 2 for a degenerate mode)."""
    return _op('K+', c)


def K_minus(c):
    """Pair annihilation of chain c."""
    return _op('K-', c)


def K_0(c):
    """K_0 = Σ(1 + N_1 + N_2) / 2 of chain c."""
    return _op('K0', c)


def N_1(c):
    """Signal photon number of chain c (the photon number of a degenerate mode)."""
    return _op('N1', c)


def N_2(c):
    """Idler photon number of chain c."""
    return _op('N2', c)


def a_1(c):
    """Signal annihilation operator of the explicit chain c (the mode of a degenerate one)."""
    return _op('a1', c)


def a_2(c):
    """Idler annihilation operator of the explicit chain c (the mode of a degenerate one)."""
    return _op('a2', c)


def diagonal(f):
    """Diagonal operator f(grid, sector, L) -> array over the grid states."""
    return _op('fn', f)
