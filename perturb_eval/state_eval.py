from sympy.core.function import diff
from functools import lru_cache
import warnings

__all__ = ['coherent_average_terms']

@lru_cache(maxsize=None)
def R_coeff(l: int, k: int) -> int:
    """
    Calculates flipped OEIS-A035342
    i.e. the (l, l-k+1) element of the array
    """
    if l == 0:
        return 1 if k == 0 else 0
    if k <= 0 or l < k:
        return 0
    return k*R_coeff(l-1, k) + (k+l-1)*R_coeff(l-1, k-1)

def weak_compositions_upto(dim, Lmax):
    def rec(i, remaining, prefix, last_nz):
        if i == dim:
            if remaining == 0:
                yield tuple(prefix), last_nz
            else:
                yield tuple(prefix), None
            return
        for x in range(remaining + 1):
            prefix.append(x)
            next_last = i if x != 0 else last_nz
            yield from rec(i + 1, remaining - x, prefix, next_last)
            del prefix[-1]

    yield from rec(0, Lmax, [], 0)

class DerivFactorCache:
    """
    Cache for (diff(f)/factorial) per variable.
    Automatically handles one-variable-at-a-time growth along 'grow_index'.
    """
    def __init__(self, f, Ns, lim=None):
        self.f = f
        self.Ns = tuple(Ns)
        self.dim = len(Ns)
        # Upper limit for the order of partial derivatives
        # Also be used for regular cache management
        self.lim = lim
        # key = orders tuple, value = expression (already divided by factorial)
        self.cache = {(0,)*self.dim: f}

    def get(self, orders, i=None, **kwargs):
        """
        Calculates and caches the partial derivatives and coefficient of f,
        given the updated index

        If the updated index is invalid or unspecified, search from the first index

        :param orders: tuple of total derivative orders (l+k)
        :param i: index of variable currently being increased (k_i)
        """
        if len(orders) != self.dim:
            raise ValueError('Wrong input dimension, expected %s' % self.dim)
        if any(n < 0 for n in orders):
            raise ValueError('The order must be nonnegative')
        if self.lim is not None and sum(orders) > self.lim:
            raise ValueError('Exceeds the maximal order limit')
        if orders in self.cache:
            expr = self.cache[orders]
        elif i is None:
            for i, n in enumerate(orders, start=1):
                expr = self.get((0,)*(self.dim-i)+(orders[-i:]), -i, **kwargs)
        else:
            n = orders[i]
            if n == 0:
                """
                If the warning is triggered, the iteration order or cache release logic
                may be inconsistent with the expected growth semantics.
                """
                warnings.warn('Got unexpected index, implementing forced method')
                return self.get(orders)
            expr = (diff(self.get((*orders[:i], 0, *orders[i+1:]), i),
                         (self.Ns[i], 2), **kwargs) / 2 if n == 2
                    else diff(self.get((*orders[:i], n-1, *orders[i+1:]), i),
                              self.Ns[i], **kwargs) / n)
            self.cache[orders] = expr
        return expr

    def release(self, orders, i):
        if len(orders) != self.dim:
            raise ValueError('Wrong input dimension, expected %s' % self.dim)
        # if sum(orders) == self.lim:
        for j in range(i, self.dim):
            for n in range(1, orders[j]+1):
                pos = (*orders[:j], n, *orders[j+1:])
                if pos in self.cache:
                    del self.cache[pos]

def coherent_average_terms(f, Ns, Lmax=0, **kwargs):
    """
    Yields terms of
        prod_i [R(l_i,k_i)/(l_i+k_i)! * N_i^k_i] * ∂^{l+k} f(N)

    :param f: sympy.Expr
    :param Ns: tuple of sympy.Symbol
    :param Lmax: maximal order to keep
    """
    from itertools import product
    cache = DerivFactorCache(f, Ns, lim=Lmax*2)
    dim = len(Ns)

    for l, r in weak_compositions_upto(dim, Lmax):
        # k_i runs from 0..l_i
        for k in product(*[(0,) if li == 0 else range(1, li + 1) for li in l]):
            coeff = 1
            lk = []
            grow_index = -1
            k_prev = [0 for _ in range(dim)]

            for i, (li, ki) in enumerate(zip(l, k)):
                lk.append(li + ki)
                Ri = R_coeff(li, ki)
                coeff *= Ri * (Ns[i] ** ki)
                if k[i] > k_prev[i]:
                    grow_index = i
                k_prev[i] = k[i]

            yield coeff * cache.get(tuple(lk), grow_index, **kwargs)

        if r is not None:
            cache.release(l, r)
