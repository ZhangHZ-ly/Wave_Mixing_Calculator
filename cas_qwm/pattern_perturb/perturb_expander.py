from sympy.core.containers import Tuple, Dict
from sympy.core.expr import Expr
from sympy.core.singleton import S
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.core.function import FunctionClass, Function, Derivative, diff, Subs, UndefinedFunction
from sympy.core.relational import Relational
from sympy.core.numbers import Integer
from sympy.concrete.summations import Sum
from sympy.integrals.integrals import Integral
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from ..op_pattern import *
from ..op_pattern.boson_pattern import fac, NumShift
from .pattern_form import PF
from collections import defaultdict
from sympy.core.traversal import bottom_up, preorder_traversal
from sympy.series.limits import limit
from sympy.utilities.iterables import multiset_permutations
from functools import lru_cache
import inspect
import warnings

__all__ = ['PE', 'perturb_rec', 'poly_perm']

warnings.simplefilter("once", FutureWarning)
class DerivCache:
    """
    Cache for diff(f) per variable.
    Automatically handles one-variable-at-a-time growth along 'grow_index'.
    """
    def __init__(self, f, xs, lim=None):
        self.f = f
        self.xs = tuple(xs)
        if not self.xs:
            raise ValueError('Requires at least one variable entrance')
        self.dim = len(xs)
        # Upper limit for the order of partial derivatives
        # Also be used for regular cache management
        self.lim = lim
        # key = orders tuple, value = expression (already divided by factorial)
        self.cache = {(0,)*self.dim: f}

    def get(self, orders, i=None, **kwargs):
        """
        Calculates and caches the partial derivatives of f,
        given the updated index.

        If the updated index is invalid or unspecified, search from the last non-zero index.

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
            return self.cache[orders]
        if i is None or orders[i] == 0:
            i = next(j for j in range(self.dim - 1, -1, -1) if orders[j] > 0)
        n = orders[i]
        expr = diff(self.get((*orders[:i], n - 1, *orders[i+1:]), i, **kwargs),
                    self.xs[i], **kwargs)
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
    return k * R_coeff(l - 1, k) + (k + l - 1) * R_coeff(l - 1, k - 1)

class DerivFactorCache:
    """
    Cache for (diff(f)/factorial) per variable.
    Automatically handles one-variable-at-a-time growth along 'grow_index'.
    """
    def __init__(self, f, Ns, lim=None):
        self.f = f
        self.Ns = tuple(Ns)
        self.dim = len(Ns)
        self.lim = lim
        self.cache = {(0,) * self.dim: f}

    def get(self, orders, i=None, **kwargs):
        if len(orders) != self.dim:
            raise ValueError('Wrong input dimension, expected %s' % self.dim)
        if any(n < 0 for n in orders):
            raise ValueError('The order must be nonnegative')
        if self.lim is not None and sum(orders) > self.lim:
            raise ValueError('Exceeds the maximal order limit')
        if orders in self.cache:
            expr = self.cache[orders]
        elif i is None:
            expr = self.f
            for idx, n in enumerate(orders):
                if n > 0:
                    expr = self.get(orders[:idx] + (n,) + (0,) * (self.dim - idx - 1), idx, **kwargs)
        else:
            if i < 0:
                i = self.dim + i
            n = orders[i]
            if n == 0:
                return self.get(orders)
            prev_orders = orders[:i] + (n - 1,) + orders[i + 1:]
            expr = diff(self.get(prev_orders, i, **kwargs), self.Ns[i], **kwargs) / n
            self.cache[orders] = expr
        return expr

    def release(self, orders, i):
        if len(orders) != self.dim:
            raise ValueError('Wrong input dimension, expected %s' % self.dim)
        if i < 0:
            i = self.dim + i
        prefix = orders[:i]
        to_del = [k for k in self.cache if k[:i] == prefix and any(k[j] > 0 for j in range(i, self.dim))]
        for k in to_del:
            del self.cache[k]
@lru_cache()
def gperm(x, m: int|Integer):
    if m < 0:
        return 0 if isinstance(m, int) else S.Zero
    if m == 0:
        return 1 if isinstance(m, int) else S.One
    return gperm(x, m-1) * (x+1-m)

extract_modes_cache = dict()
def extract_modes_cached(x):
    if x in extract_modes_cache:
        return extract_modes_cache[x]
    result = extract_modes(x)
    extract_modes_cache[x] = result
    return result
check_num_cache = dict()
commute_dep_cache = dict()
commute_ind_cache = dict()

def commute(a, b, _comm=None, _is_comm=None, _is_num=None, **hints):
    if a is b:
        return True
    if _is_comm is None:
        _is_comm = lambda x : x.is_commutative
    if _is_num is None:
        _is_num = lambda x: check_num(x, check_num_cache)
    if _is_comm(a) or _is_comm(b):
        return True
    if hints.get('independent', False):
        if a in commute_ind_cache and b in commute_ind_cache[a]:
            return commute_ind_cache[a][b]
        elif b in commute_ind_cache and a in commute_ind_cache[b]:
            return commute_ind_cache[b][a]
        elif a not in commute_ind_cache:
            commute_ind_cache[a] = dict()
        if _is_num(a) and _is_num(b):
            result = True
        elif _comm is None:
            ba, fa, na, _, _ = extract_modes_cached(a)
            bb, fb, nb, _, _ = extract_modes_cached(b)
            if fa or na or fb or nb:
                pass
            elif not (ba & bb) or Commutator(a, b).doit(independent=True) == 0:
                result = True
            result = False
        else:
            result = _comm(a, b)
        commute_ind_cache[a][b] = result
        return result
    if _comm is None:
        if a in commute_dep_cache and b in commute_dep_cache[a]:
            return commute_dep_cache[a][b]
        elif b in commute_dep_cache and a in commute_dep_cache[b]:
            return commute_dep_cache[b][a]
        elif a not in commute_dep_cache:
            commute_dep_cache[a] = dict()
        ba = extract_modes_cached(a)[0]
        bb = extract_modes_cached(b)[0]
        if (len(ba) == 1 and ba == bb
            and check_num(a, check_num_cache) and check_num(b, check_num_cache)):
            result = True
        elif Commutator(a, b).doit() == 0:
            result = True
        result = False
    else:
        result = _comm(a, b)
    commute_dep_cache[a][b] = result
    return result

def poly_perm(ts, _comm=None, _is_comm=None, _is_num=None,
              _add=Add, _mul=None, _pow=None, **hints):
    items = [(k, v) for k, v in (ts.items() if isinstance(ts, dict) else ts) if v]
    if not items:
        return S.One
    if _mul is None:
        _mul = (lambda args: Mul(*args).doit(**hints)) if hints.get('deep', False) else (lambda args: Mul(*args))
    if _pow is None:
        _pow = (lambda a, m: Pow(a, m).doit(**hints)) if hints.get('deep', False) else pow
    coeffs = []
    nc = []
    for a, m in items:
        if all(commute(a, b, _comm, _is_comm, _is_num, **hints) for b, _ in items):
            coeffs.append(_pow(a, m) / fac(m))
        else:
            nc.append((a, m))
    if nc:
        total_order = sum(m for _, m in nc)
        l = sum(([i] * m for i, (_, m) in enumerate(nc)), [])
        coeffs.append(
            _add(*(_mul([nc[i][0] for i in p]) for p in multiset_permutations(l))) / fac(total_order)
        )
    return coeffs[0] if len(coeffs) == 1 else _mul(coeffs)

def perturb_rec( _comm=None, _is_comm=None, _is_num=None,
                _add=Add, _mul=None, _pow=None, **hints):
    """
    Decorator factory that unifies multivariable perturbation recursion and Taylor expansions
    with support for non-commutative operator algebra.

    Encapsulates the branch-and-bound recursive search over perturbation orders,
    automatically assembling symmetrized operator permutations (`poly_perm`) and managing
    derivative caching via `DerivCache`.

    Parameters
    ----------
    _comm : callable, optional
        Custom commutator checker `_comm(a, b) -> bool`.
        If None, falls back to mode commutation checks or `Commutator(a, b).doit() == 0`.
    _is_comm : callable, optional
        Predicate `_is_comm(x) -> bool` testing if an element commutes with all others.
        Defaults to `lambda x: x.is_commutative`.
    _is_num : callable, optional
        Predicate `_is_num(x) -> bool` testing if an element is a scalar number.
        Defaults to `check_num(x, check_num_cache)`.
    _add : callable, optional
        Addition operator function `_add(*args)`. Defaults to SymPy's `Add`.
    _mul : callable, optional
        Multiplication operator function `_mul(args)`.
        Defaults to `Mul(*args).doit(**hints)` if `hints.get('deep')` else `Mul(*args)`.
    _pow : callable, optional
        Power operator function `_pow(a, m)`.
        Defaults to `Pow(a, m).doit(**hints)` if `hints.get('deep')` else `pow`.
    **hints : dict
        Additional algebra or expansion hints passed down to `poly_perm`, `DerivCache.get`,
        `commute`, etc. (e.g. `independent=True`, `deep=True`).

    Returns
    -------
    decorator : callable
        Takes an evaluation callback `term` and returns `wrapper(ao, n, xs, f=None)`.

    Decorated Function (`term`)
    ---------------------------
    term(arg) -> nf or None
        Evaluates the coefficient / prefactor for a given order combination.
        - If `dfc` is active (`f` is not None and `xs` is a sequence):
          `arg` is the cached partial derivative:
          `dfc.get(tuple(deriv_orders), **hints)`
        - If `dfc` is None (`xs` is an int or `f` is None):
          `arg` is `deriv_orders`, a list of non-negative integers specifying
          the variation count (derivative order) for each variable index.
        - Return value `nf`: the evaluated prefactor (e.g. scalar derivative value,
          falling factorial factor `gperm(m, tot) * lt**(m - tot)`, or WMWrapper).
          If `nf is None` or `nf == 0` (or `S.Zero`), the recursion branch is pruned.

    Wrapper Function (`wrapper`)
    ----------------------------
    wrapper(ao, n, xs, f=None) -> Generator[(term_expr, order)]
        Executes the recursive perturbation expansion.
        - `ao` : list of tuples `(idx, weight, term)`
          - `idx` (`int`): Variable index in `[0, l-1]` being perturbed.
          - `weight` (`int` or number): Perturbation order `w > 0` of this variation.
          - `term`: The perturbation term / operator variation.
        - `n` : maximum perturbation order budget.
        - `xs` : either `int` (total number of independent variables `l`),
          or a sequence of symbols/Dummy variables of length `l`.
        - `f` : optional expression for base function `f(*xs)` to differentiate.
        - Yields: `(expr, order)` pairs where `expr = _mul((*cs, nf))` and
          `order = n - remaining = sum(x_k * w_k)` is the relative perturbation order.
        - Automatically clears mode, commutativity, and number check caches on exit.
    """
    if _mul is None:
        _mul = (lambda args: Mul(*args).doit(**hints)) if hints.get('deep', False) else (lambda args: Mul(*args))
    if _pow is None:
        _pow = (lambda a, m: Pow(a, m).doit(**hints)) if hints.get('deep', False) else pow

    def decorator(term):
        def wrapper(ao, n, xs, f=None):
            if isinstance(xs, int):
                l = xs
                dfc = None
            else:
                l = len(xs)
                dfc = DerivCache(f, xs) if f is not None else None
            ao = sorted(ao, key=lambda item: (item[0], item[1]))
            dim = len(ao)
            min_i = {i for i in range(dim) if i == 0 or ao[i][0] != ao[i - 1][0]}

            def rec(i, remaining, prefix, deriv_orders):
                if i == dim:
                    arg = dfc.get(tuple(deriv_orders), **hints) if dfc is not None else deriv_orders
                    nf = term(arg)
                    if nf is None or nf == 0:
                        return
                    collection = [[] for _ in range(l)]
                    for (idx, _, ta), o in zip(ao, prefix):
                        if o:
                            collection[idx].append((ta, o))
                    cs = tuple(poly_perm(pt, _comm, _is_comm, _is_num, _add, _mul, _pow, **hints)
                               for pt in collection if pt)
                    yield _mul((*cs, nf)), n - remaining
                    return
                curr, w, _ = ao[i]
                max_count = remaining // w
                for x in range(max_count + 1):
                    prefix.append(x)
                    deriv_orders[curr] += x
                    yield from rec(i + 1, remaining - x * w, prefix, deriv_orders)
                    if dfc is not None and x == max_count and i in min_i:
                        dfc.release(deriv_orders, curr)
                    del prefix[-1]
                    deriv_orders[curr] -= x

            yield from rec(0, n, [], [0] * l)
            extract_modes_cache.clear()
            check_num_cache.clear()
            commute_dep_cache.clear()
            commute_ind_cache.clear()

        return wrapper
    return decorator

class PE:
    """
    Includes methods that can expand an expression
    composed of symbols with self-defined orders

    order of 0: ~ 1
    order of 1: << 1, adds one order of perturbation
    order of -1: >> 1, cancels one order of perturbation
    order of oo (S.Infinity): pure zero
    """
    exclude = frozenset({Add, Mul, Pow, conjugate, Abs,
                        Integral, Sum, Derivative, Subs,
                        Commutator, AntiCommutator, UidCommutator})
    __slots__ = ('assumptions', 'rc', 'rs', 'rp', '_lo_cache', '_subs_cache')

    def __init__(self, assumptions):
        self.assumptions = Dict(assumptions)
        self.rc = set()
        self.rs = {Symbol}
        self.rp = {
            BosonPat: ('name', lambda p: Add(*p.exps)),
            FermionPat: ('name', lambda p: Add(*p.exps)),
        }
        self._lo_cache = dict()
        self._subs_cache = dict()

    def register(self, t:type, category='s', cancel=False, id='name', fh=lambda _: 1):
        """
        Include the new type of expression in one of the cases
        where the order of the expression should be directly given.

        category 'c': constant type, always considered as zero-order

        category 's': symbolic type, has constant order to be given (default 0)

        category 'p': pattern type, identified by attribute id, has order
        factorized by the given coeff (default 0) times the total exponent
        determined by the received function handle

        If cancel == True, remove the given type from any 
        """
        if t in self.exclude or issubclass(t, Function):
            warnings.warn('Type excluded from registration')
            return
        if cancel:
            self.rc.discard(t)
            self.rs.discard(t)
            if t in self.rp:
                del self.rp[t]
            return
        match category:
            case 'c':
                self.rc.add(t)
            case 's':
                self.rs.add(t)
            case 'p':
                self.rp[t] = (id, fh)
    
    def update(self, orders: dict | None = None):
        """Update assumptions with custom symbol/mode orders."""
        assumptions = dict(self.assumptions)
        clr = False
        if orders:
            for k, v in orders.items():
                if v is not None:
                    if assumptions.get(k, v) != sympify(v):
                        clr = True
                    assumptions[k] = sympify(v)
        self.assumptions = Dict(assumptions)
        if clr:
            self._lo_cache.clear()
            self._subs_cache.clear()
    
    def copy(self, cache=False):
        """
        Returns a copy of the current instance.
        If cache == True, also copies the cache of leading order.
        """
        new = PE(self.assumptions)
        new.rc = self.rc.copy()
        new.rs = self.rs.copy()
        new.rp = self.rp.copy()
        if cache:
            new._lo_cache = self._lo_cache.copy()
            new._subs_cache = self._subs_cache.copy()
        return new

    def leading_order(self, expr):
        """
        Returns the leading order of the expr:
        the Expr of the leading contribution with its order.

        For registered type, the order of an expression is directly given
        by dict self.assumptions, and those are not included in the dict
        are considered as zero-order by default.
        """
        if expr is S.Zero:
            return S.Zero, S.Infinity
        if expr.is_number:
            return expr, S.Zero
        if isinstance(expr, (Abs, conjugate)):
            le, oe = self.leading_order(expr.args[0])
            return expr.func(le), oe
        if isinstance(expr, tuple(self.rc)):
            return expr, S.Zero
        if isinstance(expr, tuple(self.rs)):
            if expr in self.assumptions:
                return expr, sympify(self.assumptions[expr])
            name = getattr(expr, 'name', None)
            if name:
                for k, v in self.assumptions.items():
                    if getattr(k, 'name', str(k)) == name:
                        return expr, sympify(v)
            if expr in self._subs_cache:
                return self.leading_order(self._subs_cache[expr])
            return expr, S.Zero
        if isinstance(expr, tuple(self.rp)):
            id, fh = self.rp[next(t for t in self.rp if isinstance(expr, t))]
            eid = getattr(expr, id)
            if eid in self.assumptions and self.assumptions[eid] != 0:
                return expr, fh(expr) * sympify(self.assumptions[eid])
            name = getattr(eid, 'name', str(eid))
            if name in self.assumptions and self.assumptions[name] != 0:
                return expr, fh(expr) * sympify(self.assumptions[name])
            return expr, S.Zero
        if expr in self._lo_cache:
            return self._lo_cache[expr]
        if isinstance(expr, Function):
            # Our framework only generates Zero-order inputs for functions
            # The output may be zero but is always considered as zeroth order
            new_args = []
            for arg in expr.args:
                l, o = self.leading_order(arg)
                if (o >= 0) != True:
                    raise ValueError('Only accepts nonnegative-order inputs for functions')
                new_args.append(l)
            return expr.func(*new_args), S.Zero
        if isinstance(expr, Derivative):
            le, oe = self.leading_order(expr.expr)
            lo = []
            for vc in expr.variable_count:
                if isinstance(vc, Tuple):
                    ov = self.leading_order(vc[0])[1]
                    if ov != 0:
                        lo.append(-vc[1]*ov)
                else:
                    lo.append(-self.leading_order(vc)[1])
            return Derivative(le, expr.variable_count), Add(oe, *lo)
        if isinstance(expr, (Integral, Sum)):
            # assume that the integral is always finite and irrelevant to the order
            f, m = self.leading_order(expr.function)
            if m.atoms(Symbol) & set(expr.variables):
                raise ValueError('Expect sum/intengral indices only as coefficients')
            return expr.func(f, *expr.limits), m
        if isinstance(expr, (Commutator, AntiCommutator, UidCommutator)):
            # the commutation relation that causes order change has been considered by NumShift
            a, b = expr.args
            la, oa = self.leading_order(a)
            lb, ob = self.leading_order(b)
            return expr.func(la, lb), oa + ob
        elif isinstance(expr, Subs):
            # assume that this only appears with Derivative,
            # used for assigning values to the differential variables
            self._subs_cache = dict(zip(expr.variables, expr.point))
            subs = dict()
            for p in expr.point:
                subs[p] = self.leading_order(p)[0]
                t = expr.xreplace(subs)
                m = self.leading_order(expr.expr)[1]
            self._subs_cache.clear()
        elif expr.is_Add:
            m = S.Infinity
            terms = []
            for arg in expr.args:
                l, o = self.leading_order(arg)
                if m == o:
                    terms.append(l)
                else:
                    cond = m is S.Infinity or m > o
                    if isinstance(cond, Relational):
                        warnings.warn('Unable to determine the leading order', FutureWarning)
                        if o.is_number:
                            m = o
                            terms = [l]
                        continue
                    if cond:
                        m = o
                        terms = [l]
            t = Add(*terms)
        elif expr.is_Mul:
            ma = []
            ta = []
            for arg in expr.args:
                l, o = self.leading_order(arg)
                ma.append(o)
                ta.append(l)
            m = Add(*ma)
            t = Mul(*ta)
        elif expr.is_Pow:
            if expr.base is S.NegativeOne and expr.exp.is_real:
                return expr, S.Zero
            le, oe = self.leading_order(expr.exp)
            if oe is S.Infinity:
                return S.One, S.Zero
            lb, ob = self.leading_order(expr.base)
            if oe is S.Zero:
                t, m = lb ** le, ob * le
            else:
                raise ValueError('Nonzero order in the exponent')
        else:
            warnings.warn('Unable to determine the leading order', FutureWarning)
            return expr, S.Zero

        if m is S.Infinity:
            return S.Zero, S.Infinity
        self._lo_cache[expr] = (t, m)
        return t, m

    def expand_expr(self, expr, n, **kwargs):
        """
        Yields perturbative terms up to the order of n.
        Terms with unidentified order will be ignored.
        """
        n = sympify(n)
        # we expect the order limit to be identified
        if not n.is_number or n.is_infinite:
            raise ValueError('Expect identified finite order')
        if expr in self._subs_cache:
            yield from self.expand_expr(self._subs_cache[expr], n)
            return
        if expr.is_number or isinstance(expr, (*self.rc, *self.rs, *self.rp)):
            o = self.leading_order(expr)[1]
            if o is S.Infinity or not o.is_number:
                yield from ()
            elif o <= n:
                yield expr, o
        elif isinstance(expr, Subs):
            if not isinstance(expr.expr, Derivative):
                ne = expr.doit(**kwargs)
                if ne == expr:
                    yield from self.expand_expr(ne, n, **kwargs)
                    return
                raise ValueError('Failed to deal with the expression')
            # assume that this only appears with Derivative,
            # used for assigning values to the differential variables
            self._subs_cache = dict(zip(expr.variables, expr.point))
            yield from self.expand_expr(expr.expr, n, **kwargs)
            self._subs_cache.clear()
        elif isinstance(expr, (Abs, conjugate)):
            le, oe = self.leading_order(expr)
            if oe is not S.Infinity and oe.is_number and oe <= n:
                yield le, oe
        elif isinstance(expr, (Integral, Sum)):
            # assume that the integral is always finite and irrelevant to the order
            yield from ((expr.func(t, *expr.limits), o)
                         for t, o in self.expand_expr(expr.function, n))
        elif expr.is_Add:
            for arg in expr.args:
                yield from self.expand_expr(arg, n)
        elif expr.is_Mul:
            lo = [self.leading_order(arg)[1] for arg in expr.args]
            n0 = 0
            for o in lo:
                if o is S.Infinity:
                    yield from ()
                    return
                if not o.is_number or o.is_infinite:
                    raise ValueError('Expect the leading order to be an indentified number')
                n0 += o
            if n0 > n:
                yield from ()
                return
            ts = []
            d = {}
            for arg, o in zip(expr.args, lo):
                ni = n - n0 + o
                for ta, oa in self.expand_expr(arg, ni, **kwargs):
                    od = oa - o
                    d[od] = d.get(od, S.Zero) + ta
                ts.append(d.copy())
                d.clear()
            dim = len(ts)
            def rec(i, remaining, prefix):
                if i == dim:
                    yield Mul(*prefix), n - remaining
                    return
                for k, v in ts[i].items():
                    if k <= remaining:
                        prefix.append(v)
                        yield from rec(i + 1, remaining - k, prefix)
                        del prefix[-1]
            yield from rec(0, n-n0, [])
        elif expr.is_Pow and expr.exp.is_number:
            lt, lo = self.leading_order(expr.base)
            m = expr.exp
            if not lo.is_number:
                raise ValueError('Expect the leading order to be an identified number')
            if (lt is S.Zero
                or (lt**m is S.Zero and m.is_infinite)
                or lo * m > n):
                yield from ()
                return
            d = {}
            n_new = n - m * lo
            for t, o in self.expand_expr(expr.base, n_new):
                if o == lo:
                    continue
                o -= lo
                d[o] = d.get(o, 0) + t
            if not d:
                yield lt**m, lo * m
                return
            ao = [(0, o, t) for o, t in d.items()]
            @perturb_rec(**kwargs)
            def term(deriv_orders):
                tot = deriv_orders[0]
                if m.is_Integer and 0 <= m < tot:
                    return None
                return Mul(gperm(m, tot), lt**(m - tot))
            for res, o in term(ao, n_new, 1):
                yield res, o + lo * m
        else:
            if isinstance(expr, Derivative):
                args = expr.expr.args if isinstance(expr.expr, Function) else ()
                if any((v[0] if isinstance(v, Tuple) else v) not in args
                       for v in expr.variable_count):
                    # canonicalize the expression
                    ne = expr.doit(**kwargs)
                    if ne == expr:
                        yield from self.expand_expr(ne, n, **kwargs)
                        return
                    raise ValueError('Failed to deal with the expression')
                lo = []
                for vc in expr.variable_count:
                    if isinstance(vc, Tuple):
                        ov = self.leading_order(vc[0])[1]
                        if not ov.is_number:
                            raise ValueError('Expect differential variable with identified order')
                        if ov != 0:
                            lo.append(-vc[1]*ov)
                    else:
                        lo.append(-self.leading_order(vc)[1])
                n = Add(n, *lo)
                if n < 0:
                    yield from ()
                    return
                a0 = []
                ao = []
                syms = dict()
                d = {}
                for i, arg in enumerate(args):
                    if isinstance(arg, Symbol):
                        syms[arg] = i
                    for ta, oa in self.expand_expr(arg, n):
                        if (oa < 0) == True:
                            raise ValueError('Expect nonnegative-order inputs for functions')
                        if oa.is_number:
                            d[oa] = d.get(oa, S.Zero) + ta
                    if S.Zero in d:
                        a0.append(d.pop(S.Zero))
                    else:
                        a0.append(S.Zero)
                    for oa, ta in d.items():
                        ao.append((i, oa, ta))
                    d.clear()
                d0 = tuple(Dummy() for _ in a0)
                subs_pt = dict(zip(d0, (i[2] for i in ao)))
                ini_order = [0] * len(args)
                for v in expr.variable_count:
                    if isinstance(v, Tuple):
                        ini_order[syms[v[0]]] += v[1]
                f_arg = None
                @perturb_rec(**kwargs)
                def term(deriv_orders):
                    return Derivative(
                        expr.expr.func(*d0),
                        *((d, p + q) for d, p, q in zip(d0, ini_order, deriv_orders))
                    ).subs(subs_pt)
            elif expr.is_Pow or isinstance(expr, Function):
                if n < 0:
                    yield from ()
                    return
                a0 = []
                ao = []
                d = {}
                for i, arg in enumerate(expr.args):
                    for ta, oa in self.expand_expr(arg, n):
                        if (oa < 0) == True:
                            raise ValueError('Expect nonnegative-order inputs for functions')
                        if oa.is_number:
                            d[oa] = d.get(oa, S.Zero) + ta
                    if S.Zero in d:
                        a0.append(d.pop(S.Zero))
                    else:
                        a0.append(S.Zero)
                    for oa, ta in d.items():
                        ao.append((i, oa, ta))
                    d.clear()
                if not ao:
                    yield expr.func(*a0), S.Zero
                    return
                d0 = tuple(Dummy() for _ in a0)
                if isinstance(expr.func, UndefinedFunction):
                    subs_pt = dict(zip(d0, (i[2] for i in ao)))
                    f_arg = None
                    @perturb_rec(**kwargs)
                    def term(deriv_orders):
                        return Derivative(
                            expr.func(*d0), *((d, q) for d, q in zip(d0, deriv_orders))
                        ).subs(subs_pt)
                else:
                    subs_a0 = dict(zip(d0, a0))
                    f_arg = expr.func(*d0)
                    @perturb_rec(**kwargs)
                    def term(df):
                        return df.subs(subs_a0)
            else:
                ne = expr.doit(**kwargs)
                if ne != expr:
                    yield from self.expand_expr(ne, n, **kwargs)
                    return
                warnings.warn('Failed to deal with the expression', FutureWarning)
                o = self.leading_order(expr)[1]
                if o is not S.Infinity and o <= n:
                    yield expr, o
                return
            yield from term(ao, n, d0, f_arg)

    def lo_pf(self, pf: PF, **kwargs):
        """
        Returns the leading-order pattern-form expression and its order under kwargs rules.
        """
        m = S.Infinity
        for (bs, fs), t in pf.table.items():
            if t == 0 or t is S.Zero:
                continue
            nb = Add(*(self.leading_order(b)[1] for b in bs + fs))
            ot = self.leading_order(t)[1]
            if ot is not S.Infinity:
                o = nb + ot
                cond = m is S.Infinity or m > o
                if isinstance(cond, Relational):
                    warnings.warn('Unable to determine the leading order', FutureWarning)
                    if o.is_number:
                        m = o
                    continue
                if cond:
                    m = o
        if pf.npf != 0 and pf.npf is not S.Zero:
            on = self.leading_order(pf.npf)[1]
            if on is not S.Infinity:
                cond = m is S.Infinity or m > on
                if isinstance(cond, Relational):
                    warnings.warn('Unable to determine the leading order', FutureWarning)
                    if on.is_number:
                        m = on
                elif cond:
                    m = on

        if m is S.Infinity:
            return PF(), S.Infinity

        new_t = dict()
        for (bs, fs), t in pf.table.items():
            nb = Add(*(self.leading_order(b)[1] for b in bs + fs))
            new_t[(bs, fs)] = Add(*(e[0] for e in self.expand_expr(t, m-nb, **kwargs)))
        return PF(new_t, Add(*(e[0] for e in self.expand_expr(pf.npf, m, **kwargs)))), m

    def expand_pf(self, pf: PF, n, with_order=False, **kwargs):
        """
        Expand pattern-form table-like expressions upto the given order n.
        If with_order=True, returns a dict mapping each order to the corresponding PF.
        """
        if with_order:
            orders = dict()
            for (bs, fs), t in pf.table.items():
                nb = Add(*(self.leading_order(b)[1] for b in bs + fs))
                for term, o in self.expand_expr(t, n-nb, **kwargs):
                    tot_o = nb + o
                    if tot_o not in orders:
                        orders[tot_o] = [dict(), S.Zero]
                    tbl = orders[tot_o][0]
                    tbl[(bs, fs)] = tbl.get((bs, fs), S.Zero) + term

            if pf.npf != 0 and pf.npf is not S.Zero:
                for term, o in self.expand_expr(pf.npf, n, **kwargs):
                    if o not in orders:
                        orders[o] = [dict(), S.Zero]
                    orders[o][1] += term

            return {o: PF(tbl, npf) for o, (tbl, npf) in orders.items()}

        new_t = dict()
        for (bs, fs), t in pf.table.items():
            nb = Add(*(self.leading_order(b)[1] for b in bs + fs))
            new_t[(bs, fs)] = Add(*(e[0] for e in self.expand_expr(t, n-nb, **kwargs)))
        return PF(new_t, Add(*(e[0] for e in self.expand_expr(pf.npf, n, **kwargs))))

    def cohr_avr(self, expr, cohr_dict, n,
                 update_self=False, update_cohr_dict=True, **kwargs):
        """
        Yields perturbative expansion terms of the expectation value of expr
        under coherent state(s) for specified mode(s), up to the given order.

        :param expr: Expr or operator expression.
        :param cohr_dict: Mapping mode identifiers to coherent amplitudes alpha.
        :param n: Perturbative order cutoff.
        """
        n = sympify(n)
        if not update_self:
            pe_cohr = self.copy(cache=False)
        else:
            pe_cohr = self
        assump = dict(pe_cohr.assumptions)
        if update_cohr_dict:
            ncd = {}
            for k, v in cohr_dict.items():
                if hasattr(k, 'name'):
                    k = k.name
                k = sympify(k)
                ncd[k] = sympify(v)
            cohr_dict = ncd

        Ns = []
        subs_dict = {}
        weights = []
        def _reg(bn):
            if isinstance(bn, BosonNum) and bn.name in cohr_dict:
                bname = bn.name
                if bn not in Ns:
                    default_order = assump.get(bname, S.Zero)
                    assump[bname] = default_order if default_order < 0 else S.NegativeOne
                    Ns.append(bn)
                    subs_dict[bn] = Pow(Abs(cohr_dict[bname]), 2)
                    weights.append(-2 * assump[bname])
            return bn
        bottom_up(expr, _reg)
        pe_cohr.assumptions = Dict(assump)

        if not Ns:
            yield from pe_cohr.expand_expr(expr, n, **kwargs)
            return
        lo_base = pe_cohr.leading_order(expr)[1]
        if lo_base is S.Infinity:
            return
        rem_budget = n - lo_base
        if rem_budget < 0:
            return

        Ns = tuple(Ns)
        dim = len(Ns)
        dfc = DerivFactorCache(expr, Ns)

        def rec(i, remaining, coeff_list, deriv_orders, last_nz):
            if i == dim:
                cat = Mul(*coeff_list) * dfc.get(tuple(deriv_orders), last_nz, **kwargs)
                if cat is S.Zero or cat == 0:
                    return
                for t, ot in pe_cohr.expand_expr(cat, n, **kwargs):
                    yield t.subs(subs_dict), ot
                return

            w = weights[i]
            max_l = remaining // w
            for l in range(max_l + 1):
                k_range = (0,) if l == 0 else range(1, l + 1)
                for k in k_range:
                    c = R_coeff(l, k) * (Ns[i] ** k)
                    m = l + k
                    coeff_list.append(c)
                    deriv_orders[i] = m
                    next_last = i if m != 0 else last_nz
                    yield from rec(i + 1, remaining - l * w, coeff_list, deriv_orders, next_last)
                    del coeff_list[-1]
                    deriv_orders[i] = 0

            dfc.release(tuple(deriv_orders), i)

        yield from rec(0, rem_budget, [], [0] * dim, 0)

    def cohr_pf(self, p: PF, cohr_dict: dict, n, **kwargs) -> PF:
        """
        Evaluates a PatternForm object under coherent state assertion for specified modes,
        expanded perturbatively up to order.

        :param p: Input PatternForm object.
        :param cohr_dict: Dict mapping mode identifier (or symbol) to coherent amplitude alpha.
        :param order: Perturbative expansion order cutoff.
        :return: Evaluated and expanded PatternForm.
        """
        n = sympify(n)
        pe_cohr = self.copy(cache=False)
        assump = dict(pe_cohr.assumptions)
        ncd = {}
        for k, v in cohr_dict.items():
            if hasattr(k, 'name'):
                k = k.name
            k = sympify(k)
            ncd[k] = sympify(v)
        cohr_dict = ncd
        reg_b = set()

        def _reg(bn):
            if bn in cohr_dict:
                if not bn in reg_b:
                    default_order = assump.get(bn, S.Zero)
                    assump[bn] = default_order if default_order < 0 else S.NegativeOne
                    reg_b.add(bn)
                return True
            return False

        ntl = defaultdict(list)
        for (bs, fs), tail in p.table.items():
            bs_rest = []
            rest_order = n
            for b in bs:
                bn = b.name
                if _reg(bn):
                    exponent = b.exps[0]
                    rest_order -= exponent * assump[bn]
                    if b.args[1]:
                        tail = NumShift(tail, -1, b).doit() * cohr_dict[bn] ** exponent
                    else:
                        tail *= conjugate(cohr_dict[bn]) ** exponent
                else:
                    bs_rest.append(b)
            ntl[(tuple(bs_rest), fs)].append(tail)
        pe_cohr.assumptions = assump

        new_table = {}
        new_npf = Add(*(t[0] for t in pe_cohr.cohr_avr(p.npf, cohr_dict, n, True, False, **kwargs)))
        for k, v in ntl.items():
            new_table[k] = Add(*(t[0] for t in pe_cohr.cohr_avr(Add(*v), cohr_dict, n, True, False, **kwargs)))
        clean_table = {k: v for k, v in new_table.items() if v != 0}
        return PF(clean_table, new_npf) if (clean_table or new_npf != 0) else PF()

    def handle_order_limits(self, handle, n_args=1):
        """
        Analyzes the perturbative order constraints of handle.

        Returns (limits, out_order):

        limits: tuple of the minimum acceptable order for each input argument.
        -oo (S.NegativeInfinity) for no lower bound, oo if no legal input exists.

        out_order: lower bound (infimum) of the leading order of the output
        over all legal inputs. -oo if unbounded below,
        oo if the output vanishes or no legal input exists.

        Consistent with leading_order and expand_expr, Function and
        Pow with non-numeric exponent are considered as zeroth order,
        and their arguments are required to be of nonnegative order.
        """
        if isinstance(handle, FunctionClass):
            nargs = getattr(handle.nargs, 'args', ())
            if len(nargs) == 1:
                n_args = max(n_args, int(nargs[0]))
        elif callable(handle):
            try:
                params = inspect.signature(handle).parameters.values()
            except (TypeError, ValueError):
                params = ()
            n_args = max(n_args, sum(
                p.default is inspect.Parameter.empty and p.kind in (
                    inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                for p in params))

        xs = tuple(Dummy() for _ in range(n_args))
        # order of xs[j] is represented by ws[j]; z stands for zeroth-order nodes
        ws = tuple(Dummy() for _ in range(n_args))
        z = Dummy()
        expr = sympify(handle(*xs) if callable(handle) else handle)

        pe = self.copy()
        pe.update({**dict(zip(xs, ws)), z: S.Zero})
        zero_w = dict.fromkeys(ws, S.Zero)

        def opaque(e):
            return isinstance(e, Function) or (e.is_Pow and not e.exp.is_number)

        def lin_orders(e):
            # yields (term, (B, A)) for each additive term of e, whose
            # leading order reads B + sum_j A[j]*ws[j]; (term, None) if unidentified
            for term in Add.make_args(e.replace(opaque, lambda _: z).expand()):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', FutureWarning)
                        o = pe.leading_order(term)[1]
                except (ValueError, TypeError):
                    yield term, None
                    continue
                if o is not S.Infinity:
                    yield term, (o.subs(zero_w), tuple(o.coeff(w) for w in ws))

        # order limits of the inputs: all arguments of opaque nodes must be nonnegative,
        # evaluated term by term with the other inputs kept at zeroth order
        bounds = [[] for _ in xs]
        for sub in (a for node in preorder_traversal(expr) if opaque(node) for a in node.args):
            for term, lin in lin_orders(sub):
                for j, x in enumerate(xs):
                    if not term.has(x):
                        continue
                    try:
                        B, A = lin[0], lin[1][j]
                        if A > 0:
                            b = -B / A
                            bounds[j].append(b if b.is_number else S.Zero)
                        elif A == 0 and B < 0:
                            bounds[j].append(S.Infinity)
                    except TypeError:
                        bounds[j].append(S.Zero)
        limits = tuple(max(b, default=S.NegativeInfinity) for b in bounds)

        # output order: infimum of each term's order over ws[j] >= limits[j]
        if S.Infinity in limits:
            return limits, S.Infinity
        out_order = S.Infinity
        B_min = S.Infinity
        for term, lin in lin_orders(expr):
            try:
                B, A = lin
                if B.is_number and B < B_min:
                    B_min = B
                o = B + Add(*(a * l if a > 0 else S.NegativeInfinity
                              for a, l in zip(A, limits) if a != 0))
                if not o.is_number:
                    raise TypeError
            except TypeError:
                warnings.warn('Unable to determine the output order', FutureWarning)
                return limits, S.NegativeInfinity
            if o < out_order:
                out_order = o

        # If out_order is -oo, check whether inputs approaching infinity yield a bounded result
        if out_order is S.NegativeInfinity:
            inf_xs = [x for x, lim in zip(xs, limits) if lim is S.NegativeInfinity]
            if inf_xs:
                curr = expr
                bounded = True
                for x in inf_xs:
                    try:
                        curr = limit(curr, x, S.Infinity)
                    except Exception:
                        bounded = False
                        break
                if bounded and not (curr.has(S.Infinity, S.NegativeInfinity, S.ComplexInfinity)
                                    or getattr(curr, 'is_infinite', False) is True):
                    if curr == S.Zero or curr == 0:
                        out_order = B_min if B_min.is_number else S.Zero
                    else:
                        try:
                            o_inf = self.leading_order(curr)[1]
                            if o_inf.is_number:
                                out_order = min(B_min, o_inf) if B_min.is_number else o_inf
                            elif B_min.is_number:
                                out_order = B_min
                        except Exception:
                            if B_min.is_number:
                                out_order = B_min

        return limits, out_order


