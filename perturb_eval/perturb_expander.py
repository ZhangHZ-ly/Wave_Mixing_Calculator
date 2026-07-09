from sympy.core.containers import Tuple, Dict
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import Function, Derivative, diff, Subs, UndefinedFunction
from sympy.core.relational import Relational
from sympy.core.numbers import Integer
from sympy.concrete.summations import Sum
from sympy.integrals.integrals import Integral
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from op_patterns import *
from op_patterns.boson_pattern import fac
from .pattern_form import PFTableProcessor
from sortedcontainers import SortedDict
from functools import lru_cache, cached_property
import warnings

__all__ = ['PerturbExpander']

warnings.simplefilter("once", FutureWarning)
class DerivCache:
    """
    Cache for diff(f) per variable.
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
        Calculates and caches the partial derivatives of f,
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
            expr = diff(self.get((*orders[:i], n-1, *orders[i+1:]), i),
                              self.Ns[i], **kwargs)
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

def commute(a, b, **hints):
    if a is b or a.is_commutative or b.is_commutative:
        return True
    if hints.get('independent', False):
        if a in commute_ind_cache and b in commute_ind_cache[a]:
            return commute_ind_cache[a][b]
        elif b in commute_ind_cache and a in commute_ind_cache[b]:
            return commute_ind_cache[b][a]
        elif a not in commute_ind_cache:
            commute_ind_cache[a] = dict()
        if check_num(a, check_num_cache) and check_num(b, check_num_cache):
            result = True
        else:
            ba, fa, na, _, _ = extract_modes_cached(a)
            bb, fb, nb, _, _ = extract_modes_cached(b)
            if fa or na or fb or nb:
                pass
            elif not (ba & bb) or Commutator(a, b).doit(independent=True) == 0:
                result = True
            else:
                result = False
        commute_ind_cache[a][b] = result
        return result
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
    else:
        result = False
    commute_dep_cache[a][b] = result
    return result

class PolyPermutation(Expr):
    def __new__(cls, arg):
        if not arg:
            return S.One
        return Expr.__new__(cls, Dict(arg))

    @cached_property
    def total_order(self):
        return sum(self.args[0].values())
    
    def sep_cnc(self, **hints):
        comm = []
        ts = dict(self.args[0])
        for a in ts:
            if all(commute(a, b, **hints) for b in ts):
                comm.append(a)
        coeffs = []
        for a in comm:
            m = ts[a]
            del ts[a]
            coeffs.append(a**m / fac(m))
        return Mul(*coeffs, PolyPermutation(ts))

    def doit(self, **hints):
        comm = []
        ts = dict(self.args[0])
        for a in ts:
            if all(commute(a, b, **hints) for b in ts):
                comm.append(a)
        coeffs = []
        for a in comm:
            m = ts[a]
            del ts[a]
            if hints.get('deep', True):
                a = a.doit(**hints)
            coeffs.append(a**m / fac(m))
        if not ts:
            return Mul(*coeffs)
        from sympy.utilities.iterables import multiset_permutations as perm
        if hints.get('deep', True):
            terms = {term.doit(**hints): m for term, m in self.args[0].items()}
        else:
            terms = self.args[0]
        l = sum(([term] * m for term, m in terms.items()), [])
        return Mul(*coeffs, Add(*(Mul(*p) for p in perm(l))) / fac(self.total_order))

def coeffs(ao, prefix, l, **kwargs):
    collection = [dict() for _ in range(l)]
    for (i, _, ta), o in zip(ao, prefix):
        collection[i][ta] = o
    return tuple(PolyPermutation(pt).sep_cnc(**kwargs) for pt in collection)

class PerturbExpander:
    """
    Includes methods that can expand an expression
    composed of symbols with self-defined orders

    order of 0: ~ 1
    order of 1: << 1, adds one order of perturbation
    order of -1: >> 1, cancels one order of perturbation
    order of oo (infinity): pure zero
    """
    exclude = frozenset({Add, Mul, Pow,
                        Integral, Sum, Derivative, Subs,
                        Commutator, AntiCommutator, UidCommutator})
    __slots__ = ('assumptions', 'rc', 'rs', 'rp', '_lo_cache', '_subs_cache')

    def __init__(self, assumptions):
        self.assumptions = Dict(assumptions)
        self.rc = {FermionPat}
        self.rs = {Symbol}
        self.rp = {BosonPat: ('name', lambda p: Add(*p.exps))}
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
        if isinstance(expr, tuple(self.rc)):
            return expr, S.Zero
        if isinstance(expr, tuple(self.rs)):
            if expr in self.assumptions:
                return expr, sympify(self.assumptions[expr])
            if expr in self._subs_cache:
                return self.leading_order(self._subs_cache[expr])
            return expr, S.Zero
        if isinstance(expr, tuple(self.rp)):
            id, fh = self.rp[next(t for t in self.rp if isinstance(expr, t))]
            eid = getattr(expr, id)
            if (eid in self.assumptions and self.assumptions[eid] != 0):
                return expr, fh(expr) * self.assumptions[eid]
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
            sd = SortedDict()
            for arg, o in zip(expr.args, lo):
                ni = n - n0 + o
                for ta, oa in self.expand_expr(arg, ni, **kwargs):
                    od = oa - o
                    sd[od] = sd.get(od, S.Zero) + ta
                ts.append(sd.copy())
                sd.clear()
            dim = len(ts)
            def rec(i, remaining, prefix):
                if i == dim:
                    yield Mul(*prefix), n - remaining
                    return
                d = ts[i]
                for k in d:
                    if k > remaining:
                        break
                    prefix.append(d[k])
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
            sd = SortedDict()
            n_new = n - m * lo
            for t, o in self.expand_expr(expr.base, n_new):
                if o == lo:
                    continue
                o -= lo
                sd[o] = sd.get(o, 0) + t
            if not sd:
                yield lt**m, lo * m
                return
            dim = len(sd)
            def rec(i, remaining, prefix):
                if i == dim:
                    pp = PolyPermutation(dict(zip(sd.values(), prefix)))
                    if m.is_Integer and m < pp.total_order:
                        yield from ()
                    yield (Mul(gperm(m, pp.total_order), lt**(m-pp.total_order),
                            pp.sep_cnc(**kwargs)), n - remaining)
                    return
                w = sd.iloc[i]
                max_count = remaining // w
                for x in range(max_count + 1):
                    prefix.append(x)
                    yield from rec(i+1, remaining - x*w, prefix)
                    del prefix[-1]
            yield from rec(0, n_new, [])
            extract_modes_cache.clear()
            check_num_cache.clear()
            commute_dep_cache.clear()
            commute_ind_cache.clear()
        else:
            if isinstance(expr, Derivative):
                args = expr.expr.args if isinstance(expr.expr, Function) else ()
                if any(isinstance(v, Tuple) and v[0] not in args or
                       v not in args for v in expr.variable_count):
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
                sd = SortedDict()
                for i, arg in enumerate(args):
                    if isinstance(arg, Symbol):
                        syms[arg] = i
                    for ta, oa in self.expand_expr(arg, n):
                        if (oa < 0) == True:
                            raise ValueError('Expect nonnegative-order inputs for functions')
                        if oa.is_number:
                            sd[oa] = sd.get(oa, S.Zero) + ta
                    if S.Zero in sd:
                        a0.append(sd[S.Zero])
                        del sd[S.Zero]
                    else:
                        a0.append(S.Zero)
                    for oa, ta in sd.items():
                        ao.append((i, oa, ta))
                    sd.clear()
                dim = len(ao)
                d0 = tuple(Dummy() for _ in a0)
                ini_order = [0] * len(args)
                for v in expr.variable_count:
                    if isinstance(v, Tuple):
                        ini_order[syms[v[0]]] += v[1]
                def term(remaining, prefix, deriv_orders, *_):
                    cs = coeffs(ao, prefix, len(a0), **kwargs)
                    nd = Derivative(expr.expr.func(*d0),
                                    *((d0, p+q) for d0, p, q in zip(d0, ini_order, deriv_orders)))
                    return Mul(*cs, nd).subs(dict(zip(d0, pt))), n-remaining
            elif expr.is_Pow or isinstance(expr, Function):
                if n < 0:
                    yield from ()
                    return
                a0 = []
                ao = []
                min_i = [0]
                sd = SortedDict()
                for i, arg in enumerate(expr.args):
                    for ta, oa in self.expand_expr(arg, n):
                        if (oa < 0) == True:
                            raise ValueError('Expect nonnegative-order inputs for functions')
                        if oa.is_number:
                            sd[oa] = sd.get(oa, S.Zero) + ta
                    if S.Zero in sd:
                        a0.append(sd[S.Zero])
                        del sd[S.Zero]
                    else:
                        a0.append(S.Zero)
                    min_i.append(len(sd)+min_i[-1])
                    for oa, ta in sd.items():
                        ao.append((i, oa, ta))
                    sd.clear()
                min_i = dict(zip(min_i, range(len(a0))))
                d0 = tuple(Dummy() for _ in a0)
                pt = [i[2] for i in ao]
                if not ao:
                    yield expr.func(*a0), S.Zero
                    return
                dim = len(ao)
                dfc = DerivCache(expr.func(*d0), d0)
                if isinstance(expr.func, UndefinedFunction):
                    def term(remaining, prefix, deriv_orders, *_):
                        cs = coeffs(ao, prefix, len(a0), **kwargs)
                        nd = Derivative(expr.func(*d0), *((d0, q) for d0, q in zip(d0, deriv_orders)))
                        return Mul(*cs, nd.subs(dict(zip(d0, pt)))), n-remaining
                else:
                    def term(remaining, prefix, deriv_orders, current_sum, last_nz):
                        cs = coeffs(ao, prefix, len(a0), **kwargs)
                        nf = dfc.get(tuple(deriv_orders), last_nz, **kwargs).subs(dict(zip(d0, a0)))
                        if last_nz in min_i and ao[last_nz][1] < remaining:
                            dfc.release(deriv_orders, min_i[last_nz])
                        if nf is S.Zero:
                            return
                        return Mul(*cs, nf), current_sum
            else:
                ne = expr.doit(**kwargs)
                if ne == expr:
                    yield from self.expand_expr(ne, n, **kwargs)
                    return
                warnings.warn('Failed to deal with the expression', FutureWarning)
            def rec(i, remaining, prefix, deriv_orders, current_sum, last_nz):
                if i == dim:
                    result = term(remaining, prefix, deriv_orders, current_sum, last_nz)
                    if result is None:
                        yield from()
                    else:
                        yield result
                    return
                curr, w, _ = ao[i]
                max_count = remaining // w
                for x in range(max_count + 1):
                    prefix.append(x)
                    deriv_orders[curr] += x
                    next_last = curr if x != 0 else last_nz
                    yield from rec(
                        i + 1,
                        remaining - x * w,
                        prefix,
                        deriv_orders,
                        current_sum + x * w,
                        next_last
                    )
                    del prefix[-1]
                    deriv_orders[curr] -= x
            yield from rec(0, n, [], [0]*len(a0), 0, 0)
            extract_modes_cache.clear()
            check_num_cache.clear()
            commute_dep_cache.clear()
            commute_ind_cache.clear()

    def expand_pf(self, pf: PFTableProcessor, n, **kwargs):
        """
        Expand pattern-form table-like expressions upto the given order n
        """
        new_t = dict()
        for (bs, fs), t in pf.table.items():
            nb = Add(*(self.leading_order(b)[1] for b in bs))
            new_t[(bs, fs)] = Add(*(e[0] for e in self.expand_expr(t, n-nb, **kwargs)))
        return PFTableProcessor(new_t, Add(*(e[0] for e in self.expand_expr(pf.npf, n, **kwargs))))
