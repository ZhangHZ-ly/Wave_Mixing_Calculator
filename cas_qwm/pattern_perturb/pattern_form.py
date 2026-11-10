from __future__ import annotations
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import Function, Subs, Derivative
from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.core.containers import Tuple
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.core.function import diff as sympy_diff
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from ..op_pattern.boson_pattern import BosonPat, BosonNum, NumShift
from ..op_pattern.fermion_pattern import FermionPat, m1_base, UidCommutator
from ..op_pattern.tools import mul_pattern, extract_modes, check_num, redo_mul
from sortedcontainers import SortedDict
import warnings

__all__ = ['is_pattern_form', 'pf', 'PF', 'pf_add', 'PF0', 'PF1', 'BSC', 'FSC']

_k_cache = dict()
def key(a):
    if isinstance(a, (BosonPat, FermionPat)):
        na = a.name
        if na is None:
            return ()
        elif na in _k_cache:
            return _k_cache[na]
        else:
            k = na.sort_key()
            _k_cache[na] = k
            return k
    return None

_rem_to_del = dict()
def _rem_cache_update(cache, new):
    if cache is new:
        return
    for b in new:
        if b in cache:
            if b in _rem_to_del:
                for n in _rem_to_del[b]:
                    cache[b].pop(n, None)
                del _rem_to_del[b]
            cache[b].update(new[b])
        else:
            cache[b] = dict(new[b]) if isinstance(new[b], dict) else new[b]
    for b in _rem_to_del:
        for n in _rem_to_del[b]:
            cache[b].pop(n, None)
    _rem_to_del.clear()

class BSC:
    """
    Sort context for a tuple/list of BosonPats or number-operator functions
    """
    __slots__ = ('pats', 'result', 'tail', 'remainder',
                 '_rem_cache', 'pat_sorted', 'tail_pat', 'remainder_pat')

    def __init__(self, pats):
        if not pats:
            self.result = self.pat_sorted = tuple(pats)
            self.tail = self.tail_pat = S.One
            self.remainder = self.remainder_pat = S.Zero
        elif len(pats) == 1:
            if isinstance(pats[0], BosonPat):
                self.pat_sorted = tuple(pats)
                self.tail_pat = S.One
            else:
                self.result = self.pat_sorted = ()
                self.tail = self.tail_pat = pats[0]
                self._rem_cache = dict()
            self.remainder = self.remainder_pat = S.Zero
        else:
            self.pats = tuple(pats)
            self._rem_cache = dict()

    def expr(self) -> Expr:
        if hasattr(self, 'pats'):
            return Mul(*self.pats[0], *self.pats[1])
        return Mul(*self.result[0], self.tail)

    def sort_ind_pat(self):
        if not hasattr(self, 'pat_sorted'):
            sd = SortedDict()
            tail = []
            for a in reversed(self.pats):
                k = key(a)
                if k in sd:
                    new = mul_pattern(a, sd[k])[0]
                    sd[k] = new
                elif k is not None:
                    sd[k] = a
                if k is None:
                    tail.append(NumShift(a, 1, sd.values()).doit(deep=False))
            self.pat_sorted = tuple(sd.values())
            self.tail_pat = Mul(*reversed(tail))
        return self.pat_sorted, self.tail_pat
    def result_ind(self):
        if not hasattr(self, 'result'):
            if not hasattr(self, 'pat_sorted'):
                self.sort_ind_pat()
            result, new_tail = [], []
            for p in self.pat_sorted:
                a, t = p.num_repr()
                if a is not S.One:
                    result.append(a)
                if t is not S.One:
                    new_tail.append(t)
            self.result = tuple(result)
            self.tail = Mul(*new_tail, self.tail_pat)
        return self.result, self.tail
    def merge_ind(self, R):
        L = self.result_ind()[0]
        l, r = len(L), len(R)
        i = j = 0
        new = []
        tail = [NumShift(self.tail, 1, R).doit(deep=False)]
        while i < l and j < r:
            if key(L[i]) < key(R[j]):
                new.append(L[i])
                i += 1
            elif key(L[i]) > key(R[j]):
                new.append(R[j])
                j += 1
            else:
                a, n = mul_pattern(L[i], R[j])[0].num_repr()
                if a is not S.One:
                    new.append(a)
                if n is not S.One:
                    tail.append(n)
                i += 1
                j += 1
        return (*new, *L[i:], *R[j:]), Mul(*reversed(tail))

    def _b_rem(self, e_N: Expr, b: BosonOp) -> Expr:
        """
        Calculates the remainder after reordering
        e_N\\*b into b\\*NumShift(e_N, b).

        The global _rem_to_del may change,
        so please do not forget the corresponding cache management.
        """
        if e_N.is_commutative:
            return S.Zero
        if e_N in self._rem_cache.get(b, ()):
            if b in _rem_to_del:
                _rem_to_del[b].add(e_N)
            else:
                _rem_to_del[b] = {e_N}
            return self._rem_cache[b][e_N]
        if isinstance(e_N, NumShift):
            f_rem = self._b_rem(e_N.function, b)
            return NumShift(f_rem, e_N.direction, e_N.shift, scan_dict=False)
        if isinstance(e_N, ExprWithLimits):
            return e_N.func(self._b_rem(e_N.function, b), *e_N.limits)
        if isinstance(e_N, Add):
            return Add(*[self._b_rem(arg, b) for arg in e_N.args])
        if isinstance(e_N, Mul):
            prev, rem = 0, []
            for i, arg in enumerate(e_N.args):
                modes = extract_modes(arg)[0]
                if b in modes:
                    if i != prev:
                        rem.append(Mul(*e_N.args[:prev],
                                    Commutator(Mul(*e_N.args[prev:i]), b),
                                    *e_N.args[i+1:]))
                    prev = i + 1
                    new = self._b_rem(arg, b)
                    if new is not S.Zero:
                        rem.append(Mul(*e_N.args[:i], new, *e_N.args[prev:]))
            if prev != len(e_N.args):
                rem.append(Mul(*e_N.args[:prev],
                           Commutator(Mul(*e_N.args[prev:]), b)))
            return Add(*rem)
        if isinstance(e_N, (BosonNum, Pow, Function)):
            modes = extract_modes(e_N)[0]
            if b in modes:
                return S.Zero
        return Commutator(e_N, b)
    def merge_rem_pat(self, R, reserve_unshifted=False):
        """
        Merges a new tuple/list of sorted BosonPat
        to the right side of the sorted result
        and calculates the total remainder.
        
        The global _rem_to_del may change,
        so please do not forget the corresponding cache management.

        The parameter 'reserve_unshifted' is
        for avoiding removal of cache about the tail itself,
        which is essential for the treatment of integer powers.
        """
        L, T, rd = self.sort_rem_pat()
        new = []
        rem = []
        if rd is not S.Zero:
            rem.append(Mul(rd, *R))
        if T is S.One:
            tail = []
        elif T.is_commutative:
            tail = [T]
        else:
            rrt = []
            tail = [NumShift(T, 1, R).doit(deep=False)]
            for n, b in enumerate(R):
                bb = BosonOp(b.name)
                rn = self._b_rem(T, bb)
                if bb in self._rem_cache:
                    self._rem_cache[bb].update({tail[0]: NumShift(rn, 1, R).doit(deep=False)})
                    if bb in _rem_to_del:
                        _rem_to_del[bb].discard(tail[0])
                        if reserve_unshifted:
                            _rem_to_del[bb].discard(T)
                if rn is not S.Zero:
                    rrt.append(Mul(*R[:n], rn.xreplace({bb: b}), *R[n+1:]))
            if rrt:
                rem.append(Mul(*L, Add(*rrt), evaluate=(len(rrt)==1)))
            del rrt
        l, r = len(L), len(R)
        i = j = j_prev = 0
        while i < l and j < r:
            if key(L[i]) < key(R[j]):
                if j_prev != j:
                    c = Commutator(Mul(*L[i:], evaluate=False),
                                    Mul(*R[j_prev:j], evaluate=False))
                    rem.append(Mul(*new[:j_prev-j], c, *R[j:], *tail))
                    j_prev =j
                new.append(L[i])
                i += 1
            elif key(L[i]) > key(R[j]):
                new.append(R[j])
                j += 1
            else:
                np = mul_pattern(L[i], R[j])[0]
                j += 1
                c = Commutator(Mul(*L[i+1:], evaluate=False),
                               Mul(*R[j_prev:j], evaluate=False))
                rem.append(Mul(*new, L[i], c, *R[j:], *tail))
                i += 1
                j_prev = j
                new.append(np)
        if j_prev != j:
            c = Commutator(Mul(*L[i:], evaluate=False),
                            Mul(*R[j_prev:j], evaluate=False))
            rem.append(Mul(*new[:j_prev-j], c, *R[j:], *tail))
        return (*new, *L[i:], *R[j:]), Mul(*reversed(tail)), Add(*rem)
    def sort_rem_pat(self):
        if not hasattr(self, 'remainder_pat'):
            mid = len(self.pats)//2
            L = BSC(tuple(self.pats[:mid]))
            R = BSC(tuple(self.pats[mid:]))
            lsp, ltp, lrp = L.sort_rem_pat()
            rsp, rtp, rrp = R.sort_rem_pat()
            self.pat_sorted, t, rem = L.merge_rem_pat(rsp)
            _rem_cache_update(self._rem_cache, getattr(R, '_rem_cache', ()))
            self.tail_pat = t * rtp
            self.remainder_pat = Add(Mul(rem, rtp),
                            Mul(Mul(*lsp, ltp) + lrp, rrp))
        return self.pat_sorted, self.tail_pat, self.remainder_pat
    def result_rem(self):
        if not hasattr(self, 'remainder'):
            ps, tp, rp = self.sort_rem_pat()
            new = []
            tail = []
            rem = []
            for i, p in enumerate(ps):
                a, n = p.num_repr()
                if a is not S.One:
                    new.append(a)
                if n is not S.One:
                    rem.append(Mul(*new,
                                   Commutator(n, Mul(*ps[i+1:], evaluate=False)),
                                   *tail))
                    tail.append(n)
            self.result = tuple(new)
            self.tail = Mul(*tail, tp)
            self.remainder = Mul(Add(*rem), tp) + rp
        elif not hasattr(self, 'result'):
            return *self.result_ind(), self.remainder
        return self.result, self.tail, self.remainder

class FSC:
    """
    Sort context for a tuple/list of FermionPats
    """
    __slots__ = ('pats', 'result', 'tail', 'remainder')

    def __init__(self, pats):
        if not pats:
            if pats is None:
                self.result = None
                self.tail = S.Zero
            else:
                self.result = tuple(pats)
                self.tail = S.One
            self.remainder = S.Zero
        else:
            if len(pats) == 1:
                self.result = tuple(pats)
                self.tail = S.One
                self.remainder = S.Zero
            else:
                self.pats = tuple(pats)
                self._rem_cache = dict()

    def expr(self) -> Expr:
        if hasattr(self, 'pats'):
            return Mul(*self.pats[0], *self.pats[1])
        if not self.result:
            return self.tail
        return Mul(*self.result[0], self.tail)

    def result_ind(self):
        if not hasattr(self, 'result'):
            mid = len(self.pats)//2
            L = FSC(tuple(self.pats[:mid]))
            R = FSC(tuple(self.pats[mid:]))
            ls, _ = L.result_ind()
            rs, rt = R.result_ind()
            if ls is None:
                self.result = None
                self.tail = S.Zero
                if rs is None:
                    self.remainder = S.Zero
            elif R.result is None:
                self.result = None
                self.tail = S.Zero
            else:
                self.result, t = L.merge_ind(rs)
                self.tail = Pow(S.NegativeOne, m1_base(t) + m1_base(rt))
        return self.result, self.tail
    def merge_ind(self, R):
        L = self.result_ind()[0]
        if L is None or R is None:
            return None, S.Zero
        l, r = len(L), len(R)
        i = j = j_prev = 0
        new = []
        tail_args = [m1_base(self.tail)]
        l_arg = Add(*(m1_base(p.comm_symm) for p in L))
        def update():
            nonlocal l_arg
            l_arg -= m1_base(L[i].comm_symm)
            if j_prev != j:
                tail_args.append(l_arg * Add(*(m1_base(p.comm_symm)
                                             for p in R[j_prev:j])))
        while i < l and j < r:
            if key(L[i]) < key(R[j]):
                update()
                new.append(L[i])
                j_prev = j
                i += 1
            elif key(L[i]) > key(R[j]):
                new.append(R[j])
                j += 1
            else:
                fp = mul_pattern(L[i], R[j])[0]
                if fp is S.Zero:
                    return None, S.Zero
                j += 1
                update()
                new.append(fp)
                i += 1
                j_prev = j
        if j_prev != j:
            if i < l:
                l_arg -= m1_base(L[i].comm_symm)
            tail_args.append(l_arg * Add(*(m1_base(p.comm_symm)
                                         for p in R[j_prev:j])))
        return (*new, *L[i:], *R[j:]), Pow(S.NegativeOne, Add(*tail_args))

    def merge_rem(self, R):
        L, T, r = self.result_rem()
        if L is None or R is None:
            return None, S.Zero, S.Zero
        new = []
        rem = []
        if r is not S.Zero:
            rem.append(Mul(r, *R))
        if T is S.One:
            tail = []
        else:
            tail = [T]
        l, r = len(L), len(R)
        i = j = j_prev = 0
        def record():
            c = UidCommutator(Mul(*L[i:], evaluate=False),
                            Mul(*R[j_prev:j], evaluate=False))
            rem.append(Mul(*tail, *new[:j_prev-j], c,
                        *R[j:]))
            if isinstance(c, AntiCommutator):
                tail.append(S.NegativeOne)
            elif isinstance(c, UidCommutator):
                tail.append(c.comm_sign)
        while i < l and j < r:
            if key(L[i]) < key(R[j]):
                if j_prev != j:
                    record()
                    j_prev =j
                new.append(L[i])
                i += 1
            elif key(L[i]) > key(R[j]):
                new.append(R[j])
                j += 1
            else:
                prod = mul_pattern(L[i], R[j])[0]
                j += 1
                c = UidCommutator(Mul(*L[i+1:], evaluate=False),
                                Mul(*R[j_prev:j], evaluate=False))
                rem.append(Mul(*tail, L[i], *new, c, *R[j:]))
                if isinstance(c, AntiCommutator):
                    tail.append(S.NegativeOne)
                elif isinstance(c, UidCommutator):
                    tail.append(c.comm_sign)
                if prod == 0:
                    return None, S.Zero, Add(*rem)
                new.append(prod)
                i += 1
                j_prev = j
        if j_prev != j:
            record()
        return (*new, *L[i:], *R[j:]), Mul(*reversed(tail)), Add(*rem)
    def result_rem(self):
        if not hasattr(self, 'remainder'):
            mid = len(self.pats)//2
            L = FSC(tuple(self.pats[:mid]))
            R = FSC(tuple(self.pats[mid:]))
            ls, lt, lr = L.result_rem()
            rs, rt, rr = R.result_rem()
            if ls is None:
                self.result = None
                self.tail = S.Zero
                if rs is None:
                    self.remainder = S.Zero
                else:
                    self.remainder = Mul(lr, *rs, R.tail)
            elif R.result is None:
                self.result = None
                self.tail = S.Zero
                self.remainder = Mul(*ls, lt, rr)
            else:
                self.result, t, rem = L.merge_rem(rs)
                self.tail = Pow(S.NegativeOne, m1_base(t) + m1_base(rt))
                self.remainder = Add(Mul(rem, rt),
                                Mul(Mul(*ls, lt) + lr, rr))
        return self.result, self.tail, self.remainder

def is_pattern_form(*exprs: Expr) -> bool:
    """
    Checks whether the target expression is represented in pat-form.
    
    The pat-form represents a quantum expression into a sum of terms,
    each of which is a product of a numeric coefficient,
    some powers of canonical operators and functions of number operators,
    where the powers of canonical operators is always left-multiplied to the functions,
    and each of the canonical operators (the base of powers) is in a unique mode.
    """
    for expr in exprs:
        if (isinstance(expr, BosonPat) and len(expr.exps) == 1
            or isinstance(expr, FermionPat)
            or (expr.is_Add and is_pattern_form(*expr.args))
            or check_num(expr)):
            continue
        if expr.is_Mul:
            bs, fs, o = set(), set(), True
            for arg in expr.args_cnc()[1]:
                if o:
                    if isinstance(arg, BosonPat):
                        if len(arg.exps) == 1 and arg.name not in bs:
                            bs.add(arg.name)
                            continue
                    elif isinstance(arg, FermionPat):
                        if arg.name not in fs:
                            fs.add(arg.name)
                            continue
                if check_num(arg):
                    o = False
                    continue  # Continuing here means that the current Mul argument has passed the check
                return False             
            continue  # Continuing here means that the current Mul object has passed the check
        return False
    return True

_vpk = ((), ())

def make_contexts(pf: PF):
    return pf.make_context()

class PF:
    """
    Final representation of pattern form
    """
    __slots__ = (
        'table', 'npf', '_rem_cache', 'rem'
    )

    def __init__(
        self,
        table: dict = None,
        npf=S.Zero,
        cache=None,
        rem=S.Zero,
    ):
        self.table = dict() if table is None else table
        self.npf = sympify(npf)
        self._rem_cache = dict() if cache is None else cache
        self.rem = S.Zero if rem is None else sympify(rem)

    def __repr__(self):
        return f"PatternForm({self.table}, {self.npf})"

    def __eq__(self, value):
        if isinstance(value, PF):
            return self.table == value.table and self.npf == value.npf
        return False
    
    def __bool__(self) -> bool:
        return bool(self.table or self.npf != 0 or self.rem != 0)

    def expr(self, with_npf=True) -> Expr:
        if with_npf:
            return Add(*(Mul(*a[0], n, *a[1]) for a, n in self.table.items()), self.npf)
        return Add(*(Mul(*a[0], n, *a[1]) for a, n in self.table.items()))

    def result_ind(self):
        return self

    def result_rem(self):
        return self, getattr(self, 'rem', S.Zero)

    @classmethod
    def from_val(cls, val) -> PF:
        if isinstance(val, PF):
            return val
        if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], PF):
            pf_obj = val[0]
            pf_obj.rem = sympify(val[1])
            return pf_obj
        expr = sympify(val)
        if getattr(expr, 'is_commutative', True) or not expr.has(BosonOp, BosonPat, FermionOp, FermionPat):
            return cls({_vpk: expr}, rem=S.Zero)
        pf_res = pattern_form(expr, as_dict=None)
        if isinstance(pf_res, tuple):
            pf_obj, r = pf_res
            pf_obj.rem = r
            return pf_obj
        return pf_res

    def make_context(self):
        rem = getattr(self, 'rem', S.Zero)
        for (bs, fs), n in self.table.items():
            bsc = object.__new__(BSC)
            bsc.result = bsc.pat_sorted = bs
            bsc.tail = bsc.tail_pat = n
            bsc.remainder = bsc.remainder_pat = rem
            bsc._rem_cache = self._rem_cache
            fsc = object.__new__(FSC)
            fsc.result = fs
            fsc.tail = S.One
            fsc.remainder = S.Zero
            yield bsc, fsc

    make_contexts = make_context

    def __add__(self, other):
        if not isinstance(other, PF):
            other = PF.from_val(other)
        return pf_add(self, other)

    def __radd__(self, other):
        if other == 0 or other is S.Zero:
            return self
        return PF.from_val(other).__add__(self)

    def __iadd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return PF(
            {k: -v for k, v in self.table.items()}, -self.npf, dict(self._rem_cache),
            rem=-self.rem
        )

    def __pos__(self):
        return self

    def __sub__(self, other):
        if not isinstance(other, PF):
            other = PF.from_val(other)
        return pf_add(self, -other)

    def __rsub__(self, other):
        return PF.from_val(other).__sub__(self)

    def __isub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        if isinstance(other, (int, Integer)):
            if other == 0:
                return PF0
            if other == 1:
                return self
            return PF(
                {k: v * other for k, v in self.table.items()},
                self.npf * other,
                dict(self._rem_cache)
            )
        if isinstance(other, Expr) and getattr(other, 'is_commutative', False):
            if other == 0 or other is S.Zero:
                return PF0
            if other == 1 or other is S.One:
                return self
            return PF(
                {k: v * other for k, v in self.table.items()},
                self.npf * other,
                dict(self._rem_cache)
            )
        if not isinstance(other, PF):
            other = PF.from_val(other)
        sf = dict()
        for bsc, fsc in self.make_context():
            for (bs, fs), t in other.table.items():
                bp, bt = bsc.merge_ind(bs)
                fp, ft = fsc.merge_ind(fs)
                if fp is not None:
                    a = (bp, fp)
                    sf[a] = sf.get(a, 0) + Mul(ft, bt, t)
        cache = dict(self._rem_cache)
        _rem_cache_update(cache, other._rem_cache)
        return PF(
            sf,
            Add(Mul(self.expr(), other.npf), Mul(self.npf, other.expr(False))),
            cache
        )

    def __rmul__(self, other):
        if isinstance(other, (int, Integer)):
            return self.__mul__(other)
        if isinstance(other, Expr) and getattr(other, 'is_commutative', False):
            return self.__mul__(other)
        return PF.from_val(other) * self

    def __imul__(self, other):
        return self.__mul__(other)

    def _mul_rem(self, other: PF, reserve_unshifted=False):
        sf, rem = dict(), []
        for bsc, fsc in self.make_context():
            for (bs, fs), t in other.table.items():
                nbsc = object.__new__(BSC)
                (nbsc.pat_sorted,
                 nbsc.tail_pat,
                 nbsc.remainder_pat) = bsc.merge_rem_pat(bs, reserve_unshifted)
                bp, bt, br = nbsc.result_rem()
                fp, ft, fr = fsc.merge_rem(fs)
                if fp is not None:
                    a = (bp, fp)
                    sf[a] = sf.get(a, 0) + Mul(ft, bt, t)
                    rem.append(Mul(br, t, *fs))
                rem.append(Mul(Mul(*bp, bt) + br, t, fr))
        cache = dict(self._rem_cache)
        _rem_cache_update(cache, other._rem_cache)
        return PF(
            sf,
            Add(Mul(self.expr(), other.npf), Mul(self.npf, other.expr(False))),
            cache
        ), Add(*rem)

    def __pow__(self, power):
        if isinstance(power, PF):
            if power.npf == S.Zero and set(power.table.keys()).issubset({_vpk}):
                power = power.table.get(_vpk, S.Zero)
            else:
                return PF(npf=Pow(self.expr(), power.expr()))
        p_sym = sympify(power)
        if self.npf == S.Zero and set(self.table.keys()).issubset({_vpk}):
            return PF({_vpk: self.table.get(_vpk, S.Zero) ** p_sym})
        if getattr(p_sym, 'is_Integer', False) and p_sym >= 0:
            exp = int(p_sym)
            if exp == 0:
                return PF1
            if exp == 1:
                return self
            if len(self.table) == 1 and self.npf == 0:
                (bs, fs) = next(iter(self.table))
                t = self.table[(bs, fs)]
                if t == 1 or t is S.One:
                    bps = tuple(b**exp for b in bs)
                    fps = tuple(f**exp for f in fs)
                    return PF({(bps, fps): S.One}, cache=dict(self._rem_cache))
            sf_k = self ** (exp >> 1)
            sf_2k = sf_k * sf_k
            if exp & 1 == 0:
                return sf_2k
            return self * sf_2k
        return PF(npf=Pow(self.expr(), p_sym))

    def _pow_rem(self, exp: int | Integer, base_rd=S.Zero):
        if exp == 1:
            return self, base_rd
        if exp < 0:
            raise ValueError('Exponent must be nonnegative')
        if exp == 0:
            return PF1, S.Zero
        sf_k, rd_k = self._pow_rem(exp >> 1, base_rd)
        sf_2k, r_sf2k = sf_k._mul_rem(sf_k, True)
        rd_2k = Add(r_sf2k, Mul(sf_k.expr(), rd_k),
                    Mul(rd_k, sf_k.expr()), Mul(rd_k, rd_k))
        if exp & 1 == 0:
            return sf_2k, rd_2k
        sf, r_sf = self._mul_rem(sf_2k, True)
        rd = Add(r_sf, Mul(self.expr(), rd_2k),
                 Mul(base_rd, sf_2k.expr() + rd_2k))
        return sf, rd

    def __truediv__(self, other):
        if isinstance(other, PF):
            if other.npf == S.Zero and set(other.table.keys()).issubset({_vpk}):
                val = other.table.get(_vpk, S.One)
                return self * (val ** S.NegativeOne)
            raise ValueError("Division is only supported by scalar or number-operator expressions.")
        val = sympify(other)
        if getattr(val, 'is_commutative', True) or not val.has(BosonOp, BosonPat, FermionOp, FermionPat):
            return self * (val ** S.NegativeOne)
        raise ValueError("Division is only supported by scalar expressions.")

    def __rtruediv__(self, other):
        if self.npf == S.Zero and set(self.table.keys()).issubset({_vpk}):
            base_expr = self.table.get(_vpk, S.One)
            return PF.from_val(base_expr ** S.NegativeOne) * other
        raise ValueError("Division by non-scalar operator expression is not supported.")

    def __invert__(self):
        return self.adjoint()

    def _eval_adjoint(self):
        return self.adjoint()

    def adjoint(self, rem=False) -> PF:
        sf = dict()
        if rem:
            new_rem = dict()
            d_rem = dict()
            rem_list = []
            for (bs, fs), t in self.table.items():
                td = NumShift(t, -1, bs).doit(deep=False)._eval_adjoint()
                bd = tuple(b._eval_adjoint() for b in reversed(bs))
                bsc = BSC(bd)
                bsc._rem_cache = self._rem_cache
                br = bsc.sort_rem_pat()[2]
                r = []
                for i, b in enumerate(bd):
                    bb = BosonOp(b.name)
                    rb = bsc._b_rem(t, bb)
                    rd = NumShift(rb, -1, bs).doit(deep=False)
                    new_rem[bb] = {t: rb}
                    r.append(Mul(*bd[i:], rd.xreplace({bb: bs[-(i+1)]})._eval_adjoint(), *bd[:i+1]))
                    d_rem[bb] = {td: rd._eval_adjoint().xreplace({bb._eval_adjoint(): bb})}
                fp, ft, fr = FSC(tuple(f._eval_adjoint()
                                                      for f in reversed(fs))).result_rem()
                rem_list.append(Mul(ft, Add(*r, -br*td), *fp) + Mul(*bd, td, fr))
                sf[(bd, fp)] = ft * td
            _rem_cache_update(self._rem_cache, new_rem)
            return PF(
                sf, self.npf.adjoint(), d_rem,
                rem=getattr(self, 'rem', S.Zero)
            )
        for (bs, fs), t in self.table.items():
            new_bs = tuple(b._eval_adjoint() for b in bs)
            new_fs = tuple(f._eval_adjoint() for f in fs)
            sf[(new_bs, new_fs)] = NumShift(t.adjoint(), 1, new_bs).doit(deep=False)
        return PF(
            sf, self.npf.adjoint(),
            rem=getattr(self, 'rem', S.Zero)
        )

    def n_apply(self, f, on_npf: bool = False) -> PF:
        sf = dict()
        for k, t in self.table.items():
            if (nt := f(t)) != 0:
                sf[k] = nt
        npf = f(self.npf) if on_npf else self.npf
        return PF(
            sf, npf, self._rem_cache,
            rem=getattr(self, 'rem', S.Zero)
        )

    def diff(self, *symbols, **kwargs) -> PF:
        resolved = []
        for s in symbols:
            if isinstance(s, PF):
                if s.npf == S.Zero and set(s.table.keys()).issubset({_vpk}):
                    resolved.append(s.table.get(_vpk, S.Zero))
                else:
                    raise ValueError(f"Cannot differentiate with respect to operator: {s}")
            elif isinstance(s, str):
                resolved.append(Symbol(s))
            else:
                resolved.append(s)

        new_table = {}
        for k, v in self.table.items():
            dv = sympy_diff(v, *resolved, **kwargs)
            if dv != 0 and dv != S.Zero:
                new_table[k] = dv
        if not new_table and self.npf == S.Zero:
            new_table = {_vpk: S.Zero}
        new_npf = sympy_diff(self.npf, *resolved, **kwargs)
        new_rem = sympy_diff(self.rem, *resolved, **kwargs) if self.rem != S.Zero else S.Zero
        return PF(
            new_table, new_npf, self._rem_cache,
            rem=new_rem
        )


def pf_add(*pfs):
    if not pfs:
        return PF()
    if len(pfs) == 1:
        p = pfs[0]
        return p if isinstance(p, PF) else PF.from_val(p)
    result = dict()
    cache = dict()
    for pf in pfs:
        if not isinstance(pf, PF):
            pf = PF.from_val(pf)
        for p, f in pf.table.items():
            if p in result:
                result[p].append(f)
            else:
                result[p] = [f]
        _rem_cache_update(cache, pf._rem_cache)
    for p, f in list(result.items()):
        val = Add(*f)
        if val != 0 and val is not S.Zero:
            result[p] = val
        else:
            del result[p]
    return PF(result, Add(*(pf.npf for pf in pfs if pf.npf is not S.Zero)), cache)

PF.add = staticmethod(pf_add)

def clear_cache():
    _k_cache.clear()
    _rem_to_del.clear()

PF0 = PF()  # Singleton null PatternForm
PF1 = PF({_vpk: S.One})  # Singleton identity PatternForm

class PFTreeNode:
    """
    Processor that refines the expression to pat-form.

    The non-commutative expression types that is irrelavant of the ordering
    should be included in the "ignore" parameter, in a concrete iterable form.
    """
    __slots__ = ('expr', 'limits', 'ivar_outer', '_cn_cache', 'ignore')

    def __init__(self, expr: Expr, limits=None, ivar_outer=None,
                 ignore=(), _cn_cache=None, _s=True):
        if _s:
            from sympy.simplify.simplify import simplify
            expr = redo_mul(simplify(expr), num_repr=True)
        self._cn_cache = dict() if _cn_cache is None else _cn_cache
        if isinstance(expr, AddWithLimits):
            if not limits:
                self.limits = (expr.func, *expr.limits)
            elif limits[0] is expr.func:
                self.limits = (limits[0], *expr.limits, *limits[1:])
            else:
                self.limits = (expr.func, *expr.limits, *limits)
            expr = expr.function
        elif limits:
            self.limits = tuple(limits)
        self.expr = expr
        if ivar_outer:
            self.ivar_outer = ivar_outer
        self.ignore = tuple(ignore)

    def ilv(self):
        """integral/summation variables of this layer"""
        v = set()
        for x in getattr(self, 'limits', ()):
            if not isinstance(x, type):
                v.add(x[0])
        return v
    def ov(self):
        """outer integral/summation variables"""
        return set(getattr(self, 'ivar_outer', {}))
    def ivar(self):
        """integral/summation variables"""

        return self.ilv() | self.ov()

    def ls_apply(self, expr: Expr | None = None) -> Expr:
        """
        apply the integral/summation limits on the given Expr
        """
        if expr is None:
            expr = self.expr
        l = []
        for x in getattr(self, 'limits', ()):
            if issubclass(x, AddWithLimits):
                if l:
                    expr = f(expr, *l)
                f = x
            else:
                l.append(x)
        if l:
            expr = f(expr, *l)
        return expr

    def recur(self):
        """
        Next layer of recursion in the pattern-form reordering
        """
        ov = self.ov()
        ivar = self.ivar()
        v = lambda a: not (Tuple(a.exps).atoms(Symbol) & ivar)
        match self.expr:
            case 0:
                yield PF()
            case a if a.is_Atom:
                if not self.expr.is_commutative:
                    yield PF(npf=a)
                yield PF({((), ()): a})
            case BosonPat():
                if v(self.expr):
                    yield PFLeaf((self.expr,), ())
                yield PF(npf=self.expr)
            case FermionPat():
                if v(self.expr):
                    yield PF({((), (self.expr,)): S.One})
                yield PF(npf=self.expr)
            case Add():
                for arg in self.expr.args:
                    yield PFTreeNode(arg, getattr(self, 'limits', None), ov,
                                     self.ignore, self._cn_cache, False)
            case Pow() | Function():
                for arg in self.expr.args:
                    yield PFTreeNode(arg, None, ivar,
                                     self.ignore, self._cn_cache, False)
            case Subs():
                yield PFTreeNode(self.expr.expr, None, ivar,
                                 self.ignore, self._cn_cache, False)
                for p in self.expr.point:
                    yield PFTreeNode(p, None, ivar,
                                     self.ignore, self._cn_cache, False)
            case Derivative():
                yield PFTreeNode(self.expr.expr, None, ivar,
                                 self.ignore, self._cn_cache, False)
            case Mul():
                bs, fs = [], []
                for arg in self.expr.args:
                    if isinstance(arg, BosonPat) and v(arg):
                        bs.append(arg)
                    elif isinstance(arg, FermionPat) and v(arg):
                        fs.append(arg)
                    elif arg in self.ignore or check_num(arg, self._cn_cache):
                        bs.append(arg)
                    else:
                        if bs or fs:
                            yield PFLeaf(tuple(bs), tuple(fs))
                            bs.clear()
                            fs.clear()
                        yield PFTreeNode(arg, None, ivar,
                                         self.ignore, self._cn_cache, False)
                if bs or fs:
                    yield PFLeaf(tuple(bs), tuple(fs))
            case a if self.ignore and isinstance(a, self.ignore):
                yield PF({((), ()): a})
            case _:
                warnings.warn('Received unexpected expression')
                yield PF(npf=self.ls_apply())

    def result_ind(self) -> PF:
        """
        Pattern-formed result assuming independent modes,
        operators reordered canonically

        If a node includes any child that can never be represented in pat-form,
        the other chidren will still be treated
        """
        match self.expr:
            case Function():
                t = True
                new_args = []
                for c in self.recur():
                    pf = c.result_ind()
                    new_args.append(pf.expr())
                    if pf.npf is not S.Zero or not set(pf.table.keys()).issubset({((), ())}):
                        t = False
                if tuple(new_args) == self.expr.args:
                    expr = self.ls_apply()
                else:
                    expr = self.ls_apply(self.expr.func(*new_args))
                if t:
                    return PF({((), ()): expr})
                return PF(npf=expr)
            case Subs():
                t = True
                new_point = []
                r = self.recur()
                pf = next(r).result_ind()
                e0 = pf.expr()
                if pf.npf is not S.Zero or not set(pf.table.keys()).issubset({((), ())}):
                    t = False
                for c in r:
                    pf = c.result_ind()
                    new_point.append(pf.expr())
                    if pf.npf is not S.Zero or not set(pf.table.keys()).issubset({((), ())}):
                        t = False
                if e0 == self.expr.expr and tuple(new_point) == self.expr.point:
                    expr = self.ls_apply()
                else:
                    expr = self.ls_apply(Subs(e0, self.expr.variables, tuple(new_point)))
                if t:
                    return PF({((), ()): expr})
                return PF(npf=expr)
            case Derivative():
                pf = next(self.recur()).result_ind()
                e0 = pf.expr()
                if e0 == self.expr.expr:
                    expr = self.ls_apply()
                else:
                    expr = self.ls_apply(*self.expr.variable_count)
                if e0.npf is S.Zero:
                    return PF({((), ()): expr})
                return PF(npf=expr)
            case Add():
                pf = pf_add(*(r.result_ind() for r in self.recur()))
            case Pow():
                rs = self.recur()
                base = next(rs).result_ind()
                exp = next(rs).result_ind()
                if base.npf is not S.Zero:
                    return PF(npf=self.ls_apply(Pow(base.expr(), exp.expr())))
                if exp.npf is not S.Zero or not set(exp.table.keys()).issubset({((), ())}):
                    return PF(npf=self.ls_apply(Pow(base.expr(), exp.expr())))
                if set(base.table.keys()).issubset({((), ())}):
                    return PF({((), ()):self.ls_apply(Pow(base.expr(), exp.expr()))})
                e = exp.expr()
                if not e.is_Integer or e < 0:
                    return PF(npf=self.ls_apply(Pow(base.expr(), e)))
                pf = base ** e
            case Mul():
                pf = PF1
                for r in self.recur():
                    pf = pf * r.result_ind()
            case _:
                pf = next(self.recur()).result_ind()
        if hasattr(self, 'limits'):
            if pf.table:
                c = self.ls_apply(S.One).doit()
                if c is not S.One:
                    for p, n in pf.table.items():
                        pf.table[p] = c * n
            if pf.npf is not S.Zero:
                pf.npf = self.ls_apply(pf.npf)
        return pf

    def result_rem(self) -> tuple[PF, Expr]:
        """
        Pattern-formed result with remainder due to non-commutativity among modes,
        operators reordered canonically

        Non-num_only args of Pow/Function will not be treated further;
        if the number operators in different modes appear
        in the args of a single Pow/Function,
        they will be temporarily considered as independent.
        """
        if isinstance(self.expr, (Function, Derivative, Subs)):
            if all(check_num(c.expr, self._cn_cache) for c in self.recur()):
                return PF({((), ()): self.ls_apply()}), S.Zero
            return PF(npf=self.ls_apply()), S.Zero
        if isinstance(self.expr, Add):
            pfs, rds = [], []
            for r in self.recur():
                pf, rd = r.result_rem()
                pfs.append(pf)
                rds.append(rd)
            pf, rem = pf_add(*pfs), Add(*rds)
        elif isinstance(self.expr, Pow):
            exp = self.expr.exp
            if not check_num(exp, self._cn_cache):
                return PF(npf=self.ls_apply()), S.Zero
            base_pf, base_rd = next(self.recur()).result_rem()
            if base_pf.npf is not S.Zero:
                return PF(npf=self.ls_apply()), S.Zero
            if not (exp.is_Integer and exp > 1):
                if set(base_pf.table.keys()).issubset({((), ())}):
                    warnings.warn("Some residual table may be left uncalculated")
                    return PF({((), ()): Pow(base_pf[((), ())], exp)
                                                        * self.ls_apply(S.One).doit()}), S.Zero
                return PF(npf=self.ls_apply()), S.Zero
            pf, rem = base_pf._pow_rem(exp, base_rd)
        elif isinstance(self.expr, Mul):
            pf, rem = PF1, S.Zero
            for r in self.recur():
                new, rd = r.result_rem()
                new_rem = Mul(pf.expr(), rd) if rd is not S.Zero else S.Zero
                pf, mr = pf._mul_rem(new)
                rem = Add(Mul(rem, Add(new.expr(), mr)), mr, new_rem)
        else:
            pf, rem = next(self.recur()).result_rem()
        if hasattr(self, 'limits'):
            if pf.table:
                c = self.ls_apply(S.One).doit()
                if c is not S.One:
                    for p, n in pf.table.items():
                        pf.table[p] = c * n
            if pf.npf is not S.Zero:
                pf.npf = self.ls_apply(pf.npf)
            if rem is not S.Zero:
                rem = self.ls_apply(rem)
        return pf, rem

class PFLeaf:
    __slots__ = ('bsc', 'fsc')
    def __init__(self, bs:tuple[BosonPat], fs:tuple[FermionPat]):
        self.bsc = BSC(bs)
        self.fsc = FSC(fs)

    def result_ind(self):
        bp, bt = self.bsc.result_ind()
        fp, ft = self.fsc.result_ind()
        if fp is None:
            return PF(dict())
        return PF({(bp, fp): bt * ft})
        
    def result_rem(self):
        bp, bt, br = self.bsc.result_rem()
        fp, ft, fr = self.fsc.result_rem()
        if fp is None:
            return PF(dict()), (Mul(*bp, bt) + br) * fr
        table = {(bp, fp): bt * ft}
        remainder = Add(Mul(*bp, bt, fr), Mul(*fp, ft, br), Mul(br, fr))
        return PF(table, cache=getattr(self.bsc, '_rem_cache', None)), remainder

def pattern_form(expr, as_dict=False, rem=False, ignore=(), _s=True):
    """
    Returns the pattern-form result of the expr.
    
    :param as_dict: Whether to present the result as a dict (True),
    PF (None) or Expr (False)
    :param rem: Whether to consider the non-independency among the operators
    and include the commutator remainder
    :param ignore: Uncommtable types to be processed as if they are BosonNums
    :param _s: Whether to symplify the expression before reordering
    """
    pftn = PFTreeNode(expr, ignore=ignore, _s=_s)
    if rem:
        result, remainder = pftn.result_rem()
        clear_cache()
        if as_dict is None:
            return result, remainder
        if as_dict:
            return result.table, result.npf, remainder
        return result.expr(), remainder
    if as_dict is None:
        return pftn.result_ind()
    if as_dict:
        result = pftn.result_ind()
        return result.table, result.npf
    return pftn.result_ind().expr()

pf = pattern_form
