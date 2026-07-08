from sympy.core.singleton import S
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import Function
from sympy.concrete.expr_with_limits import ExprWithLimits
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from op_patterns.boson_pattern import BosonPat, BosonNum, NumShift
from op_patterns.fermion_pattern import FermionPat, m1_base, UidCommutator
from op_patterns.tools import mul_pattern, extract_modes
from sortedcontainers import SortedDict

__all__ = ['BosonSortContext', 'FermionSortContext', 'PFTableProcessor']

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

_res_to_del = dict()
def _res_cache_update(cache, new):
    if cache is new:
        return
    for b in new:
        if b in cache:
            if b in _res_to_del:
                for n in _res_to_del[b]:
                    del cache[b][n]
                del _res_to_del[b]
            cache[b].update(new[b])
        else:
            cache[b] = new
    for b in _res_to_del:
        for n in _res_to_del[b]:
            del cache[b][n]
    _res_to_del.clear()

class BosonSortContext():
    """
    Sort context for a tuple/list of BosonPats or number-operator functions
    """
    __slots__ = ('pats', 'result', 'tail', 'residue',
                 '_res_cache', 'pat_sorted', 'tail_pat', 'residue_pat')

    def __init__(self, pats):
        if not pats:
            self.result = self.pat_sorted = tuple(pats)
            self.tail = self.tail_pat = S.One
            self.residue = self.residue_pat = S.Zero
        elif len(pats) == 1:
            if isinstance(pats[0], BosonPat):
                self.pat_sorted = tuple(pats)
                self.tail_pat = S.One
            else:
                self.result = self.pat_sorted = ()
                self.tail = self.tail_pat = pats[0]
                self._res_cache = dict()
            self.residue = self.residue_pat = S.Zero
        else:
            self.pats = tuple(pats)
            self._res_cache = dict()

    def expr(self) -> Expr:
        if hasattr(self, 'pats'):
            return Mul(*self.pats[0], *self.pats[1])
        if not self.result:
            return self.tail
        return self.result[0]

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

    def _b_res(self, e_N: Expr, b: BosonOp) -> Expr:
        """
        Calculates the residue after reordering
        e_N\\*b into b\\*NumShift(e_N, b).

        The global _res_to_del may change,
        so please do not forget the corresponding cache management.
        """
        if e_N.is_commutative:
            return S.Zero
        if e_N in self._res_cache.get(b, ()):
            if b in _res_to_del:
                _res_to_del[b].add(e_N)
            else:
                _res_to_del[b] = {e_N}
            return self._res_cache[b][e_N]
        if isinstance(e_N, NumShift):
            f_res = self._b_res(e_N.function, b)
            return NumShift(f_res, e_N.direction, e_N.shift, scan_dict=False)
        if isinstance(e_N, ExprWithLimits):
            return e_N.func(self._b_res(e_N.function, b), *e_N.limits)
        if isinstance(e_N, Add):
            return Add(*[self._b_res(arg, b) for arg in e_N.args])
        if isinstance(e_N, Mul):
            prev, res = 0, []
            for i, arg in enumerate(e_N.args):
                modes = extract_modes(arg)[0]
                if b in modes:
                    if i != prev:
                        res.append(Mul(*e_N.args[:prev],
                                    Commutator(Mul(*e_N.args[prev:i]), b),
                                    *e_N.args[i+1:]))
                    prev = i + 1
                    new = self._b_res(arg, b)
                    if new is not S.Zero:
                        res.append(Mul(*e_N.args[:i], new, *e_N.args[prev:]))
            if prev != len(e_N.args):
                res.append(Mul(*e_N.args[:prev],
                           Commutator(Mul(*e_N.args[prev:]), b)))
            return Add(*res)
        if isinstance(e_N, (BosonNum, Pow, Function)):
            modes = extract_modes(e_N)[0]
            if b in modes:
                return S.Zero
        return Commutator(e_N, b)
    def merge_res_pat(self, R, reserve_unshifted=False):
        """
        Merges a new tuple/list of sorted BosonPat
        to the right side of the sorted result
        and calculates the total residue.
        
        The global _res_to_del may change,
        so please do not forget the corresponding cache management.

        The parameter 'reserve_unshifted' is
        for avoiding removal of cache about the tail itself,
        which is essential for the treatment of integer powers.
        """
        L, T, rd = self.sort_res_pat()
        new = []
        res = []
        if rd is not S.Zero:
            res.append(Mul(rd, *R))
        if T is S.One:
            tail = []
        elif T.is_commutative:
            tail = [T]
        else:
            rrt = []
            tail = [NumShift(T, 1, R).doit(deep=False)]
            for n, b in enumerate(R):
                bb = BosonOp(b.name)
                rn = self._b_res(T, bb)
                if bb in self._res_cache:
                    self._res_cache[bb].update({tail[0]: NumShift(rn, 1, R).doit(deep=False)})
                    if bb in _res_to_del:
                        _res_to_del[bb].discard(tail[0])
                        if reserve_unshifted:
                            _res_to_del[bb].discard(T)
                if rn is not S.Zero:
                    rrt.append(Mul(*R[:n], rn.xreplace({bb: b}), *R[n+1:]))
            if rrt:
                res.append(Mul(*L, Add(*rrt), evaluate=(len(rrt)==1)))
            del rrt
        l, r = len(L), len(R)
        i = j = j_prev = 0
        while i < l and j < r:
            if key(L[i]) < key(R[j]):
                if j_prev != j:
                    c = Commutator(Mul(*L[i:], evaluate=False),
                                    Mul(*R[j_prev:j], evaluate=False))
                    res.append(Mul(*new[:j_prev-j], c, *R[j:], *tail))
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
                res.append(Mul(*new, L[i], c, *R[j:], *tail))
                i += 1
                j_prev = j
                new.append(np)
        if j_prev != j:
            c = Commutator(Mul(*L[i:], evaluate=False),
                            Mul(*R[j_prev:j], evaluate=False))
            res.append(Mul(*new[:j_prev-j], c, *R[j:], *tail))
        return (*new, *L[i:], *R[j:]), Mul(*reversed(tail)), Add(*res)
    def sort_res_pat(self):
        if not hasattr(self, 'residue_pat'):
            mid = len(self.pats)//2
            L = BosonSortContext(tuple(self.pats[:mid]))
            R = BosonSortContext(tuple(self.pats[mid:]))
            lsp, ltp, lrp = L.sort_res_pat()
            rsp, rtp, rrp = R.sort_res_pat()
            self.pat_sorted, t, res = L.merge_res_pat(rsp)
            _res_cache_update(self._res_cache, getattr(R, '_res_cache', ()))
            self.tail_pat = t * rtp
            self.residue_pat = Add(Mul(res, rtp),
                            Mul(Mul(*lsp, ltp) + lrp, rrp))
        return self.pat_sorted, self.tail_pat, self.residue_pat
    def result_res(self):
        if not hasattr(self, 'residue'):
            ps, tp, rp = self.sort_res_pat()
            new = []
            tail = []
            res = []
            for i, p in enumerate(ps):
                a, n = p.num_repr()
                if a is not S.One:
                    new.append(a)
                if n is not S.One:
                    res.append(Mul(*new,
                                   Commutator(n, Mul(*ps[i+1:], evaluate=False)),
                                   *tail))
                    tail.append(n)
            self.result = tuple(new)
            self.tail = Mul(*tail, tp)
            self.residue = Mul(Add(*res), tp) + rp
        elif not hasattr(self, 'result'):
            return *self.result_ind(), self.residue
        return self.result, self.tail, self.residue

class FermionSortContext:
    """
    sort context for a tuple/list of FermionPats
    """
    __slots__ = ('pats', 'result', 'tail', 'residue')

    def __init__(self, pats):
        if not pats:
            if pats is None:
                self.result = None
                self.tail = S.Zero
            else:
                self.result = tuple(pats)
                self.tail = S.One
            self.residue = S.Zero
        else:
            if len(pats) == 1:
                self.result = tuple(pats)
                self.tail = S.One
                self.residue = S.Zero
            else:
                self.pats = tuple(pats)
                self._res_cache = dict()

    def expr(self) -> Expr:
        if hasattr(self, 'pats'):
            return Mul(*self.pats[0], *self.pats[1])
        if not self.result:
            return self.tail
        return self.result[0]

    def result_ind(self):
        if not hasattr(self, 'result'):
            mid = len(self.pats)//2
            L = FermionSortContext(tuple(self.pats[:mid]))
            R = FermionSortContext(tuple(self.pats[mid:]))
            ls, _ = L.result_ind()
            rs, rt = R.result_ind()
            if ls is None:
                self.result = None
                self.tail = S.Zero
                if rs is None:
                    self.residue = S.Zero
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
                tail_args.append(l_arg * Add(m1_base(p.comm_symm)
                                             for p in R[j_prev:j]))
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
            l_arg -= m1_base(L[i].comm_symm)
            tail_args.append(l_arg * Add(m1_base(p.comm_symm)
                                         for p in R[j_prev:j]))
        return (*new, *L[i:], *R[j:]), Pow(S.NegativeOne, Add(*tail_args))

    def merge_res(self, R):
        L, T, r = self.result_res()
        if L is None or R is None:
            return None, S.Zero, S.Zero
        new = []
        res = []
        if r is not S.Zero:
            res.append(Mul(r, *R))
        if T is S.One:
            tail = []
        else:
            tail = [T]
        l, r = len(L), len(R)
        i = j = j_prev = 0
        def record():
            c = UidCommutator(Mul(*L[i:], evaluate=False),
                            Mul(*R[j_prev:j], evaluate=False))
            res.append(Mul(*tail, *new[:j_prev-j], c,
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
                j += 1
                c = UidCommutator(Mul(*L[i+1:], evaluate=False),
                                Mul(*R[j_prev:j], evaluate=False))
                res.append(Mul(*tail, L[i], *new, c, *R[j:]))
                if isinstance(c, AntiCommutator):
                    tail.append(S.NegativeOne)
                elif isinstance(c, UidCommutator):
                    tail.append(c.comm_sign)
                prod = mul_pattern(L[i], R[j])[0]
                if prod == 0:
                    return None, S.Zero, Add(*res)
                new.append(prod)
                i += 1
                j_prev = j
        if j_prev != j:
            record()
        return (*new, *L[i:], *R[j:]), Mul(*reversed(tail)), Add(*res)
    def result_res(self):
        if not hasattr(self, 'residue'):
            mid = len(self.pats)//2
            L = FermionSortContext(tuple(self.pats[:mid]))
            R = FermionSortContext(tuple(self.pats[mid:]))
            ls, lt, lr = L.result_res()
            rs, rt, rr = R.result_res()
            if ls is None:
                self.result = None
                self.tail = S.Zero
                if rs is None:
                    self.residue = S.Zero
                else:
                    self.residue = Mul(lr, *rs, R.tail)
            elif R.result is None:
                self.result = None
                self.tail = S.Zero
                self.residue = Mul(*ls, lt, rr)
            else:
                self.result, t, res = L.merge_res(rs)
                self.tail = Pow(S.NegativeOne, m1_base(t) + m1_base(rt))
                self.residue = Add(Mul(res, rt),
                                Mul(Mul(*ls, lt) + lr, rr))
        return self.result, self.tail, self.residue

class PFTableProcessor:
    """
    Final representation of pattern form
    """
    __slots__ = ('table', 'npf', '_res_cache')

    def __init__(self, table: dict = dict(), npf=S.Zero, cache=None):
        self.table = table
        self.npf = npf
        if cache is None:
            self._res_cache = dict()
        else:
            self._res_cache = cache

    def __repr__(self):
        return f"PatternForm({self.table}, {self.npf})"

    def expr(self, with_npf=True) -> Expr:
        if with_npf:
            return Add(*(Mul(*a[0], n, *a[1]) for a, n in self.table.items()), self.npf)
        else:
            return Add(*(Mul(*a[0], n, *a[1]) for a, n in self.table.items()))

    def result_ind(self):
        return self
    def result_res(self):
        return self, S.Zero
    
    @classmethod
    def add(cls, *pfs):
        if not pfs:
            return PFTableProcessor()
        if len(pfs) == 1:
            return pfs[0]
        result = dict()
        cache = dict()
        for pf in pfs:
            for p, f in pf.table.items():
                if p in result:
                    result[p].append(f)
                else:
                    result[p] = [f]
            _res_cache_update(cache, pf._res_cache)
        for p, f in result.items():
            result[p] = Add(*f)
        return cls(result, Add(*(pf.npf for pf in pfs if pf.npf is not S.Zero)), cache)

    def make_contexts(self):
        for (bs, fs), n in self.table.items():
            bsc = object.__new__(BosonSortContext)
            bsc.result = bsc. pat_sorted = bs
            bsc.tail = bsc.tail_pat = n
            bsc.residue = bsc.residue_pat = S.Zero
            bsc._res_cache = self._res_cache
            fsc = object.__new__(FermionSortContext)
            fsc.result = fs
            fsc.tail = S.One
            fsc.residue = S.Zero
            yield bsc, fsc

    def mul_ind(self, other):
        sf = dict()
        for bsc, fsc in self.make_contexts():
            for (bs, fs), t in other.table.items():
                bp, bt = bsc.merge_ind(bs)
                fp, ft = fsc.merge_ind(fs)
                if fp is not None:
                    a = (bp, fp)
                    sf[a] = sf.get(a, 0) + Mul(ft, bt, t)
        return PFTableProcessor(sf, Add(Mul(self.expr(), other.npf),
                                        Mul(self.npf, other.expr(False))))

    def mul_res(self, other, reserve_unshifted=False):
        sf, res = dict(), []
        for bsc, fsc in self.make_contexts():
            for (bs, fs), t in other.table.items():
                nbsc = object.__new__(BosonSortContext)
                (nbsc.pat_sorted,
                 nbsc.tail_pat,
                 nbsc.residue_pat) = bsc.merge_res_pat(bs, reserve_unshifted)
                bp, bt, br = nbsc.result_res()
                fp, ft, fr = fsc.merge_res(fs)
                if fp is not None:
                    a = (bp, fp)
                    sf[a] = sf.get(a, 0) + Mul(ft, bt, t)
                    res.append(Mul(br, t, *fs))
                res.append(Mul(Mul(*bp, bt) + br, t, fr))
        _res_cache_update(self._res_cache, other._res_cache)
        return PFTableProcessor(sf, Add(Mul(self.expr(), other.npf),
                                        Mul(self.npf, other.expr(False))),
                                self._res_cache), Add(*res)

    def power_ind(self, exp: Integer):
        if exp is S.One:
            return self
        if exp < 0:
            raise ValueError('Exponent must be nonnegative')
        if exp is S.Zero:
            return PFTableProcessor({((), ()): S.One})
        sf_k = self.power_ind(exp >> 1)
        sf_2k = sf_k.mul_ind(sf_k)
        if exp & 1 == 0:
            return sf_2k
        return self.mul_ind(sf_2k)

    def power_res(self, exp: Integer, base_rd):
        if exp is S.One:
            return self, base_rd
        if exp < 0:
            raise ValueError('Exponent must be nonnegative')
        if exp is S.Zero:
            return PFTableProcessor({((), ()): S.One}), S.Zero
        sf_k, rd_k = self.power_res(exp >> 1, base_rd)
        sf_2k, r_sf2k = sf_k.mul_res(sf_k, True)
        rd_2k = Add(r_sf2k, Mul(sf_k.expr(), rd_k),
                    Mul(rd_k, sf_k.expr()), Mul(rd_k, rd_k))
        if exp & 1 == 0:
            return sf_2k, rd_2k
        sf, r_sf = self.mul_res(sf_2k, True)
        rd = Add(r_sf, Mul(self.expr(), rd_2k),
                 Mul(base_rd, sf_2k.expr() + rd_2k))
        return sf, rd
    
    def adjoint(self, res=False):
        sf = dict()
        if res:
            new_res = dict()
            d_res = dict()
            res = []
            for (bs, fs), t in self.table.items():
                td = NumShift(t, -1, bs).doit(deep=False)._eval_adjoint()
                bd = tuple(b._eval_adjoint() for b in reversed(bs))
                bsc = BosonSortContext(bd)
                bsc._res_cache = self._res_cache
                br = bsc.sort_res_pat()[2]
                r = []
                for i, b in enumerate(bd):
                    bb = BosonOp(b.name)
                    rb = bsc._b_res(t, bb)
                    rd = NumShift(rb, -1, bs).doit(deep=False)
                    new_res[bb] = {t: rb}
                    r.append(Mul(*bd[i:], rd.xreplace({bb: bs[-(i+1)]})._eval_adjoint(), *bd[:i+1]))
                    d_res[bb] = {td: rd._eval_adjoint().xreplace({bb._eval_adjoint(): bb})}
                fp, ft, fr = FermionSortContext(tuple(f._eval_adjoint()
                                                      for f in reversed(fs))).result_res()
                res.append(Mul(ft, Add(*r, -br*td), *fp) + Mul(*bd, td, fr))
                sf[(bd, fp)] = ft * td
            _res_cache_update(self._res_cache, new_res)
            return PFTableProcessor(sf, self.npf.adjoint(), d_res)
        for (bs, fs), t in self.table.items():
            new_bs = tuple(b._eval_adjoint() for b in bs)
            new_fs = tuple(f._eval_adjoint() for f in fs)
            sf[(new_bs, new_fs)] = NumShift(t.adjoint(), 1, new_bs).doit(deep=False)
        return PFTableProcessor(sf, self.npf.adjoint())
    
    def n_apply(self, f, on_npf=False):
        """
        Apply the same operation on all the number-operator tails, i.e. table.values()
        
        Able to control whether to apply the same operation
        on non-pattern-form part with parameter 'on_npf'
        """
        sf = dict()
        for k, t in self.table.items():
            sf[k] = f(t)
        npf = f(self.npf) if on_npf else self.npf
        return PFTableProcessor(sf, npf)

def clear_cache():
    _k_cache.clear()
    _res_to_del.clear
