from collections import defaultdict
from collections.abc import Sequence
from functools import cached_property, reduce
from operator import mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import _mexpand
from sympy.core.traversal import bottom_up
from sympy.core.exprtools import factor_terms
from sympy.physics.quantum.boson import BosonOp
from ..pattern_form import BSC, PF, pf, PF0, PF1, PE
from ..op_pattern import BosonPat, BosonNum, FermionPat
from ..pattern_form.pattern_form import pf_add, pf_mul_ind, _vpk
from ..pattern_form.perturb_expander import perturb_rec
from .dsolve_pm import group_terms, _ps
from .multimode import (MultimodeCOP, MultimodeNum, MultimodeNum2,
                        t_mul, mmn_shift, _mps, _tdz, _pd)
import warnings

__all__ = ['PMProcessor', 'wm_processors', 'a_p', 'a_s', 'a_i', 't_', '_g', '_ps', 'nlc', 'nm', 'vac_pf']

warnings.simplefilter('once', UserWarning)


def _collapse_boson_num(expr, vac_names):
    if not isinstance(expr, Expr) or expr.is_number:
        return expr
    return expr.replace(
        lambda x: isinstance(x, BosonNum),
        lambda x: S.Zero if (x.name in vac_names or str(x.name) in vac_names) else x
    )


def vac_pf(p: PF, vac_mode) -> PF:
    """
    Evaluates the vacuum projection of a PatternForm (PF) for given mode(s).

    :param p: Input PatternForm object.
    :param vac_mode: Mode name, Symbol, or iterable of mode names asserted to be in vacuum.
    :return: Collapsed PatternForm.
    """
    if not p:
        return PF0

    if isinstance(vac_mode, (str, Symbol)) or not hasattr(vac_mode, '__iter__'):
        vac_set = {vac_mode.name if hasattr(vac_mode, 'name') else vac_mode}
    else:
        vac_set = {m.name if hasattr(m, 'name') else m for m in vac_mode}
    vac_names = {sympify(m) for m in vac_set} | {str(m) for m in vac_set}

    new_table = {}
    for (bs, fs), tail in p.table.items():
        if any((b.name in vac_names or str(b.name) in vac_names) for b in bs):
            continue
        c_tail = _collapse_boson_num(tail, vac_names)
        if c_tail != 0:
            new_table[(bs, fs)] = c_tail

    new_npf = _collapse_boson_num(p.npf, vac_names)
    return PF(new_table, new_npf) if (new_table or new_npf != 0) else PF0


def _merge_pf_dicts(ds) -> dict:
    """Merges multiple {tuple_key: PF} dicts by summing PF values for identical keys."""
    grouped = defaultdict(list)
    for d in ds:
        for k, v in d.items():
            grouped[k].append(v)
    return {k: p for k, vs in grouped.items() if (p := pf_add(*vs))}


# Dummy symbols for MultimodeNum, MultimodeNum2, driving BosonNum
_n1, _n2, _nd = symbols('n_1 n_2 n_d', Dummy=True, nonnegative=True)
# Field names: a0 - pump (high frequency), a1, a2 - signal and idler (low frequency)
a_p = BosonOp('a_0')
a_s = BosonOp('a_1')
a_i = BosonOp('a_2')
_g = Dummy('g', real=True)  # Dummy symbol for the parametric gain

# Time / nonlinear coefficient / mode count symbols: t, χ, n_m
t_ = Dummy('t', real=True)
nlc = Symbol('χ', real=True, nonnegative=True)
nm = Symbol('n_m', nonnegative=True)

_e = lambda p: S.Zero if p == 1 else (p.exp if p.is_Pow else S.One)


class PMProcessor:
    """
    Base processor for quantum multimode parametric wave-mixing processes (TWM, FWM).

    Retains multi-object addition (d_add) and methods depending on lower-level
    objects (Expr, PF, BosonOp, tuple) or external parameters (order, func, vac_modes).
    """

    def __init__(self, dc=True):
        self.dc = bool(dc)
        self.nd = BosonNum(a_p.name) if dc else BosonNum(a_i.name)
        self.a = ((MultimodeCOP(a_s.name, a_i.name, annih1=0),
                   MultimodeCOP(a_s.name, a_i.name))
                  if dc else (MultimodeCOP(a_p.name, a_s.name, 1, 0),
                              MultimodeCOP(a_p.name, a_s.name, 1)))
        self.mmn2 = MultimodeNum2(*self.a[0].args[:3])
        self.reset_pe()

    def reset_pe(self, fwm: bool = False):
        self.pe = PE({nlc: S.One, self.nd.name: -S.Half if fwm else S.NegativeOne})
        self.pe.register(MultimodeNum)
        self.pe.register(MultimodeNum2)
        self.pe.register(MultimodeCOP)

    @cached_property
    def init_ops(self) -> dict:
        from .pm_wrapper import PMWrapper
        if self.dc:
            return {
                'a_0': PMWrapper(self, {(None,): PF({((a_p,), ()): S.One})}),
                'a_1': PMWrapper(self, {a_s: {(None,): PF1}}),
                'a_2': PMWrapper(self, {a_i: {(None,): PF1}}),
                'a_p': PMWrapper(self, {(None,): PF({((a_p,), ()): S.One})}),
                'a_s': PMWrapper(self, {a_s: {(None,): PF1}}),
                'a_i': PMWrapper(self, {a_i: {(None,): PF1}}),
                'A': PMWrapper(self, {(True, 1): PF1}),
                'N_t': PMWrapper(self, {(None,): PF({_vpk: MultimodeNum2(a_s.name, a_i.name, S.Zero)})}),
            }
        return {
            'a_0': PMWrapper(self, {a_p: {(None,): PF1}}),
            'a_1': PMWrapper(self, {a_s: {(None,): PF1}}),
            'a_2': PMWrapper(self, {(None,): PF({((a_i,), ()): S.One})}),
            'a_p': PMWrapper(self, {(None,): PF({((a_i,), ()): S.One})}),
            'a_s': PMWrapper(self, {a_p: {(None,): PF1}}),
            'a_i': PMWrapper(self, {a_s: {(None,): PF1}}),
            'A': PMWrapper(self, {(True, 1): PF1}),
            'N_t': PMWrapper(self, {(None,): PF({_vpk: MultimodeNum2(a_p.name, a_s.name, S.One)})}),
        }

    def odag(self, b: BosonOp) -> BosonOp:
        """
        Returns the partner single-mode operator according to the parity rule of self.a[0].
        """
        n1, n2, p = self.a[0].args[:3]
        if not isinstance(b, BosonOp):
            raise ValueError(f'{b} is not a legal weak-field key for this instance')
        if b.name == n1:
            return BosonOp(n2, b.is_annihilation == p)
        if b.name == n2:
            return BosonOp(n1, b.is_annihilation == p)
        raise ValueError(f'{b} is not a legal weak-field key for this instance')

    def f_parity(self, dk: Sequence[BosonOp]):
        """
        Determines the parity class (0 or 1) of a sequence of weak-field BosonOp keys.
        Raises ValueError if keys have mixed parities.
        """
        if not dk:
            return None
        dk = tuple(dk)
        b0 = dk[0]
        b0_od = self.odag(b0)
        for b in dk[1:]:
            if not (b == b0 or b == b0_od):
                raise ValueError('Inconsistent BosonOp keys within weak-field expression')
        n1, _, p = self.a[0].args[:3]
        if b0.name == n1:
            return bool(b0.is_annihilation)
        return bool(b0.is_annihilation) == bool(p)

    def expand_func_d(self, func, order, *ds, sub_g: bool = False, gc=None):
        from .pm_wrapper import PMWrapper
        if not ds:
            return self.from_expr(func())

        ord_lims, flo = self.pe.handle_order_limits(func, len(ds))
        x0_list = []
        perturb_list = []
        for j, (d, ol) in enumerate(zip(ds, ord_lims)):
            w = PMWrapper.from_val(d, self)
            if w.b_left is not None:
                raise ValueError('The expression does not support expand-with-func')

            odt = defaultdict(dict)
            for k, v in w.dict_repr.items():
                for op, p in self.pe.expand_pf(v, order, True).items():
                    if op < ol:
                        raise ValueError('The input exceeds order limit')
                    odt[op][k] = p
            if ol is S.NegativeInfinity:
                if not odt:
                    raise ValueError('Unappropriate input for the function')
                ol = min(odt)
            v0 = odt.pop(ol, None)
            if not v0:
                x0_list.append(S.Zero)
            elif (v0.keys() == {(None,)} and v0[(None,)].npf == 0
                  and v0[(None,)].table.keys() == {_vpk}):
                x0_list.append(v0[(None,)].table[_vpk])
            else:
                raise ValueError('Unappropriate input for the function')
            for od, td in odt.items():
                if td:
                    perturb_list.append((j, od - ol, PMWrapper(self, td, None)))

        dummies = tuple(Dummy() for _ in ds)
        f_sym = func(*dummies)
        subs_dict = dict(zip(dummies, x0_list))
        fl = f_sym.subs(subs_dict)
        fle, fo = self.pe.leading_order(fl)
        if flo is S.NegativeInfinity:
            flo = fo
        if flo == fo and fle != fl:
            raise ValueError('The function is not properly defined')

        nol = order - flo
        if nol < 0:
            return PMWrapper(self, {}, None)

        @perturb_rec(_comm = lambda *_: False,
                     _is_comm = lambda d: getattr(d.as_scalar, 'is_commutative', False),
                     _is_num = lambda d: bool(d.as_scalar),
                     _add=self.d_add, _mul=lambda args: reduce(mul, args), independent=True)
        def term(df):
            if (val := df.subs(subs_dict)) != 0:
                return PMWrapper(self, {(None,): PF({_vpk: val})}, None)

        res = self.d_add(*(t for t, _ in term(perturb_list, nol, dummies, f_sym)))
        if sub_g and gc is not None:
            return res.apply(lambda t: t.subs({t_: _g / gc}))
        return res

    def n_simp(self, expr):
        n1, n2 = self.a[0].names
        def mmn_d(n, d):
            if n == n1:
                return _n1 + d * nm
            if n == n2:
                if self.dc:
                    return _n2 - _n1 + (d - 1) * nm
                return _n1 - _n2 + d * nm
            return MultimodeNum(n, d)

        def _simp_frac(e):
            if e.is_Pow and e.base.is_Add:
                return Pow(factor_terms(e.base, fraction=True), e.exp)
            return e

        def combine_n(expr):
            if expr.is_Add:
                weak_in_expr = {m.name for m in expr.atoms(MultimodeNum) if m.name in (n1, n2)}
                has_mmn2 = self.mmn2 in expr.atoms(MultimodeNum2)
                if len(weak_in_expr) == 2 or (len(weak_in_expr) == 1 and has_mmn2):
                    subbed = expr.replace(MultimodeNum, mmn_d).subs({self.mmn2: _n2})
                    subbed = bottom_up(subbed, _simp_frac)
                    r = factor_terms(_mexpand(subbed))
                    if r.has(_n1):
                        warnings.warn('Failed to combine the weak-field multimode number operators')
                    expr = r.subs({_n1: MultimodeNum(n1), _n2: self.mmn2})
            if expr.is_Mul:
                expr = _mps(expr, self.nd, _nd)
            return _ps(expr)

        return bottom_up(expr, combine_n)

    def p2s_group(self, expr):
        """
        Group the functions of g and process product-to-sum to
        hyperbolic/triangular functions.
        """
        grouped = defaultdict(list)
        for term in Add.make_args(expr):
            coeffs = []
            x_part = []
            for factor in Mul.make_args(term):
                if factor.has(_g):
                    x_part.append(factor)
                else:
                    coeffs.append(factor)
            term_grouped = group_terms(_mexpand(Mul(*x_part)), _g, self.dc)
            for k, v in term_grouped.items():
                grouped[k].append(Mul(Add(*v), *coeffs))
        return grouped

    def d_add(self, *ws):
        """
        Adds multiple WMWrapper objects together, collecting identical keys.
        """
        from .pm_wrapper import PMWrapper
        items = []
        for w in ws:
            if not w:
                continue
            if not isinstance(w, PMWrapper):
                w = PMWrapper.from_val(w, self)
            if w.dict_repr:
                items.append(w)

        if not items:
            return PMWrapper(self, {}, None)
        if len(items) == 1:
            return items[0]

        if all(w.b_left is None for w in items):
            return PMWrapper(self, _merge_pf_dicts(w.dict_repr for w in items), None)

        if all(w.b_left is not None for w in items):
            parities = {self.f_parity(tuple(w.dict_repr)) for w in items}
            if len(parities) > 1:
                raise ValueError("Cannot add single-mode weak-field operators of different parities")
            b_left = items[0].b_left
            result = defaultdict(list)
            for w in items:
                if w.b_left == b_left:
                    for k, v in w.dict_repr.items():
                        result[k].append(v)
                else:
                    for k, v in w.dict_repr.items():
                        for nk, nv in self.bd_swap(k, v, to_left=b_left).items():
                            result[nk].append(nv)
            out = {k: m for k, vs in result.items() if (m := _merge_pf_dicts(vs))}
            return PMWrapper(self, out, b_left if out else None)

        raise ValueError("Cannot add expressions with and without single-mode weak-field operators")

    def d_mul(self, d0: dict, d1: dict, d2: dict, mid=None):
        if not d1 or not d2:
            return
        if isinstance(mid, PF):
            d1 = {k1: v for k1, v1 in d1.items() if (v := pf_mul_ind(v1, mid))}
            if not d1:
                return
            mid = None
        weak_names = set(self.a[0].names)
        shift_n = lambda expr, pd: mmn_shift(expr, self.a[1], pd)
        is_mmn = isinstance(mid, Expr)
        t_join = (lambda k1, k2: t_mul(k1, k2)) if is_mmn else (lambda k1, k2: t_mul(k1, k2, mid))

        def _acc(k, p):
            if k in d0:
                p = pf_add(d0[k], p)
            if p:
                d0[k] = p
            elif k in d0:
                del d0[k]

        for k2, e2 in d2.items():
            pd = _pd(k2)
            for k1, e1 in d1.items():
                has_weak_pair = False
                for bs, _ in e1.table:
                    bw = tuple(b for b in bs if b.name in weak_names)
                    if bw:
                        try:
                            if self.match_cop(bw)[1] > 0:
                                has_weak_pair = True
                                break
                        except ValueError:
                            pass

                if not has_weak_pair:
                    new = e1.n_apply(lambda t: shift_n(mid * t if is_mmn else t, pd), on_npf=True)
                    _acc(t_join(k1, k2), pf_mul_ind(new, e2))
                else:
                    if e1.npf != 0:
                        c_npf = shift_n(mid * e1.npf if is_mmn else e1.npf, pd)
                        _acc(t_join(k1, k2), pf_mul_ind(PF({}, c_npf), e2))

                    for (bs, fs), tail in e1.table.items():
                        c_tail = shift_n(mid * tail if is_mmn else tail, pd)
                        try:
                            cop_a1, cop_exp, bs_other = self.match_cop(bs)
                        except ValueError:
                            cop_a1, cop_exp, bs_other = None, 0, bs

                        if cop_exp == 0 or cop_a1 is None:
                            _acc(t_join(k1, k2), pf_mul_ind(PF({(bs, fs): c_tail}, S.Zero), e2))
                        else:
                            for k_sub, new_m, P_num in self.tcp_swap(cop_a1, cop_exp, k2, to_left=False):
                                if new_m == 0:
                                    new_bw = ()
                                else:
                                    cop_tgt = self.a[1 if cop_a1 else 0]
                                    op1 = BosonOp(cop_tgt.names[0], cop_tgt.annih1)
                                    op2 = BosonOp(cop_tgt.names[1], cop_tgt.annih2)
                                    new_bw = (op1, op2) if new_m == 1 else (op1**new_m, op2**new_m)
                                p_term = pf_mul_ind(PF({(bs_other + new_bw, fs): c_tail * P_num}, S.Zero), e2)
                                _acc(t_join(k1, k_sub), p_term)

    def bp_swap(self, a: BosonOp, p: PF, s, to_left=True):
        """
        Swaps the single-mode boson operator (a) and given PF (p) with Multimode operators.
        """
        shift_n = lambda expr, b=a: s * mmn_shift(expr, b, 1, 1 if to_left else -1)
        a_od = self.odag(a)
        table_a, table_od = dict(), dict()
        new_npf = shift_n(p.npf)
        pa = self.f_parity((a,))
        for (bs, fs), tail in p.table.items():
            try:
                annih1 = self.match_cop(bs)[0]
            except ValueError:
                annih1 = None
            if annih1 is None or pa == annih1:
                table_a[(bs, fs)] = shift_n(tail)
            else:
                if to_left:
                    br = object.__new__(BSC)
                    br.result = bs
                    br.tail = S.One
                    bs_merged, n = br.merge_ind((a,))
                    tail_od = n * shift_n(tail)
                else:
                    bs_merged, n = BSC((a,)).merge_ind(bs)
                    tail_od = shift_n(n * tail, a_od)
                j, bp = next((j, bp) for j, bp in enumerate(bs_merged)
                             if bp.name == a_od.name)
                if bp == a_od:
                    table_od[((*bs_merged[:j], *bs_merged[j+1:]), fs)] = tail_od
                else:
                    nbpj = BosonPat(*bp.args[:2], bp.exps[0] - 1)
                    table_od[((*bs_merged[:j], nbpj, *bs_merged[j+1:]), fs)] = tail_od
        p_a = PF(table_a, new_npf) if (table_a or new_npf != 0) else PF0
        p_od = PF(table_od) if table_od else PF0
        return p_a, p_od

    def bt_swap(self, a: BosonOp, i, exps, to_left=False):
        yield S.One, a, exps
        if to_left:
            for j, e in enumerate(exps):
                if i ^ (j & 1) ^ (not self.dc and a.name == a_s.name) != a.is_annihilation:
                    for s, b, nes in self.bt_swap(self.odag(a), i, exps[:j], True):
                        yield (S.NegativeOne**int(a.is_annihilation) * s * e,
                               b, (*nes, e - 1, *exps[j+1:]))
        else:
            for j, e in enumerate(exps):
                ic = i ^ (j & 1)
                if ic ^ (not self.dc and a.name == a_s.name) != a.is_annihilation:
                    for s, b, nes in self.bt_swap(self.odag(a), not ic, exps[j+1:], False):
                        yield (S.NegativeOne**int(not a.is_annihilation) * s * e,
                               b, (*exps[:j], e - 1, *nes))

    def bd_swap(self, a: BosonOp, d: dict, to_left=True) -> dict:
        """
        Swaps the order of the operator a and the {tuple: PF} dict d.
        Returns a {BosonOp: {tuple: PF}} dict.
        """
        new_fdl = {a: defaultdict(list), self.odag(a): defaultdict(list)}
        for t, p in d.items():
            i, *exps = t
            for s, b, nes in self.bt_swap(a, i, exps, to_left=to_left):
                key = _tdz(i, nes)
                p_b, p_od = self.bp_swap(b, p, s, to_left)
                new_fdl[b][key].append(p_b)
                new_fdl[self.odag(b)][key].append(p_od)
        res_fd = dict()
        for k, dl in new_fdl.items():
            k_dict = {key: v for key, ps in dl.items() if (v := pf_add(*ps))}
            if k_dict:
                res_fd[k] = k_dict
        return res_fd

    def match_cop(self, bs: tuple):
        """
        Matches the boson operator tuple bs from a PF table key against the
        single-mode canonical operators defined in self.a.
        """
        *names, p = self.a[0].args[:3]
        b_exps = dict(zip(names, (0, 0)))
        bs_other = []
        for b in bs:
            if b.name in b_exps:
                b_exps[b.name] += b.exps[0] if b.args[1] else -b.exps[0]
            else:
                bs_other.append(b)

        es = tuple(b_exps[k] for k in names)
        if es[0] != es[1] * (-1)**p:
            raise ValueError('Failed to extract single-mode coupled operator product from the pattern')

        return (None if es[0] == 0 else bool(es[0] > 0)), abs(es[0]), tuple(bs_other)

    def tcp_swap(self, cop_a1, cop_exp, k, to_left=False, _head=(), _tail=S.One):
        """
        Swaps a single-mode weak-field coherent operator product (COP) power
        with a multimode exponent tuple k = (i, *exps).
        """
        if k == (None,):
            yield (None,), cop_exp, _tail
            return

        i, *exps = k
        cop_a1_b = None if cop_a1 is None else bool(cop_a1)
        cop_idx = 1 if cop_a1_b else 0

        if cop_a1_b is None or cop_exp == 0 or not exps:
            i0 = i ^ (len(_head) & 1) if not to_left else i
            combined = (_head + tuple(exps)) if not to_left else (tuple(exps) + _head)
            yield _tdz(i0, combined), cop_exp, _tail
            return

        if to_left:
            em = exps.pop()
            nj = i ^ (len(exps) & 1)

            if nj != cop_idx:
                for m1, m2, t in self.a[nj].swap_exps(em, cop_exp):
                    new_cop_exp = _e(m1)
                    new_b_exp = _e(m2)
                    new_head = (new_b_exp,) + _head
                    t_shift = mmn_shift(t, self.a[0], _pd((not nj, *_head))) if _head else t
                    new_tail = t_shift * _tail

                    if new_cop_exp == 0 or not exps:
                        yield _tdz(i, tuple(exps) + new_head), new_cop_exp, new_tail
                    else:
                        yield from self.tcp_swap(
                            cop_a1_b, new_cop_exp, (i, *exps),
                            to_left=True, _head=new_head, _tail=new_tail
                        )

            new_head = (em,) + _head
        else:
            i0 = i ^ (len(_head) & 1)
            em = exps.pop(0)

            if i != cop_idx:
                for m1, m2, t in self.a[cop_idx].swap_exps(cop_exp, em):
                    new_b_exp = _e(m1)
                    new_cop_exp = _e(m2)
                    new_head = _head + (new_b_exp,)
                    t_shift = mmn_shift(t, self.a[0], _pd((not i, *exps))) if exps else t
                    new_tail = t_shift * _tail

                    if new_cop_exp == 0 or not exps:
                        yield _tdz(i0, new_head + tuple(exps)), new_cop_exp, new_tail
                    else:
                        yield from self.tcp_swap(
                            cop_a1_b, new_cop_exp, (not i, *exps),
                            to_left=False, _head=new_head, _tail=new_tail
                        )

            new_head = _head + (em,)
            i = not i

        yield from self.tcp_swap(
            cop_a1_b, cop_exp, (i, *exps),
            to_left=to_left, _head=new_head, _tail=_tail
        )

    def _sm_expr(self, expr):
        """Simplifies number operators and mode count in scalar/coefficient expressions."""
        return expr.replace(
            lambda x: isinstance(x, (MultimodeNum, MultimodeNum2, MultimodeCOP)),
            lambda x: x.mode_expr
        ).subs({nm: 1, Symbol('n_m'): S.One})

    def _smk(self, k) -> PF:
        """Single-mode PF representation of a dict key (BosonOp or multimode tuple)."""
        if isinstance(k, BosonOp):
            return PF({((k,), ()): S.One})
        bs = []
        i, *exps = k
        if not exps:
            return PF1
        for e in exps:
            b1, b2 = Mul.make_args(self.a[i].mode_expr)
            bs.append(b1**e)
            bs.append(b2**e)
            i = not i
        b, t = BSC(bs).result_ind()
        return PF({(b, ()): t})

    def _mmn_only(self, expr: Expr) -> bool:
        if isinstance(expr, BosonPat):
            return expr.is_hermitian or not expr.has(*self.a[0].names)
        if isinstance(expr, (FermionPat, MultimodeNum, MultimodeNum2)):
            return True
        if isinstance(expr, MultimodeCOP):
            return False
        if hasattr(expr, 'args'):
            return all(self._mmn_only(arg) for arg in expr.args)
        return True

    def from_expr(self, val):
        """
        Converts an Expr, PF, dict, WMWrapper, or string/number into a WMWrapper.
        """
        from .pm_wrapper import PMWrapper
        if isinstance(val, PMWrapper):
            return val if val.dc == self.dc else PMWrapper(self, val.dict_repr, val.b_left)
        if isinstance(val, PF):
            return PMWrapper(self, val, None)
        if isinstance(val, dict):
            return PMWrapper(self, val, None)
        if isinstance(val, str) and val in self.init_ops:
            return self.init_ops[val]

        expr = sympify(val)
        if isinstance(expr, MultimodeCOP):
            return PMWrapper(self, {(bool(expr.annih1), 1): PF1}, None)

        weak_names = self.a[0].names
        if isinstance(expr, BosonOp) and expr.name in weak_names:
            return PMWrapper(self, {expr: {(None,): PF1}})

        if self._mmn_only(expr):
            return PMWrapper(
                self,
                {(None,): pf(expr, as_dict=None, ignore=(MultimodeNum, MultimodeNum2, MultimodeCOP))},
                None
            )

        if isinstance(expr, BosonPat) and expr.name in weak_names:
            pp, np = expr.num_repr()
            if isinstance(pp, BosonPat):
                if pp.exps[0] > 1:
                    raise ValueError('Solution parity unbalanced')
                b = BosonOp(*pp.args[:2])
                w = PMWrapper(self, {b: {(None,): PF({_vpk: np})}}, b_left=True)
                return w if w.b_left == (self.dc ^ w.parity) else w.flip()
            return PMWrapper(self, {(None,): PF({_vpk: np})}, None)

        if expr.has(MultimodeCOP) or any(
            getattr(b, 'name', None) in weak_names
            for b in expr.atoms(BosonOp, BosonPat)
            if not isinstance(b, BosonNum)
        ):
            if expr.is_Add:
                return self.d_add(*(self.from_expr(arg) for arg in expr.args))
            if expr.is_Mul:
                res = self.from_expr(expr.args[0])
                for arg in expr.args[1:]:
                    res = res * self.from_expr(arg)
                return res
            if expr.is_Pow:
                return self.from_expr(expr.base) ** expr.exp

        return PMWrapper(self, {(None,): pf(expr, as_dict=None)}, None)


wm_processors = (PMProcessor(False), PMProcessor())
