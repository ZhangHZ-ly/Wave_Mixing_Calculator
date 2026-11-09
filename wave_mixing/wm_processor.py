from typing import Any, List
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import _mexpand
from sympy.core.traversal import bottom_up
from sympy.core.exprtools import factor_terms
from sympy.polys.polytools import cancel
from .pattern_form import BSC, PF, pf, PF0, PF1, PE
from .pattern_form.op_patterns.boson_pattern import fac
from .pattern_form.op_patterns import BosonPat, BosonNum, FermionPat
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from .pattern_form.pattern_form import pf_add, pf_mul_ind, pf_pow_ind, _vpk
from .pattern_form.perturb_expander import gperm, DerivCache
from sympy.utilities.iterables import multiset_permutations
from .dsolve_pm import group_terms, _ps
from .multimode import (f_dict, MultimodeCOP, MultimodeNum, MultimodeNum2,
                        d_apply, t_mul, mmn_shift, _mps, _tdz, _pd)
from collections import defaultdict
from collections.abc import Sequence
from itertools import product
import warnings

__all__ = ['WMProcessor', 'INIT_OPS', 'a_p', 'a_s', 'a_i', 't_', '_g', '_ps', 'nlc', 'nm']

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
    if not p.table and p.npf == 0:
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

# Dummy symbols for:
# the MultimodeNum of the signal field
# the MultimodeNum2 of the process
# the BosonNum of the driving field
# the total number of modes
_n1, _n2, _nd = symbols('n_1 n_2 n_d', Dummy=True, nonnegative=True)
# Field names: a0 - pump (high frequency), a1, a2 - signal and idler (low frequency)
a_p = BosonOp('a_0')
a_s = BosonOp('a_1')
a_i = BosonOp('a_2')
_g = Dummy('g', real=True)  # Dummy symbol for the parametric gain

# Time / nonlinear coefficient symbols: t, χ
t_ = Dummy('t', real=True)
nlc = Symbol('χ', real=True, nonnegative=True)
nm = Symbol('n_m', nonnegative=True)

# Singleton initial operator representations at t=0 for down-conversion (dc=True)
# and frequency conversion / state transfer (dc=False):
INIT_OPS = {
    True: {
        'a_0': PF({((a_p,), ()): S.One}),
        'a_1': f_dict({a_s: {(None,): PF1}}, b_left=True),
        'a_2': f_dict({a_i: {(None,): PF1}}, b_left=True),
        'a_p': PF({((a_p,), ()): S.One}),
        'a_s': f_dict({a_s: {(None,): PF1}}, b_left=True),
        'a_i': f_dict({a_i: {(None,): PF1}}, b_left=True),
        'A': {(True, 1): PF1},
        'N_t': PF({_vpk: MultimodeNum2(a_s.name, a_i.name, S.Zero)}),
    },
    False: {
        'a_0': f_dict({a_p: {(None,): PF1}}, b_left=True),
        'a_1': f_dict({a_s: {(None,): PF1}}, b_left=True),
        'a_2': PF({((a_i,), ()): S.One}),
        'a_p': PF({((a_i,), ()): S.One}),
        'a_s': f_dict({a_p: {(None,): PF1}}, b_left=True),
        'a_i': f_dict({a_s: {(None,): PF1}}, b_left=True),
        'A': {(True, 1): PF1},
        'N_t': PF({_vpk: MultimodeNum2(a_p.name, a_s.name, S.One)}),
    },
}

class WMProcessor:
    """
    Base processor for quantum multimode parametric wave-mixing processes (TWM, FWM).

    Ordering principle: The strong field and commutative parameters
    are extracted to the end, ordered in pat-form;
    the multimode patterns are left-multiplied to multimode number operators;
    the annihilation operator of single weak field with larger index number
    is left-multiplied to the rest of the expression (multi-mode part),
    and the other operators are ordered as such that the commutation
    between singlemode and multimode operators is avoided.
    """

    def __init__(self, dc=True):
        self.dc = bool(dc)
        self.nd = BosonNum(a_p.name) if dc else BosonNum(a_i.name)
        self.a = ((MultimodeCOP(a_s.name, a_i.name, annih1=0),
                   MultimodeCOP(a_s.name, a_i.name))
                  if dc else (MultimodeCOP(a_p.name, a_s.name, 1, 0),
                              MultimodeCOP(a_p.name, a_s.name, 1)))
        self.mmn2 = MultimodeNum2(*self.a[0].args[:3])
        self.sol = []
        self.pe = PE({nlc: S.One})
        self.pe.register(MultimodeNum)
        self.pe.register(MultimodeNum2)
        self.pe.register(MultimodeCOP)

    def odag(self, b: BosonOp):
        """
        Returns the partner single-mode operator
        according to the parity rule of self.a[0].

        If the operator does not belong to the primary modes,
        returns the operator itself.
        """
        n1, n2, p = self.a[0].args[:3]
        if not isinstance(b, BosonOp):
            raise ValueError(f'{b} is not a legal f_dict key for this instance')
        if b.name == n1:
            return BosonOp(n2, b.is_annihilation==p)
        if b.name == n2:
            return BosonOp(n1, b.is_annihilation==p)
        raise ValueError(f'{b} is not a legal f_dict key for this instance')

    def f_parity(self, dk: f_dict | Sequence[BosonOp]):
        """
        Determines the parity class (0 or 1) of an f_dict.
        Two f_dicts can be multiplied into a dict iff they have opposite parities
        (i.e. each key b1 in p1 satisfies b1 == b2.adjoint() or b1 == self._od(b2).adjoint()).
        Raises ValueError if the f_dict itself contains keys of mixed parities.
        """
        if not dk:
            return None
        if isinstance(dk, f_dict):
            return self.f_parity(tuple(dk))
        b0 = dk[0]
        b0_od = self.odag(b0)
        for b in dk[1:]:
            if not (b == b0 or b == b0_od):
                raise ValueError('Inconsistent BosonOp keys within f_dict')
        n1, _, p = self.a[0].args[:3]
        if b0.name == n1:
            return b0.is_annihilation
        return b0.is_annihilation ^ p

    def expand(self, item, order, sub_g=False):
        """
        Expands the item to the given perturbative order
        
        :param sub_g: whether to substitute the time with g/gc
        """
        if isinstance(item, Expr):
            res = self.pe.expand_expr(item, order)[0]
        elif isinstance(item, PF):
            res = self.pe.expand_pf(item, order)
        elif isinstance(item, f_dict):
            res = f_dict(b_left=item.b_left)
            for k, v in item.items():
                res[k] = self.expand(v, order, sub_g=sub_g)
            return res
        elif isinstance(item, dict):
            res = dict()
            for k, v in item.items():
                res[k] = self.expand(v, order, sub_g=sub_g)
            return res
        if sub_g and hasattr(self, 'gc'):
            return d_apply(res, lambda t: t.subs({t_: _g/self.gc}))
        return res

    def expand_func_d(self, func, order, *ds, sub_g=False):
        if not ds:
            return func()

        ord_lims, flo = self.pe.handle_order_limits(func, len(ds))
        x0_list = []
        perturb_list = []
        for j, (d, ol) in enumerate(zip(ds, ord_lims)):
            if isinstance(d, f_dict):
                raise ValueError('The expression does not support expand-with-func')
            if isinstance(d, dict):
                pass
            elif isinstance(d, PF):
                d = {(None,): d}
            elif isinstance(d, Expr) and not self._mmn_only(d):
                d = {(None,): pf(d)}
            else:
                raise ValueError('The expression does not support expand-with-func')

            odt = defaultdict(dict)
            for k, v in d.items():
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
                    perturb_list.append((j, od - ol, td))

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
            return dict()

        result_dicts = []
        if fl != 0 and fl is not S.Zero:
            result_dicts.append({(None,): PF({_vpk: fl})})
        dim = len(perturb_list)
        dfc = DerivCache(f_sym, dummies)

        def rec(i, remaining, prefix, current_sum):
            if i == dim:
                m = sum(prefix)
                if m == 0:
                    return
                deriv_orders = [0] * len(ds)
                for r, cnt in enumerate(prefix):
                    deriv_orders[perturb_list[r][0]] += cnt

                deriv_val = dfc.get(tuple(deriv_orders)).subs(subs_dict)
                if deriv_val == 0:
                    return
                c_dict = {(None,): PF({_vpk: deriv_val / fac(m)})}

                l = sum(([r] * cnt for r, cnt in enumerate(prefix)), [])
                sym_dicts = []
                for p in multiset_permutations(l):
                    prod_d = perturb_list[p[0]][2]
                    for next_r in p[1:]:
                        prod_d = self.e_mul(prod_d, perturb_list[next_r][2])
                    sym_dicts.append(prod_d)
                result_dicts.append(self.e_mul(self.d_add(*sym_dicts), c_dict))
                return

            curr_w = perturb_list[i][1]
            max_count = remaining // curr_w
            for x in range(max_count + 1):
                prefix.append(x)
                rec(i + 1, remaining - x * curr_w, prefix, current_sum + x * curr_w)
                del prefix[-1]

        rec(0, nol, [], 0)
        if sub_g and hasattr(self, 'gc'):
            return d_apply(self.d_add(*result_dicts), lambda t: t.subs({t_: _g/self.gc}))
        return self.d_add(*result_dicts)

    def n_simp(self, expr):
        n1, n2 = self.a[0].names
        def mmn_d(n, d):
            if n == n1:
                return _n1 + d*nm
            if n == n2:
                if self.dc:
                    return _n2 - _n1 + (d-1) * nm
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
        hyperbolic/triangular functions
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

    def to_expr(self, x):
        """
        Converts the class representation into expression.

        The perturbative solutions to the quantum parametric processes
        are saved in dicts. The corresponding expression is the sum over
        the products key * value for each key of the dict.
        The values of dict can also be dict, whose corresponding expressions
        are given by the same regulation.

        The multimode factor is represented by a tuple,
        similar to the pattern exponents.
        """
        if isinstance(x, Expr):
            return x
        if isinstance(x, tuple):
            i, *exps = x
            factors = []
            for j in exps:
                factors.append(self.a[i]**j)
                i = not i
            return Mul(*factors)
        if isinstance(x, PF):
            return x.expr(False)
        if isinstance(x, dict):
            b_left = getattr(x, 'b_left', True)
            if b_left:
                return Add(*(self.to_expr(b)*self.to_expr(e) for b, e in x.items()))
            return Add(*(self.to_expr(e)*self.to_expr(b) for b, e in x.items()))
        raise ValueError('Got unexpected format')

    def d_adjoint(self, d: dict):
        """
        Takes adjoint for the dict representation of field solutions.
        The direction of the single-mode weak-field operator is reversed upon conjugation.
        """
        if isinstance(d, f_dict):
            new = f_dict(b_left=not d.b_left)
        else:
            new = dict()
            
        for k, e in d.items():
            if isinstance(k, tuple):
                i, *exps = k
                if exps:
                    i = i ^ (len(exps)&1)
                    exps.reverse()
                pd = _pd(k)
                new[(i, *exps)] = e.adjoint().n_apply(lambda t: mmn_shift(t, self.a[1], pd))
            else:
                new[Dagger(k)] = self.d_adjoint(e)
        return new

    def d_add(self, *ds):
        """
        Adds multiple dicts together, collecting the same keys
        """
        ds = [d for d in ds if d]
        if not ds:
            return dict()
        if len(ds) == 1:
            return ds[0]
        result = defaultdict(list)
        if all(isinstance(d, f_dict) for d in ds):
            parities = {self.f_parity(d) for d in ds if d}
            if len(parities) > 1:
                raise ValueError("Cannot add single-mode weak-field operators"
                    " of different parities")
            b_left = ds[0].b_left
            for d in ds:
                if d.b_left == b_left:
                    for k, v in d.items():
                        result[k].append(v)
                if d.b_left != b_left:
                    for k, v in d.items():
                        for nk, nv in self.bd_swap(k, v, to_left=b_left).items():
                            result[nk].append(nv)
            return f_dict({k: self.d_add(*vs) for k, vs in result.items()}, b_left=b_left)
        if any(isinstance(d, f_dict) for d in ds):
            raise ValueError("Cannot add expressions with and without"
                " single-mode weak-field operators")
        for d in ds:
            for k, v in d.items():
                result[k].append(v)
        return {k: pf_add(*vs) for k, vs in result.items()}

    def d_mul(self, d0, d1, d2, mid=None):
        if not d1 or not d2:
            return
        weak_names = set(self.a[0].names)
        shift_n = lambda expr, pd: mmn_shift(expr, self.a[1], pd)

        for (k2, e2) in d2.items():
            pd = _pd(k2)
            for (k1, e1) in d1.items():
                has_weak_pair = False
                if e1.table:
                    for bs, fs in e1.table.keys():
                        bw = tuple(b for b in bs if b.name in weak_names)
                        if bw:
                            try:
                                _, cop_exp, _ = self.match_cop(bw)
                                if cop_exp > 0:
                                    has_weak_pair = True
                                    break
                            except ValueError:
                                pass

                if not has_weak_pair:
                    k = t_mul(k1, k2) if isinstance(mid, MultimodeNum) else t_mul(k1, k2, mid)
                    factor_mid = lambda t: shift_n(mid * t if isinstance(mid, MultimodeNum) else t, pd)
                    new = e1.n_apply(factor_mid)
                    term_prod = pf_mul_ind(new, e2)
                    if k in d0:
                        d0[k] = pf_add(d0[k], term_prod)
                    else:
                        d0[k] = term_prod
                else:
                    if e1.npf != 0:
                        k = t_mul(k1, k2) if isinstance(mid, MultimodeNum) else t_mul(k1, k2, mid)
                        c_npf = shift_n(mid * e1.npf if isinstance(mid, MultimodeNum) else e1.npf, pd)
                        p_npf = pf_mul_ind(PF({}, c_npf), e2)
                        if k in d0:
                            d0[k] = pf_add(d0[k], p_npf)
                        else:
                            d0[k] = p_npf

                    for (bs, fs), tail in e1.table.items():
                        c_tail = shift_n(mid * tail if isinstance(mid, MultimodeNum) else tail, pd)
                        try:
                            cop_a1, cop_exp, bs_other = self.match_cop(bs)
                        except ValueError:
                            cop_a1, cop_exp, bs_other = None, 0, bs

                        if cop_exp == 0 or cop_a1 is None:
                            k = t_mul(k1, k2) if isinstance(mid, MultimodeNum) else t_mul(k1, k2, mid)
                            p_term = pf_mul_ind(PF({(bs, fs): c_tail}, S.Zero), e2)
                            if k in d0:
                                d0[k] = pf_add(d0[k], p_term)
                            else:
                                d0[k] = p_term
                        else:
                            for k_sub, new_m, P_num in self.tcp_swap(cop_a1, cop_exp, k2, to_left=False):
                                k_tot = t_mul(k1, k_sub) if isinstance(mid, MultimodeNum) else t_mul(k1, k_sub, mid)
                                if new_m == 0:
                                    new_bw = ()
                                else:
                                    cop_tgt = self.a[1 if cop_a1 else 0]
                                    op1 = BosonOp(cop_tgt.names[0], cop_tgt.annih1)
                                    op2 = BosonOp(cop_tgt.names[1], cop_tgt.annih2)
                                    if new_m == 1:
                                        new_bw = (op1, op2)
                                    else:
                                        new_bw = (op1**new_m, op2**new_m)

                                new_bs = bs_other + new_bw
                                final_tail = c_tail * P_num
                                p_term = pf_mul_ind(PF({(new_bs, fs): final_tail}, S.Zero), e2)
                                if k_tot in d0:
                                    d0[k_tot] = pf_add(d0[k_tot], p_term)
                                else:
                                    d0[k_tot] = p_term

    def e_mul(self, e1: dict, e2: dict):
        """
        Multiplies two dict representations of operators.
        
        if both dict contain BosonOps, this will generate an intermediate dict
        including MultimodeNums (to be further collected as MultimodeNum2)
        
        By default, single-mode weak-field operators are assumed to be
        at the edges. If a multiplication causes a single-mode operator
        to be trapped in the middle, bdswap is automatically invoked
        to swap it to the correct boundary.
        """
        if not e1 or not e2:
            return dict()
        is_f1 = isinstance(e1, f_dict)
        is_f2 = isinstance(e2, f_dict)
        result = dict()
        if is_f1 and is_f2:
            if self.f_parity(e1) == self.f_parity(e2):
                raise ValueError('Solution parity unbalanced')
            if e1.b_left:
                new_p1 = f_dict(b_left=False)
                for b, d in e1.items():
                    for nb, nd in self.bd_swap(b, d, to_left=False).items():
                        if nd: new_p1[nb] = self.d_add(new_p1.get(nb, dict()), nd)
                e1 = new_p1
            if not e2.b_left:
                new_p2 = f_dict(b_left=True)
                for b, d in e2.items():
                    for nb, nd in self.bd_swap(b, d, to_left=True).items():
                        if nd: new_p2[nb] = self.d_add(new_p2.get(nb, dict()), nd)
                e2 = new_p2
            for (b1, d1), (b2, d2) in product(e1.items(), e2.items()):
                if b1 == Dagger(b2):
                    mn = (MultimodeNum(b1.name, S.One) if b1.is_annihilation
                          else MultimodeNum(b1.name))
                    self.d_mul(result, d1, d2, mn)
                elif b1.compare(b2) == 1:
                    self.d_mul(result, d1, d2, b2.is_annihilation)
                else:
                    self.d_mul(result, d1, d2, b1.is_annihilation)
        elif is_f1:
            if e1.b_left:
                result = f_dict(b_left=True)
                for b, d in e1.items():
                    result[b] = dict()
                    self.d_mul(result[b], d, e2)
            else:
                result = f_dict(b_left=False)
                for b, d1 in e1.items():
                    for nb, p2_prime in self.bd_swap(b, e2, to_left=False).items():
                        self.d_mul(result.setdefault(nb, dict()), d1, p2_prime)
        elif is_f2:
            if not e2.b_left:
                result = f_dict(b_left=False)
                for b, d in e2.items():
                    result[b] = dict()
                    self.d_mul(result[b], e1, d)
            else:
                result = f_dict(b_left=True)
                for b, d2 in e2.items():
                    for nb, p1_prime in self.bd_swap(b, e1, to_left=True).items():
                        self.d_mul(result.setdefault(nb, dict()), p1_prime, d2)
        else:
            self.d_mul(result, e1, e2)

        return result
    
    def f_to_dict(self, fd: f_dict) -> dict:
        """
        Moves the key (BosonOp) into between tuple_key and PF,
        converting an f_dict into a dict with PF-values.

        - If fd.b_left is True: b is to the left of t * p;
        swaps b past t to the right using bt_swap.

        - If fd.b_left is False: b is to the right of t * p;
        swaps b past p to the left using bp_swap.
        """
        res = dict()
        if fd.b_left:
            for b, d in fd.items():
                for t, p in d.items():
                    i, *exps = t
                    for s, nb, nes in self.bt_swap(b, i, exps, to_left=False):
                        key = _tdz(i, nes)
                        pf_nb = pf(nb, as_dict=None)
                        val = s * pf_mul_ind(pf_nb, p)
                        if key in res:
                            res[key] = pf_add(res[key], val)
                        else:
                            res[key] = val
        else:
            for b, d in fd.items():
                for t, p in d.items():
                    p_b, p_od = self.bp_swap(b, p, 1, to_left=True)
                    val = PF0
                    if p_b.table or p_b.npf != 0:
                        val = pf_add(val, pf_mul_ind(pf(b, as_dict=None), p_b))
                    if p_od.table or p_od.npf != 0:
                        val = pf_add(val, pf_mul_ind(pf(self.odag(b), as_dict=None), p_od))
                    if val.table or val.npf != 0:
                        if t in res:
                            res[t] = pf_add(res[t], val)
                        else:
                            res[t] = val
        return res

    def f_mul(self, e1, e2):
        if not e1 or not e2:
            return dict()
        if not isinstance(e1, f_dict) or not isinstance(e2, f_dict):
            return self.e_mul(e1, e2)
        if self.f_parity(e1) == self.f_parity(e2):
            raise ValueError('Solution parity unbalanced')
        d1 = self.f_to_dict(e1)
        d2 = self.f_to_dict(e2)
        result = dict()
        self.d_mul(result, d1, d2)
        return result

    def bp_swap(self, a: BosonOp, p: PF, s, to_left=True):
        """
        Swaps the single-mode boson operator (a)
        and given PF (p) with Multimode operators

        :param s: the sign factor for the swap, either 1 or -1
        :param to_left: whether the operator is swapped to the left of the PF
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
                    nbpj = BosonPat(*bp.args[:2], bp.exps[0]-1)
                    table_od[((*bs_merged[:j], nbpj, *bs_merged[j+1:]), fs)] = tail_od
        p_a = PF(table_a, new_npf) if (table_a or new_npf != 0) else PF0
        p_od = PF(table_od) if table_od else PF0
        return p_a, p_od

    def bt_swap(self, a: BosonOp, i, exps, to_left=False):
        yield S.One, a, exps
        if to_left:
            for j, e in enumerate(exps):
                if i ^ (j & 1) ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                    for s, b, nes in self.bt_swap(self.odag(a), i, exps[:j], True):
                        yield (S.NegativeOne**int(a.is_annihilation)*s*e,
                               b, (*nes, e-1, *exps[j+1:]))
        else:
            for j, e in enumerate(exps):
                ic = i ^ (j & 1)
                if ic ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                    for s, b, nes in self.bt_swap(self.odag(a), not ic, exps[j+1:], False):
                        yield (S.NegativeOne**int(not a.is_annihilation)*s*e,
                               b, (*exps[:j], e-1, *nes))

    def bd_swap(self, a: BosonOp, d, to_left=True):
        """
        Swaps the order of the operator a and the dict d
        The direction of the swap (for a) is determined by the to_left parameter.
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
            k_dict = dict()
            for key, ps in dl.items():
                v = pf_add(*ps)
                if v.table or v.npf != 0:
                    k_dict[key] = v
            if k_dict:
                res_fd[k] = k_dict
        return f_dict(res_fd, b_left=to_left)

    def match_cop(self, bs: tuple):
        """
        Matches the boson operator tuple bs from a PF table key against the
        single-mode canonical operators defined in self.a.

        :param bs: Boson operator tuple from a PF table key (bs, fs).
        :return: (annih1, exp_cop, bs_other) if bs can be fully separated
                 into a single-mode COP and other field operators;
                 otherwise raise an exception.
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
            raise ValueError('Failed to extract single-mode'
                             ' coupled operator product from the pattern')

        return (None if es[0] == 0 else bool(es[0] > 0)), abs(es[0]), tuple(bs_other)

    def tcp_swap(self, cop_a1, cop_exp, k, to_left=False, _head=(), _tail=S.One):
        """
        Swaps a single-mode weak-field coherent operator product (COP) power cop(†)^ce
        with a multimode exponent tuple k = (i, *exps).

        Yields (k_new, exp_new, tail):
        - k_new: resulting exponent tuple
        - exp_new: new exp on the single-mode coupled operator product
        - tail: single-mode number operator polynomial in BosonNum
                right-multiplied to the yielded term

        :param cop_a1: Whether the left operator of the pair is annihilation
        :param cop_exp: The exponent of the power on the COP
        :param k: The multimode exponent tuple
        :param to_left: Whether to move the COP to the left-hand-side of the tuple
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

    def _smk(self, k):
        """Single mode key"""
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

    def to_single_mode(self, obj):
        """
        Converts multimode operator representations (MultimodeCOP, MultimodeNum, MultimodeNum2)
        into single-mode boson operators (mode_expr) with n_m = 1,
        and evaluates into a unified PatternForm (PF).

        :param obj: Expr, PF, dict, or f_dict.
        :return: PF
        """
        if isinstance(obj, (dict, f_dict)):
            b_left = getattr(obj, 'b_left', True)
            if b_left:
                return pf_add(*(pf_mul_ind(self._smk(k), self.to_single_mode(v))
                             for k, v in obj.items()))
            return pf_add(*(pf_mul_ind(self.to_single_mode(v), self._smk(k))
                         for k, v in obj.items()))
        elif isinstance(obj, PF):
            return obj.n_apply(lambda t: self._sm_expr(t), on_npf=True)
        elif isinstance(obj, Expr):
            return pf(self._sm_expr(obj), as_dict=None)

        return self.to_single_mode(sympify(obj))

    def _mmn_only(self, expr: Expr):
        if isinstance(expr, BosonPat):
            return expr.is_hermitian or not expr.has(*self.a[0].names)
        if isinstance(expr, (FermionPat, MultimodeNum, MultimodeNum2)):
            return True
        if isinstance(expr, MultimodeCOP):
            return False
        if hasattr(expr, 'args'):
            return all(self._mmn_only(arg) for arg in expr.args)
        return True

    def to_dict(self, val):
        """
        Converts a PF, scalar, or single-mode expression into a multimode dict representation:
        - If already a dict or f_dict: returned as is.
        - If PF: returns {(None,): val}.
        - If Expr / number / symbol: converts to representation first, then lifts to {(None,): pf}.
        """
        if isinstance(val, (dict, f_dict)):
            return val
        if isinstance(val, PF):
            return {(None,): val}
        rep = self.from_expr(val)
        if isinstance(rep, (dict, f_dict)):
            return rep
        return {(None,): rep}

    def op_mul(self, v1, v2, single_mode: bool = False):
        """Multiply two resolved operator representations (PF, dict, or f_dict)."""
        if isinstance(v1, PF) and isinstance(v2, PF):
            return pf_mul_ind(v1, v2)
        d1 = {(None,): v1} if isinstance(v1, PF) else v1
        d2 = {(None,): v2} if isinstance(v2, PF) else v2
        if single_mode:
            return self.f_mul(d1, d2)
        return self.e_mul(d1, d2)

    def op_add(self, v1, v2):
        """Add two resolved operator representations (PF, dict, or f_dict)."""
        if isinstance(v1, PF) and isinstance(v2, PF):
            return pf_add(v1, v2)
        d1 = self.to_dict(v1)
        d2 = self.to_dict(v2)
        return self.d_add(d1, d2)

    def op_neg(self, v):
        """Negate an operator representation."""
        if isinstance(v, PF):
            return -v
        return d_apply(v, lambda t: -t)

    def op_div(self, v1, v2):
        """Divide an operator representation by a scalar PF representation."""
        if isinstance(v2, PF) and v2.npf == 0 and set(v2.table.keys()).issubset({_vpk}):
            denom = v2.table.get(_vpk, S.One)
            inv_pf = PF({_vpk: denom ** S.NegativeOne})
            return self.op_mul(v1, inv_pf)
        raise ValueError("Division is only supported by scalar or number-operator expressions.")

    def op_pow(self, v, n):
        """Exponentiate a resolved operator representation (PF, dict, or f_dict)."""
        if isinstance(n, (int, Integer)):
            if n >= 0:
                n_int = int(n)
                if isinstance(v, PF):
                    if v.npf == 0 and set(v.table.keys()).issubset({_vpk}):
                        base_expr = v.table.get(_vpk, S.Zero)
                        return PF({_vpk: base_expr ** n_int})
                    try:
                        return pf_pow_ind(v, Integer(n_int))
                    except Exception:
                        return PF(npf=Pow(v.expr(), n_int))
                if isinstance(v, dict) and not isinstance(v, f_dict):
                    if n_int == 0:
                        return {(None,): PF1}
                    if n_int == 1:
                        return v
                    res = None
                    base = v
                    exp_n = n_int
                    while exp_n > 0:
                        if exp_n & 1:
                            res = base if res is None else self.e_mul(res, base)
                        exp_n >>= 1
                        if exp_n > 0:
                            base = self.e_mul(base, base)
                    return res
        if isinstance(v, PF):
            if v.npf == 0 and set(v.table.keys()).issubset({_vpk}):
                base_expr = v.table.get(_vpk, S.Zero)
                return PF({_vpk: base_expr ** sympify(n)})
            return PF(npf=Pow(v.expr(), sympify(n)))
        if isinstance(v, dict) and not isinstance(v, f_dict):
            order = getattr(self, '_evolved_order', None)
            if order is None:
                order = 1
            return self.expand_func_d(lambda x: x ** sympify(n), order, v)
        raise ValueError("Exponentiation for this operator type only supports non-negative integers or expanded multimode dicts.")

    def op_prod(self, factors: List[Any], single_mode: bool = False):
        """
        Evaluate a product chain of factors across mixed representations:
        1. Checks parity balance of all f_dict factors.
        2. Merges adjacent PF and dict factors first.
        3. Prioritizes multiplying complementary f_dict pairs that cancel into dict.
        """
        if not factors:
            return PF1
        if len(factors) == 1:
            return factors[0]

        items = list(factors)
        f_list = [f for f in items if isinstance(f, f_dict)]

        if f_list:
            parities = [self.f_parity(f) for f in f_list]
            count_0 = sum(1 for p in parities if p == 0)
            count_1 = sum(1 for p in parities if p == 1)
            if abs(count_0 - count_1) > 1:
                raise ValueError('Cannot use multimode operators to represent the product')

        def _merge_non_f(seq):
            if len(seq) <= 1:
                return seq
            m_pfs = []
            for item in seq:
                if m_pfs and isinstance(m_pfs[-1], PF) and isinstance(item, PF):
                    m_pfs[-1] = self.op_mul(m_pfs[-1], item, single_mode=single_mode)
                else:
                    m_pfs.append(item)
            merged = []
            for item in m_pfs:
                if merged and not isinstance(merged[-1], f_dict) and not isinstance(item, f_dict):
                    merged[-1] = self.op_mul(merged[-1], item, single_mode=single_mode)
                else:
                    merged.append(item)
            return merged

        items = _merge_non_f(items)

        if f_list:
            while True:
                f_pos = [
                    (idx, self.f_parity(item))
                    for idx, item in enumerate(items)
                    if isinstance(item, f_dict)
                ]
                candidates = []
                for k in range(len(f_pos) - 1):
                    idx_l, p_l = f_pos[k]
                    idx_r, p_r = f_pos[k + 1]
                    if p_l is not None and p_r is not None and p_l != p_r:
                        f_l = items[idx_l]
                        f_r = items[idx_r]
                        swaps = int(bool(f_l.b_left)) + int(not bool(f_r.b_left))
                        candidates.append((idx_r - idx_l, swaps, idx_l, idx_r))
                if not candidates:
                    break
                _, _, best_l, best_r = min(candidates)
                sub_val = items[best_l]
                for step_idx in range(best_l + 1, best_r + 1):
                    sub_val = self.op_mul(sub_val, items[step_idx], single_mode=single_mode)
                items[best_l : best_r + 1] = [sub_val]
                items = _merge_non_f(items)

        res = items[0]
        for item in items[1:]:
            res = self.op_mul(res, item, single_mode=single_mode)
        return res

    def from_expr(self, val):
        """
        Converts an Expr, PF, dict, f_dict, or string/number into the standard
        wave-mixing representation (PF, multimode dict, or weak-field f_dict).
        """
        if isinstance(val, (PF, dict, f_dict)):
            return val
        if val in INIT_OPS[self.dc]:
            return INIT_OPS[self.dc][val]
        expr = sympify(val)

        if isinstance(expr, MultimodeCOP):
            return {(bool(expr.annih1), 1): PF1}

        weak_names = self.a[0].names
        if isinstance(expr, BosonOp) and expr.name in weak_names:
            return f_dict({expr: {(None,): PF1}}, b_left=True)

        if self._mmn_only(expr):
            return pf(expr, as_dict=None, ignore=(MultimodeNum, MultimodeNum2, MultimodeCOP))
        
        if isinstance(expr, BosonPat) and expr.name in weak_names:
            pp, np = expr.num_repr()
            if isinstance(pp, BosonPat):
                b = BosonOp(*pp.args[:2])
                db = f_dict({b: {(None,): PF1}}, b_left=True)
                res = PF1
                for _ in range(pp.exps[0] -1):
                    res = self.op_mul(res, db)
                res = self.op_mul(res, f_dict({b: {(None,): PF({_vpk: np})}}))
                return res
            return PF({_vpk: np})

        if expr.has(MultimodeCOP) or any(
            getattr(b, 'name', None) in weak_names
            for b in expr.atoms(BosonOp, BosonPat)
            if not isinstance(b, BosonNum)
        ):
            if expr.is_Add:
                res = self.from_expr(expr.args[0])
                for arg in expr.args[1:]:
                    res = self.op_add(res, self.from_expr(arg))
                return res
            if expr.is_Mul:
                factors = [self.from_expr(arg) for arg in expr.args]
                return self.op_prod(factors)
            if expr.is_Pow:
                base = self.from_expr(expr.base)
                return self.op_pow(base, expr.exp)

        return pf(expr, as_dict=None)

    def op_adjoint(self, v):
        """Takes the adjoint of an operator representation."""
        if isinstance(v, PF):
            return v.adjoint(rem=False)
        return self.d_adjoint(v)

    def _combine_tcp_vac(self, k, bs_weak):
        """
        Combines weak-field BosonPat operators with a multimode exponent tuple k.
        Yields (s, new_k) where s is a sign/scalar multiplier.

        When weak-field operators are on the right (acting on vacuum |0>):
        1. Annihilation COP (b_1^m): b_1^m |0> = 0, so vacuum expectation is zero.
        2. Creation COP (b_0^m): b_0^m is converted to multimode B_0^m by appending to k.
           By U(n_m) spherical symmetry, the extra factor introduced by converting
           single-mode b_0^m to multimode B_0^m is exactly:
               C(m, _nm) = binomial(_nm + m - 1, m)
           Thus we yield multiplier s = 1 / C(m, _nm).
        3. Unpaired operators: if annihilation on the right, b |0> = 0;
           otherwise bt_swap swaps through k.
        """
        if not bs_weak:
            yield S.One, k
            return

        try:
            cop_a1, cop_exp, bs_other = self.match_cop(bs_weak)
        except ValueError:
            cop_a1, cop_exp, bs_other = None, S.Zero, bs_weak

        if cop_exp > 0 and not bs_other:
            # Annihilation COP on the right gives 0 on vacuum
            if cop_a1:
                return

            # Creation COP (cop_a1 is False):
            new_k = k
            for _ in range(cop_exp):
                new_k = t_mul(new_k, (None,), 0)
            s_comb = fac(cop_exp) / gperm(nm + cop_exp - 1, cop_exp)
            yield (s_comb, new_k)
            return

        if len(bs_weak) == 1:
            b = bs_weak[0]
            # Annihilation operator on the right gives 0 on vacuum
            if b.left_base.is_annihilation:
                return
            b_op = BosonOp(b.name, b.left_base.is_annihilation)
            if k == (None,):
                return
            else:
                i, *exps = k
                for s, _, nes in self.bt_swap(b_op, i, exps, to_left=True):
                    yield s, _tdz(i, nes)
            return

        yield S.One, k

    def eval_vacuum(self, obj, vac_modes):
        """
        Evaluates the partial or full expectation value of an operator expression
        under vacuum input assertions for the specified mode(s).

        Uses MultimodeCOP.swap_exps to exchange powers of multimode canonical
        operators and eagerly collapses generated MultimodeCOP, MultimodeNum,
        and MultimodeNum2 operators using the active vacuum mode presets.
        For modes outside the weak-field pair, projects PatternForm values
        via vac_pf directly.

        :param obj: Expr, PF, dict, or f_dict.
        :param vac_modes: Iterable of mode names / BosonOps asserted to be in vacuum.
        :return: Collapsed PF, dict, or f_dict representation.
        """
        vac_sets = set(), set()
        weak_names = self.a[0].names
        for m in (vac_modes or ()):
            m = sympify(m)
            if not m.is_Symbol and hasattr(m, 'name'):
                m = m.name
            vac_sets[m in weak_names].add(m)
        if not vac_sets[0] and not vac_sets[1]:
            return obj

        def _collapse_mmn(name, displacement):
            if name in vac_sets[1]:
                return displacement * nm
            return MultimodeNum(name, displacement)

        def _collapse_mmn2(n1, n2, p):
            in1 = n1 in vac_sets[1]
            in2 = n2 in vac_sets[1]
            if not in1 and not in2:
                return MultimodeNum2(n1, n2, p)
            if not p:
                if in1 and in2:
                    return nm
                n_surv = n2 if in1 else n1
                return MultimodeNum(n_surv, 1)
            else:
                if in1 and in2:
                    return S.Zero
                if in1:
                    return MultimodeNum(n2)
                return -MultimodeNum(n1)
            
        _collapse_num = (lambda expr: sympify(expr).replace(MultimodeNum, _collapse_mmn)
                         .replace(MultimodeNum2, _collapse_mmn2))

        def _vac_pf(pf: PF) -> PF:
            vp = vac_pf(pf, vac_sets[0] | vac_sets[1])
            if not vp.table and vp.npf == 0:
                return PF0
            applied = vp.n_apply(_collapse_num, on_npf=True)
            clean_table = {k: v for k, v in applied.table.items() if v != 0}
            return PF(clean_table, applied.npf) if (clean_table or applied.npf != 0) else PF0

        tuple_cache = {}

        def _vac_tuple(k: tuple):
            if k in tuple_cache:
                return tuple_cache[k]
            if len(k) == 1:
                tuple_cache[k] = S.One
                return S.One
            if not self.dc and len(vac_sets[1]) == 2:
                tuple_cache[k] = S.Zero
                return S.Zero

            if self.dc:
                ann_idx = True
            elif a_s.name in vac_sets[1]:
                ann_idx = False
            else:
                ann_idx = True

            i, *exps = k
            if bool(i) != ann_idx or (len(exps) & 1) or sum(exps[0::2]) != sum(exps[1::2]):
                tuple_cache[k] = S.Zero
                return S.Zero

            B = self.a[ann_idx]
            exps = list(exps)
            acc = S.One
            while exps:
                c_r = exps.pop()
                a_r = exps.pop()
                if a_r > c_r:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                swapped = B.swap_exps(a_r, c_r, order=a_r, left_p=True)
                step_terms = []
                for term in Add.make_args(swapped):
                    if term.has(B):
                        continue
                    num_factors = [
                        f for f in Mul.make_args(term)
                        if not isinstance(f, MultimodeCOP)
                        and not (f.is_Pow and isinstance(f.base, MultimodeCOP))
                    ]
                    step_terms.append(_collapse_num(Mul(*num_factors)))
                if not step_terms:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                step_op = _collapse_num(Add(*step_terms))
                if step_op == 0:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                acc = _collapse_num(_mexpand(step_op * acc))
                if acc == 0:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                rem_c = c_r - a_r
                if rem_c > 0:
                    if not exps:
                        tuple_cache[k] = S.Zero
                        return S.Zero
                    exps[-1] += rem_c

            tuple_cache[k] = acc
            return acc

        def _vac_dict(d: dict):
            if vac_sets[0]:
                d = {k: vac_pf(pf, vac_sets[0]) for k, pf in d.items()}
                d = {k: pf for k, pf in d.items() if pf.table or pf.npf != 0}
                if not d:
                    return dict()

            if not vac_sets[1]:
                res = {}
                for k, pf in d.items():
                    applied = pf.n_apply(_collapse_num, on_npf=True)
                    clean_table = {k_t: v for k_t, v in applied.table.items() if v != 0}
                    if clean_table or applied.npf != 0:
                        res[k] = PF(clean_table, applied.npf)
                return res if res else dict()

            pfs = []
            for k, pf in d.items():
                if not pf.table and pf.npf == 0:
                    continue

                if pf.npf != 0:
                    ck = _vac_tuple(k)
                    if ck != 0:
                        c_npf = _collapse_num(pf.npf)
                        if c_npf != 0:
                            pfs.append(PF({}, _collapse_num(ck * c_npf)))

                for (bs, fs), tail in pf.table.items():
                    bs_weak = tuple(b for b in bs if b.name in weak_names)
                    bs_strong = tuple(b for b in bs if b.name not in weak_names)

                    if not bs_weak:
                        ck = _vac_tuple(k)
                        if ck == 0:
                            continue
                        sub_pf = PF({(bs_strong, fs): tail}, S.Zero)
                        applied = sub_pf.n_apply(_collapse_num, on_npf=True)
                        clean_table = {k_t: v for k_t, v in applied.table.items() if v != 0}
                        if clean_table or applied.npf != 0:
                            vp = PF(clean_table, applied.npf)
                            pfs.append(_collapse_num(ck) * vp)
                    else:
                        for s_comb, new_k in self._combine_tcp_vac(k, bs_weak):
                            ck = _vac_tuple(new_k)
                            if ck == 0:
                                continue
                            sub_pf = PF({(bs_strong, fs): tail}, S.Zero)
                            applied = sub_pf.n_apply(_collapse_num, on_npf=True)
                            clean_table = {k_t: v for k_t, v in applied.table.items() if v != 0}
                            if clean_table or applied.npf != 0:
                                vp = PF(clean_table, applied.npf)
                                pfs.append(_collapse_num(cancel(s_comb * ck)) * vp)

            return pf_add(*pfs) if pfs else dict()

        def _vac_f_dict(fd: f_dict):
            if len(vac_sets[1]) == 2:
                return dict()
            if not vac_sets[1]:
                res = f_dict(b_left=fd.b_left)
                for b, d in fd.items():
                    vd = _vac_dict(d)
                    if isinstance(vd, dict) and vd:
                        res[b] = vd
                    elif isinstance(vd, PF) and (vd.table or vd.npf != 0):
                        res[b] = {(None,): vd}
                return res if res else dict()

            fds = []
            for b, d in fd.items():
                if not d:
                    continue
                if b.name in vac_sets[1]:
                    if (b.is_annihilation and not fd.b_left
                        or not b.is_annihilation and fd.b_left):
                        continue
                    swapped_fd = self.bd_swap(b, d, to_left=not b.is_annihilation)
                    b_partner = self.odag(b)
                    d_partner = swapped_fd.get(b_partner)
                    if d_partner:
                        vp = _vac_dict(d_partner)
                        if isinstance(vp, PF) and (vp.table or vp.npf != 0):
                            fds.append(f_dict({b_partner: {(None,): vp}},
                                              b_left=swapped_fd.b_left))
                        elif isinstance(vp, dict) and vp:
                            fds.append(f_dict({b_partner: vp},
                                              b_left=swapped_fd.b_left))
                else:
                    vp = _vac_dict(d)
                    if isinstance(vp, PF) and (vp.table or vp.npf != 0):
                        fds.append(f_dict({b: {(None,): vp}}, b_left=fd.b_left))
                    elif isinstance(vp, dict) and vp:
                        fds.append(f_dict({b: vp}, b_left=fd.b_left))

            if not fds:
                return dict()
            combined = self.d_add(*fds)
            cleaned = f_dict(b_left=combined.b_left)
            for b, sub in combined.items():
                for k, pf in sub.items():
                    if pf.table or pf.npf != 0:
                        cleaned.setdefault(b, {})[k] = pf
            return cleaned if cleaned else dict()

        if isinstance(obj, f_dict):
            res = _vac_f_dict(obj)
        elif isinstance(obj, dict):
            res = _vac_dict(obj)
        elif isinstance(obj, PF):
            res = _vac_pf(obj)
        elif isinstance(obj, Expr):
            res = _vac_pf(pf(obj, as_dict=None))
        else:
            res = _vac_pf(pf(sympify(obj), as_dict=None))

        return res

    def eval_coherent(self, obj, order: int = 0):
        """
        Evaluates an operator representation under coherent state assertion
        for the strong pump field.

        :param obj: Expr, PF, dict, or f_dict.
        :param order: Target perturbative expansion order.
        """
        alpha = Symbol('alpha_0') if self.dc else Symbol('alpha_2')
        sub_dict = {self.nd.name: alpha}

        def _apply_cohr(item):

            if isinstance(item, f_dict):
                res = f_dict(b_left=item.b_left)
                for b, d in item.items():
                    child = _apply_cohr(d)
                    if child:
                        res[b] = child
                return res if res else dict()
            elif isinstance(item, dict):
                res = {}
                for k, v in item.items():
                    child = _apply_cohr(v)
                    if isinstance(child, PF):
                        if child.table or child.npf != 0:
                            res[k] = child
                    elif child:
                        res[k] = child
                return res if res else dict()
            elif isinstance(item, PF):
                return self.pe.cohr_pf(item, sub_dict, order)
            elif isinstance(item, Expr):
                return self.pe.cohr_pf(pf(item, as_dict=None), sub_dict, order)
            else:
                return self.pe.cohr_pf(pf(sympify(item), as_dict=None),
                                       sub_dict, order)

        return _apply_cohr(obj)

_e = lambda p: S.Zero if p == 1 else (p.exp if p.is_Pow else S.One)
