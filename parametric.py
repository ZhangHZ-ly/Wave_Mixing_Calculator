from sympy.core.singleton import S
from sympy.core.numbers import I
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.symbol import Dummy, Symbol, symbols
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.function import _mexpand
from sympy.functions.elementary.exponential import exp
from sympy.concrete.summations import Sum
from sympy.physics.quantum.operator import Operator, HermitianOperator
from sympy.core.traversal import bottom_up
from sympy.core.exprtools import factor_terms
from sympy.polys.polytools import cancel
from sympy.core.containers import Dict
from op_patterns.boson_pattern import BosonNum, NumShift, fac
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from perturb_eval import PF, PE, bsc, pattern_form
from perturb_eval.perturb_expander import gperm
from dsolve_pm import group_terms, dsd1, dsd2, _ps
from collections import defaultdict
from itertools import product
from functools import cached_property, cache, lru_cache
from sympy.functions.combinatorial.factorials import factorial, binomial
import warnings

warnings.simplefilter("once", UserWarning)

_mcc_indices = tuple(Dummy(integer=True, nonnegative=True) for _ in range(2))
@lru_cache(8)  # The number of terms EXPONENTIALLY EXPLODES!
def mm_comm_dummy(order: int = 0):
    if order == 0:
        return {0: S.One}
    s = _mcc_indices[not (order & 1)]  # sum index
    n = _mcc_indices[order & 1]  # variable to be presented in the result
    r = dict()  # result
    for i, c in mm_comm_dummy(order-1).items():
        r[i] = Sum(c, (s, 0, n-1)).doit()
        # wrapp the factors in a binary number
        r[i | (1 << (order-1))] = Sum(c*s, (s, 0, n-1)).doit()
    return r

@lru_cache(64)
def _weak_cop_p(m: int, p: int, n1, n2):
    """
    Polynomial P_{m, p}(N_1, N_2) generated from normal-ordering single-mode
    operators when commuting (a_1 a_2)^m with (A†)^N or (a_1† a_2†)^m with A^N.
    """
    delta = p - m
    terms = []
    N1 = BosonNum(n1)
    N2 = BosonNum(n2)
    for k_term in range(max(0, p - m), min(m, p) + 1):
        j_term = p - k_term
        c_comb = binomial(m, k_term) * binomial(m, j_term)
        u_term, v_term = m - j_term, k_term
        if delta >= 0:
            f1 = Mul(*[N1 + r for r in range(1, u_term + 1)]) if u_term > 0 else S.One
            f2 = Mul(*[N2 - r for r in range(m - k_term)]) if (m - k_term) > 0 else S.One
            terms.append(c_comb * f1 * f2)
        else:
            d_term = -delta
            f1 = Mul(*[N1 - d_term + r for r in range(1, v_term + 1)]) if v_term > 0 else S.One
            f2 = Mul(*[N2 - r for r in range(j_term)]) if j_term > 0 else S.One
            terms.append(c_comb * f1 * f2)
    return Add(*terms)

class MultimodeCOP(Operator):
    """
    Multimode canonical operator Σ a_1(†)*a_2(†).

    paraty: whether the two operators has different is_annihilation properties.

    annih1: whether the left operator is annihilation.
    
    The two operators will be ordered cannonically according to their names.
    """
    def __new__(cls, name1, name2, parity=0, annih1=1):
        name1 = sympify(name1)
        name2 = sympify(name2)
        if name1 == name2:
            raise ValueError('The two names of modes must be different')
        parity = S.Zero if not bool(parity) else S.One
        annih1 = Integer(annih1)
        if name2 is None or name1 is not None and name1.compare(name2) == 1:
            tn = name2
            name2 = name1
            name1 = tn
            annih1 = parity ^ bool(annih1)
        return Operator.__new__(cls, name1, name2, parity, annih1)
    
    @property
    def names(self):
        return self.args[:2]
    @property
    def parity(self):
        return self.args[2]
    @property
    def annih1(self):
        return bool(self.args[3])
    @property
    def annih2(self):
        return self.annih1 ^ bool(self.parity)
    @property
    def mode_expr(self):
        return (BosonOp(self.names[0], self.annih1) * BosonOp(self.names[1], self.annih2))

    def swap_exps(self, exp1, exp2, order=0, left_p=True):
        """
        Result for rewriting self ** exp1 * self† ** exp2
        into Σ l(N) * self† ** e2 * self ** e1 * r(N)
        
        l(N) / r(N) is 1 if left_p is set True / False

        In other words, the 'left_p' parameter dictates whether the exponents of 
        MultimodeCOP is left- or right- multiplied to the number operators
        """
        order = min(order, exp1, exp2)


        s = 1 if self.annih1 else -1
        tau = 1 if self.annih2 else -1
        S0 = MultimodeNum2(*self.names, self.parity)
        if not left_p:
            S0 = S0 + 2 * tau * (exp1 - exp2)
        c0 = []
        c1 = []
        for j in range(order):
            mj = exp2 - j
            c0.append(s * mj * (S0 + tau * (mj - 1 + 2 * j)))
            c1.append(-2 * s * tau * mj)

        result = [Mul(Dagger(self)**exp2, self**exp1)]
        factors = []
        for o in range(1, order + 1):
            sub = {_mcc_indices[o & 1]: exp1}
            e1, e2 = exp1 - o, exp2 - o
            for i, term in mm_comm_dummy(o).items():
                t_sub = term.subs(sub)
                if t_sub is S.Zero:
                    continue
                factors.append(t_sub)
                for j in range(o):
                    factors.append(c1[j] if (i >> j) & 1 else c0[j])
                if left_p:
                    result.append(Mul(Dagger(self)**e2, self**e1, *factors))
                else:
                    result.append(Mul(*factors, Dagger(self)**e2, self**e1))
                factors.clear()
        return Add(*result)

    def _eval_adjoint(self):
        return MultimodeCOP(*self.names, self.parity, not self.annih1)
    def _eval_commutator_BosonPat(self, other, **options):
        if other.name not in self.names:
            return S.Zero
        n1, n2 = self.args[:2]
        if isinstance(other, BosonOp):
            if other.name == n1:
                if self.annih1 == other.is_annihilation:
                    return S.Zero
                return (BosonOp(n2, self.annih2) if self.annih1
                        else -BosonOp(n2, self.annih2))
            else:
                if self.annih2 == other.is_annihilation:
                    return S.Zero
                return (BosonOp(n1, self.annih1) if self.annih2
                        else -BosonOp(n1, self.annih1))
        if isinstance(other, BosonNum):
            if other.name == n1:
                return self if self.annih1 else -self
            else:
                return self if self.annih2 else -self
    def _eval_commutator_FermionPat(self, other, **options):
        return S.Zero
    def _eval_commutator_MultimodeNum(self, other, **options):
        a1, a2 = self.names
        if other.name == a1:
            return self if self.annih1 else -self
        if other.name == a2:
            return self if self.annih2 else -self
    def _eval_commutator_MultimodeNum2(self, other, **options):
        if self.names == other.names:
            if self.parity != other.parity:
                return S.Zero
            return 2*self if self.annih2 else -2*self
    def _eval_commutator_MultimodeCOP(self, other, **options):
        if self.names == other.names and self.parity==other.parity:
            if self.annih1 != other.annih1:
                return (MultimodeNum2(*self.names, self.parity) if self.annih1
                        else -MultimodeNum2(*self.names, self.parity))

    def _print_contents(self, printer, *args):
        str1 = str(self.names[0]) if self.annih1 else f'D({self.names[0]})'
        str2 = str(self.names[1]) if self.annih2 else f'D({self.names[1]})'
        return r'Σ(%s*%s)' % (str1, str2)
    def _print_contents_latex(self, printer, *args):
        a1 = (f'{str(self.names[0])}') if self.annih1 else (r'%s^\dagger' % self.names[0])
        a2 = (f'{str(self.names[1])}') if self.annih2 else (r'%s^\dagger' % self.names[1])
        return r'\sum_\text{modes}{\left(%s %s\right)}' % (a1, a2)

class MultimodeNum(HermitianOperator):
    """
    Sum of multimode number operators.
    
    The displacement is the shared shift in each mode.
    This is summed over modes,
    so it cannot be represented by a direct Add to the operators.
    """

    boson_num_only = True

    def __new__(cls, name, displacement=0):
        displacement = sympify(displacement)
        if displacement.is_real == False:
            raise ValueError('The displacement must be a real number')
        return HermitianOperator.__new__(cls, name, displacement)
    
    @property
    def name(self):
        return self.args[0]
    @property
    def displacement(self):
        return self.args[1]
    @cached_property
    def mode_expr(self):
        return (BosonNum(self.name) + self.displacement)
    
    def _eval_commutator_BosonPat(self, other, **options):
        if other.name == self.name:
            return other.parity_diff * other
    def _eval_commutator_MultimodeNum(self, other, **options):
        return S.Zero
    def _eval_commutator_MultimodeNum2(self, other, **options):
        return S.Zero
    def _eval_commutator_FermionPat(self, other, **options):
        return S.Zero

    def _print_contents(self, printer, *args):
        return r'Σ(%s)' % str(self.mode_expr)
    def _print_contents_latex(self, printer, *args):
        if self.displacement == 0:
            return r'\sum_\text{modes}{%s}' % printer._print(self.mode_expr)
        return r'\sum_\text{modes}\left({%s}\right)' % printer._print(self.mode_expr)

class MultimodeNum2(HermitianOperator):
    """
    Sum of multimode number operators, defined as
    [a_1 * a_2(†), (a_1 * a_2(†))†] = MultimodeNum(a_1.name, a_2.name, parity)

    The parity indicates whether a_2 is a creation operator
    
    The displacement is the shared shift in each mode.
    This is summed over modes,
    so it cannot be represented by a direct Add to the operators.

    Note: For the degenerate mode with parity=0, the displacement should be doubled,
    so the exact expression for the sum is in fact
    Σ(N_1 + N_2 + displacement + 1)  + displacement.
    """

    boson_num_only = True

    def __new__(cls, name1, name2, parity=0, displacement=0):
        name1 = sympify(name1)
        name2 = sympify(name2)
        displacement = sympify(displacement)
        if displacement.is_real == False:
            raise ValueError('The displacement must be a real number')
        if name1 == name2:
            raise ValueError('The two operators must be in different modes')
        parity = bool(parity)
        if name2 is None or name1 is not None and name1.compare(name2) == 1:
            return (-1)**parity*HermitianOperator.__new__(cls, name2, name1,
                                                          parity, displacement)
        return HermitianOperator.__new__(cls, name1, name2, parity, displacement)
    
    @property
    def names(self):
        return self.args[:2]
    @property
    def parity(self):
        return self.args[2]
    @property
    def displacement(self):
        return self.args[3]
    @cached_property
    def mode_expr(self):
        a_1, a_2 = self.names
        return (BosonNum(a_2) - BosonNum(a_1) + self.displacement if self.parity
                else BosonNum(a_1) + BosonNum(a_2) + 1 + self.displacement)
    
    def _eval_commutator_BosonPat(self, other, **options):
        n1, n2 = self.args[:2]
        if other.name == n2:
            return other.parity_diff * other
        if other.name == n1:
            if self.parity:
                return -other.parity_diff * other
            return other.parity_diff * other
    def _eval_commutator_MultimodeNum2(self, other, **options):
        return S.Zero
    def _eval_commutator_FermionPat(self, other, **options):
        return S.Zero
    
    def _print_contents(self, printer, *args):
        return r'Σ(%s)' % str(self.mode_expr)
    def _print_contents_latex(self, printer, *args):
        return r'\sum_\text{modes}\left({%s}\right)' % printer._print(self.mode_expr)

def mmn_shift(e, mmop: MultimodeCOP, pd):
    if not pd or not isinstance(e, Expr):
        return e
    if e.is_Add:
        return Add(*(mmn_shift(arg, mmop, pd) for arg in e.args))
    if e.is_Mul:
        return Mul(*(mmn_shift(arg, mmop, pd) for arg in e.args))
    if e.is_Pow:
        return mmn_shift(e.base, mmop, pd)**e.exp
    if e.is_Function:
        return e.func(*(mmn_shift(arg, mmop, pd) for arg in e.args))
    if isinstance(e, MultimodeNum):
        n, d = e.args
        n1, n2 = mmop.names
        if n == n1:
            return MultimodeNum(n, d+pd) if mmop.annih1 else MultimodeNum(n, d-pd)
        if n == n2:
            return MultimodeNum(n, d+pd) if mmop.annih2 else MultimodeNum(n, d-pd)
    if isinstance(e, MultimodeNum2):
        if mmop.names == e.names:
            return (MultimodeNum2(*e.names, e.parity, e.displacement+2*pd) if mmop.annih2
                    else MultimodeNum2(*e.names, e.parity, e.displacement-2*pd))
    return e

def bp_swap(a: BosonOp, p: PF, s, to_left=True):
    """
    Swaps the single-mode boson operator (a)
    and given PF (p) with Multimode operators

    :param s: the sign factor for the swap, either 1 or -1
    :param to_left: whether the operator is swapped to the left of the PF
    """
    def mmn_shift(*args):
        if args[0] == a.name:
            if a.is_annihilation ^ to_left:
                return MultimodeNum(*args)+1
            return MultimodeNum(*args)-1
        return MultimodeNum(*args)
    def mmn2_shift(*args):
        n1, n2 = args[:2]
        if a.name == n2:
            if a.is_annihilation ^ to_left:
                return MultimodeNum2(*args)+1
            return MultimodeNum2(*args)-1
        if a.name == n1:
            if args[2] ^ a.is_annihilation == to_left:
                return MultimodeNum2(*args)-1
            return MultimodeNum2(*args)+1
        return MultimodeNum2(*args)
    def shift_e(e):
        res = NumShift(e.replace(MultimodeNum, mmn_shift)
                       .replace(MultimodeNum2, mmn2_shift),
                       1 if to_left else -1, a).doit()
        return s * res

    weak_names = {a_s.name, a_i.name, 'a_1', 'a_2'}
    new_table = defaultdict(lambda: S.Zero)
    new_npf = shift_e(p.npf) if p.npf != S.Zero else S.Zero
    pf_a = PF({((a,), ()): S.One})

    for (bs, fs), tail in p.table.items():
        has_weak = any(getattr(b, 'name', None) in weak_names for b in bs)
        if not has_weak:
            new_table[(bs, fs)] = Add(new_table[(bs, fs)], shift_e(tail))
        else:
            pf_bs = PF({(bs, fs): S.One})
            if to_left:
                prod = pf_bs.mul_ind(pf_a)
                base_tail = shift_e(tail)
            else:
                prod = pf_a.mul_ind(pf_bs)
                base_tail = s * tail

            for (new_bs, new_fs), factor in prod.table.items():
                new_table[(new_bs, new_fs)] = Add(new_table[(new_bs, new_fs)],
                                                  factor * base_tail)

            if prod.npf != S.Zero:
                new_npf = Add(new_npf, prod.npf * base_tail)

    clean_table = {k: v for k, v in new_table.items() if v != S.Zero}
    return PF(clean_table, new_npf)

def _tdz(i, exps):
    lz=False
    new_exps = []
    for e in exps:
        if lz:
            new_exps[-1] += e
            lz = False
        elif e == 0:
            if not new_exps:
                i = not i
            else:
                lz = True
        else:
            new_exps.append(e)
    if new_exps:
        return (i, *new_exps)
    return (None,)

def _pd(t: tuple):
    total = 0
    es = iter(t)
    temp = next(es)
    for j in es:
        if temp:
            total -= j
        else:
            total += j
        temp = not temp
    return total

def t_mul(t1, t2, mid=None):
    i1, *exps1 = t1
    if not exps1:
        if mid is None:
            return t2
        i1 = mid
        exps1 = [1]
    elif mid is not None:
        if i1^(len(exps1)&1)^mid:
            # the annih1 of last multimode operator in t1 equals mid
            exps1[-1] += 1
        else:
            exps1.append(1)
    i2, *exps2 = t2
    if not exps2:
        return (i1, *exps1)
    if i1^(len(exps1)&1)^i2:
        return (i1, *exps1[:-1], exps1[-1]+ exps2[0], *exps2[1:])
    return (i1, *exps1, *exps2)

def d_apply(e, f):
    if isinstance(e, f_dict):
        return f_dict({k: d_apply(v, f) for k, v in e.items()}, b_left=e.b_left)
    if isinstance(e, dict):
        return {k: d_apply(v, f) for k, v in e.items()}
    if isinstance(e, PF):
        try:
            return e.n_apply(f, on_npf=True)
        except (TypeError, AttributeError):
            pass
    return f(e)

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
    if not p.table and p.npf == S.Zero:
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
        if c_tail != S.Zero:
            new_table[(bs, fs)] = c_tail

    new_npf = _collapse_boson_num(p.npf, vac_names)
    return PF(new_table, new_npf) if (new_table or new_npf != S.Zero) else PF0

# Dummy symbols for:
# the MultimodeNum of the signal field
# the MultimodeNum2 of the process
# the BosonNum of the driving field
# the total number of modes
_n1, _n2, _nd, _nm = symbols('n_1 n_2 n_d n_m', Dummy=True, nonnegative=True)
_g = Dummy('g', real=True)  # Dummy symbol for the parametric gain

PF0 = PF()  # Singleton null PatternForm
PF_ONE = PF({((), ()): S.One})  # Singleton identity PatternForm

# Time / nonlinear coefficient symbols: t, χ
t_ = Dummy('t', real=True)
nlc = Symbol('χ', real=True, nonnegative=True)
# Field names: a0 - pump (high frequency), a1, a2 - signal and idler (low frequency)
a_p = BosonOp('a_0')
a_s = BosonOp('a_1')
a_i = BosonOp('a_2')
pm = Symbol('Δk', real=True)

class f_dict(dict):
    """
    A dictionary subclass representing operator expressions.
    Includes the 'b_left' attribute: indicates whether the single-mode
    weak-field operator is left-multiplied (True) or right-multiplied (False)
    in the expression.
    """
    def __init__(self, *args, b_left=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.b_left = b_left

# Singleton initial operator representations at t=0 for down-conversion (dc=True)
# and frequency conversion / state transfer (dc=False):
INIT_OPS = {
    True: {
        'a_0': PF({((a_p,), ()): S.One}),
        'a_1': f_dict({a_s: {(None,): PF_ONE}}, b_left=True),
        'a_2': f_dict({a_i: {(None,): PF_ONE}}, b_left=True),
        'A': {(True, 1): PF_ONE},
        'N_t': PF({((), ()): MultimodeNum2(a_s.name, a_i.name, S.Zero)}),
    },
    False: {
        'a_0': f_dict({a_p: {(None,): PF_ONE}}, b_left=True),
        'a_1': f_dict({a_s: {(None,): PF_ONE}}, b_left=True),
        'a_2': PF({((a_i,), ()): S.One}),
        'A': {(True, 1): PF_ONE},
        'N_t': PF({((), ()): MultimodeNum2(a_p.name, a_s.name, S.One)}),
    },
}

def nd_simp(expr, nd):
    """
    Simplifies expressions involving powers of nd (such as BosonNum,
    Symbol, or Abs(alpha)**2) by:
    1. Extracting common factors of nd from Add expressions inside Mul;
    2. Merging powers of nd (using dummy commutative substitution if nd is non-commutative or composite);
    """
    if isinstance(expr, (PF, dict, f_dict)):
        return d_apply(expr, lambda x: nd_simp(x, nd))
    target_sym = nd.base if nd.is_Pow else nd
    if not isinstance(expr, Expr) or not expr.has(target_sym):
        return expr

    need_dummy = not (nd.is_commutative and nd.is_Symbol)
    nd_sym = _nd if need_dummy else nd

    def simp_mul(e):
        if e.is_Mul and e.has(target_sym):
            expr_sub = e.subs({nd: nd_sym}) if need_dummy else e
            if not expr_sub.has(nd_sym):
                return e
            if not expr_sub.is_Mul:
                return expr_sub.subs({nd_sym: nd}) if need_dummy else expr_sub

            args = []
            for a in Mul.make_args(expr_sub):
                if a.is_Add and a.has(nd_sym):
                    args.extend(Mul.make_args(factor_terms(a, clear=True, fraction=True)))
                elif a.is_Pow and a.base.is_Add and a.base.has(nd_sym):
                    args.extend(
                        arg**a.exp
                        for arg in Mul.make_args(factor_terms(a.base, clear=True, fraction=True))
                    )
                else:
                    args.append(a)

            n_factors = []
            other = []
            for arg in args:
                if arg.has(nd_sym):
                    n_factors.append(arg)
                else:
                    other.append(arg)

            n_part = _ps(Mul(*n_factors)) if n_factors else S.One
            if need_dummy:
                n_part = n_part.subs({nd_sym: nd})
            return Mul(*other, n_part)
        return e

    return bottom_up(expr, simp_mul)


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

    def __init__(self, dc=True, mismatch=False):
        self.dc = bool(dc)
        self.nd = BosonNum(a_p.name) if dc else BosonNum(a_i.name)
        self.a = ((MultimodeCOP(a_s.name, a_i.name, annih1=0),
                   MultimodeCOP(a_s.name, a_i.name))
                  if dc else (MultimodeCOP(a_p.name, a_s.name, 1, 0),
                              MultimodeCOP(a_p.name, a_s.name, 1)))
        if isinstance(mismatch, (Expr, Symbol)):
            self.pm = mismatch
        else:
            self.pm = pm if mismatch else S.Zero
        self.sol = []
        self.pe = PE({nlc: S.One})
        self.pe.register(MultimodeNum)
        self.pe.register(MultimodeNum2)
        self.pe.register(MultimodeCOP)

    def update_pe(self, orders: dict = None):
        """Update self.pe assumptions with custom symbol/mode orders."""
        assumptions = dict(self.pe.assumptions)
        clr = False
        if orders:
            for k, v in orders.items():
                if v is not None:
                    if assumptions.get(k, v) != sympify(v):
                        clr = True
                    assumptions[k] = sympify(v)
        self.pe.assumptions = Dict(assumptions)
        if clr:
            self.pe._lo_cache.clear()
            self.pe._subs_cache.clear()

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

    def _od(self, b: BosonOp):
        """
        Returns the partner single-mode operator
        according to the parity rule of self.a[0].

        If the operator does not belong to the primary modes,
        returns the operator itself.
        """
        n1, n2, p = self.a[0].args[:3]
        if b.name == n1:
            return BosonOp(n2, b.is_annihilation==p)
        if b.name == n2:
            return BosonOp(n1, b.is_annihilation==p)
        return b

    def f_dict_parity(self, d: f_dict):
        """
        Determines the parity class (0 or 1) of an f_dict.
        Two f_dicts can be multiplied into a dict iff they have opposite parities
        (i.e. each key b1 in p1 satisfies b1 == b2.adjoint() or b1 == self._od(b2).adjoint()).
        Raises ValueError if the f_dict itself contains keys of mixed parities.
        """
        if not d:
            return None
        keys = list(d.keys())
        b0 = keys[0]
        for b in keys[1:]:
            if not (b == b0 or b == self._od(b0)):
                raise ValueError('Inconsistent BosonOp keys within f_dict')
        n1, n2 = self.a[0].names
        p = self.a[0].parity
        if b0.name == n1:
            return 0 if b0.is_annihilation else 1
        elif b0.name == n2:
            return (0 if b0.is_annihilation else 1) ^ (0 if p else 1)
        else:
            raise ValueError(f'Unexpected BosonOp {b0} in f_dict')

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
            parities = {self.f_dict_parity(d) for d in ds if d}
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
        return {k: PF.add(*vs) for k, vs in result.items()}

    def d_mul(self, d0, d1, d2, mid=None):
        if not d1 or not d2:
            return
        weak_names = set(self.a[0].names)
        n1, n2 = self.a[0].names
        parity = self.a[0].parity

        def shift_num_ops(expr, pd):
            if not pd:
                return expr
            expr = mmn_shift(expr, self.a[0], pd)
            bn1 = BosonNum(n1)
            bn2 = BosonNum(n2)
            sub = {}
            if expr.has(bn1):
                sub[bn1] = bn1 + pd
            if expr.has(bn2):
                shift2 = -pd if parity else pd
                sub[bn2] = bn2 + shift2
            if sub:
                expr = expr.subs(sub)
            return expr

        for (k2, e2) in d2.items():
            pd = _pd(k2)
            for (k1, e1) in d1.items():
                has_weak_pair = False
                if e1.table:
                    for bs, fs in e1.table.keys():
                        bw = tuple(b for b in bs if b.name in weak_names)
                        if bw:
                            if (self._match_cop(self.a[1], bw) is not None
                                or self._match_cop(self.a[0], bw) is not None):
                                has_weak_pair = True
                                break

                if not has_weak_pair:
                    k = t_mul(k1, k2) if isinstance(mid, MultimodeNum) else t_mul(k1, k2, mid)
                    factor_mid = lambda t: shift_num_ops(mid * t if isinstance(mid, MultimodeNum) else t, pd)
                    new = e1.n_apply(factor_mid)
                    term_prod = new.mul_ind(e2)
                    if k in d0:
                        d0[k] = PF.add(d0[k], term_prod)
                    else:
                        d0[k] = term_prod
                else:
                    if e1.npf != S.Zero:
                        k = t_mul(k1, k2) if isinstance(mid, MultimodeNum) else t_mul(k1, k2, mid)
                        c_npf = shift_num_ops(mid * e1.npf if isinstance(mid, MultimodeNum) else e1.npf, pd)
                        p_npf = PF({}, c_npf).mul_ind(e2)
                        if k in d0:
                            d0[k] = PF.add(d0[k], p_npf)
                        else:
                            d0[k] = p_npf

                    for (bs, fs), tail in e1.table.items():
                        bw = tuple(b for b in bs if b.name in weak_names)
                        bs_strong = tuple(b for b in bs if b.name not in weak_names)

                        m_ann = self._match_cop(self.a[1], bw) if bw else None
                        m_cre = self._match_cop(self.a[0], bw) if bw else None

                        if m_ann is None and m_cre is None:
                            k = t_mul(k1, k2) if isinstance(mid, MultimodeNum) else t_mul(k1, k2, mid)
                            c_tail = shift_num_ops(mid * tail if isinstance(mid, MultimodeNum) else tail, pd)
                            p_term = PF({(bs, fs): c_tail}, S.Zero).mul_ind(e2)
                            if k in d0:
                                d0[k] = PF.add(d0[k], p_term)
                            else:
                                d0[k] = p_term
                        else:
                            cop_idx = 1 if m_ann is not None else 0
                            m = m_ann if m_ann is not None else m_cre
                            c_tail = shift_num_ops(mid * tail if isinstance(mid, MultimodeNum) else tail, pd)

                            for s, k_sub, bw_info, P_num in self.swap_tuple_weak_cop(cop_idx, m, k2):
                                k_tot = t_mul(k1, k_sub) if isinstance(mid, MultimodeNum) else t_mul(k1, k_sub, mid)
                                if bw_info is None:
                                    new_bw = ()
                                else:
                                    new_cop_idx, new_m = bw_info
                                    cop_tgt = self.a[new_cop_idx]
                                    op1 = BosonOp(cop_tgt.names[0], cop_tgt.annih1)
                                    op2 = BosonOp(cop_tgt.names[1], cop_tgt.annih2)
                                    if new_m == 1:
                                        new_bw = (op1, op2)
                                    else:
                                        new_bw = (op1**new_m, op2**new_m)

                                new_bs = bs_strong + new_bw
                                final_tail = s * c_tail * P_num
                                p_term = PF({(new_bs, fs): final_tail}, S.Zero).mul_ind(e2)
                                if k_tot in d0:
                                    d0[k_tot] = PF.add(d0[k_tot], p_term)
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
            if self.f_dict_parity(e1) == self.f_dict_parity(e2):
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
                        pf_nb = pattern_form(nb, as_dict=None)
                        val = (pf_nb.mul_ind(p) if s == 1
                               else pf_nb.mul_ind(p).n_apply(lambda x: s * x, on_npf=True))
                        if key in res:
                            res[key] = PF.add(res[key], val)
                        else:
                            res[key] = val
        else:
            for b, d in fd.items():
                for t, p in d.items():
                    p_prime = bp_swap(b, p, 1, to_left=True)
                    pf_b = pattern_form(b, as_dict=None)
                    val = pf_b.mul_ind(p_prime)
                    if t in res:
                        res[t] = PF.add(res[t], val)
                    else:
                        res[t] = val
        return res

    def f_mul(self, e1, e2):
        if not e1 or not e2:
            return dict()
        if not isinstance(e1, f_dict) or not isinstance(e2, f_dict):
            return self.e_mul(e1, e2)
        if self.f_dict_parity(e1) == self.f_dict_parity(e2):
            raise ValueError('Solution parity unbalanced')
        d1 = self.f_to_dict(e1)
        d2 = self.f_to_dict(e2)
        result = dict()
        self.d_mul(result, d1, d2)
        return result

    def bt_swap(self, a: BosonOp, i, exps, to_left=False):
        yield S.One, a, exps
        if to_left:
            for j, e in enumerate(exps):
                if i ^ (j & 1) ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                    for s, b, nes in self.bt_swap(self._od(a), i, exps[:j], True):
                        yield (S.NegativeOne**int(a.is_annihilation)*s*e,
                               b, (*nes, e-1, *exps[j+1:]))
        else:
            for j, e in enumerate(exps):
                ic = i ^ (j & 1)
                if ic ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                    for s, b, nes in self.bt_swap(self._od(a), not ic, exps[j+1:], False):
                        yield (S.NegativeOne**int(not a.is_annihilation)*s*e,
                               b, (*exps[:j], e-1, *nes))

    def bd_swap(self, a: BosonOp, d, to_left=True):
        """
        Swaps the order of the operator a and the dict d
        The direction of the swap (for a) is determined by the to_left parameter.
        """
        new_dict = f_dict({a: dict(), self._od(a): dict()}, b_left=to_left)
        for t, p in d.items():
            i, *exps = t
            for s, b, nes in self.bt_swap(a, i, exps, to_left=to_left):
                key = _tdz(i, nes)
                if key in new_dict[b]:
                    new_dict[b][_tdz(i, nes)] = PF.add(new_dict[b][_tdz(i, nes)],
                                                       bp_swap(b, p, s, to_left))
                else:
                    new_dict[b][_tdz(i, nes)] = bp_swap(b, p, s, to_left)
        return new_dict

    def _match_cop(self, cop, bw):
        """
        Matches a tuple of weak-field boson operators bw with a MultimodeCOP cop.
        Returns the common exponent m if matched, otherwise None.
        """
        if len(bw) != 2:
            return None
        b1, b2 = bw
        m1, m2 = b1.exps[0], b2.exps[0]
        if m1 != m2:
            return None
        base1, base2 = b1.left_base, b2.left_base
        n1, n2 = cop.names
        a1, a2 = cop.annih1, cop.annih2
        if base1.name == n1 and base2.name == n2:
            if base1.is_annihilation == a1 and base2.is_annihilation == a2:
                return m1
        elif base1.name == n2 and base2.name == n1:
            if base1.is_annihilation == a2 and base2.is_annihilation == a1:
                return m1
        return None

    def swap_tuple_weak_cop(self, cop_idx: int, m: int, k: tuple):
        """
        Swaps a single-mode weak-field COP pair power (cop_idx, m)
        with a multimode tuple key k = (i, *exps).

        Yields (s, k_sub, bw_info, P_num):
        - s: scalar combinatorial multiplier
        - k_sub: resulting tuple key
        - bw_info: None (if ladder operators eliminate) or (new_cop_idx, new_m)
        - P_num: single-mode number operator polynomial in BosonNum
        """
        if not k or k == (None,) or len(k) == 1:
            yield S.One, (None,), (cop_idx, m), S.One
            return

        i, *exps = k
        if not exps:
            yield S.One, (None,), (cop_idx, m), S.One
            return

        e_0 = exps[0]
        exps_rest = exps[1:]
        block_idx = int(bool(i))
        k_rest = _tdz(not i, exps_rest) if exps_rest else (None,)

        if block_idx == cop_idx:
            # Same kind, commutes trivially
            for s_r, sub_k, bw_r, P_r in self.swap_tuple_weak_cop(cop_idx, m, k_rest):
                new_k = t_mul((i, e_0), sub_k)
                yield s_r, new_k, bw_r, P_r
            return

        # Opposite kind, non-trivial swap
        sigma = 1 if cop_idx == 1 else -1
        n1, n2 = self.a[0].names
        parity = self.a[0].parity
        N1 = BosonNum(n1)
        N2 = BosonNum(n2)

        max_p = min(2 * m, e_0)
        for p in range(max_p + 1):
            c_p = (sigma**p) * (factorial(e_0) // factorial(e_0 - p))
            e_0_prime = e_0 - p
            delta = p - m
            P_p = _weak_cop_p(m, p, n1, n2)

            prefix_k = (i, e_0_prime) if e_0_prime > 0 else (None,)

            if k_rest == (None,):
                bw_info = (cop_idx if delta <= 0
                           else 1 - cop_idx, abs(delta)) if delta != 0 else None
                yield c_p, prefix_k, bw_info, P_p
            else:
                pd_rest = _pd(k_rest)
                shift2 = -pd_rest if parity else pd_rest
                P_p_shifted = P_p.subs({N1: N1 + pd_rest, N2: N2 + shift2})

                if delta == 0:
                    new_k = t_mul(prefix_k, k_rest)
                    yield c_p, new_k, None, P_p_shifted
                else:
                    next_cop_idx = cop_idx if delta < 0 else 1 - cop_idx
                    next_m = abs(delta)
                    for s_r, sub_k, bw_r, P_r in self.swap_tuple_weak_cop(next_cop_idx,
                                                                          next_m, k_rest):
                        new_k = t_mul(prefix_k, sub_k)
                        yield c_p * s_r, new_k, bw_r, P_p_shifted * P_r


    def nd_simp(self, expr):
        return nd_simp(expr, self.nd)

    def n_simp(self, expr):
        n1, n2 = self.a[0].names
        def mmn_d(n, d):
            if n == a_s.name:
                return _n1 + d*_nm
            if self.dc:
                return _n2 - _n1 + (d-1) * _nm
            return _n1 - _n2 + d * _nm
        def combine_n(expr):
            if (expr.is_Add
                and len(expr.atoms(MultimodeNum, MultimodeNum2)) > 1
                and any(any(isinstance(f, MultimodeNum)
                            for f in Mul.make_args(arg))
                        for arg in expr.args)):
                r = expr.replace(MultimodeNum, mmn_d).replace(MultimodeNum2,
                                                              lambda *args: _n2+args[-1]*_nm)
                c_m = r.subs({_n2: 0})
                c = c_m.subs({_nm: 0})
                c_n2 = (_mexpand(r.subs({_nm: 0}) - c) / _n2).doit()
                d =(_mexpand((c_m - c) / _nm) / c_n2).doit()
                if d.is_number and d.is_real is not False:
                    expr = (c_n2 * _n2 + c).factor().subs({_n1: MultimodeNum(a_s.name),
                                                           _n2: MultimodeNum2(n1, n2,
                                                                              not self.dc, d)})
                else:
                    warnings.warn('Failed to combine weak-field multimode number operators')
            if expr.is_Mul:
                expr = nd_simp(expr, self.nd)
            expr = _ps(expr)
            return expr
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

    def _sm_expr(self, expr):
        """Simplifies number operators and mode count in scalar/coefficient expressions."""
        expr = expr.replace(
            lambda x: isinstance(x, (MultimodeNum, MultimodeNum2, MultimodeCOP)),
            lambda x: x.mode_expr
        ).subs({_nm: 1, Symbol('n_m'): S.One})
        return nd_simp(expr, self.nd)

    def _smk(self, k):
        """Single mode key"""
        if k is None:
            return PF_ONE
        if isinstance(k, tuple):
            fs = []
            i, *exps = k
            if not exps:
                return PF_ONE
            for e in exps:
                b1, b2 = Mul.make_args(self.a[i].mode_expr)
                fs.append(b1**e)
                fs.append(b2**e)
                i = not i
            b, t = bsc(fs).result_ind()
            return PF({(b, ()): t})
        if isinstance(k, Expr):
            return pattern_form(k, as_dict=None)
        return k

    def to_single_mode(self, obj):
        """
        Converts multimode operator representations (MultimodeCOP, MultimodeNum, MultimodeNum2)
        into single-mode boson operators (mode_expr) with n_m = 1,
        and evaluates into a unified PatternForm (PF).

        :param obj: Expr, PF, dict, f_dict, or WmWrapper.
        :return: PF (PFTableProcessor)
        """

        if hasattr(obj, 'val'):
            obj = obj.val

        if isinstance(obj, (dict, f_dict)):
            b_left = getattr(obj, 'b_left', True)
            if b_left:
                res = PF.add(*(self._smk(k).mul_ind(self.to_single_mode(v))
                               for k, v in obj.items()))
            else:
                res = PF.add(*(self.to_single_mode(v).mul_ind(self._smk(k))
                               for k, v in obj.items()))
            return res.n_apply(lambda x: nd_simp(x, self.nd), on_npf=True)
        elif isinstance(obj, PF):
            return obj.n_apply(lambda t: self._sm_expr(t), on_npf=True)
        elif isinstance(obj, Expr):
            return pattern_form(self._sm_expr(obj), as_dict=None)

        return self.to_single_mode(sympify(obj))

    def _combine_weak_with_tuple(self, k, bs_weak):
        """
        Combines weak-field BosonPat operators with a multimode tuple key k.
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

        # Check for annihilation COP (cop_idx = 1):
        # Since bs_weak is on the right, annihilation operator on vacuum gives 0
        m_ann = self._match_cop(self.a[1], bs_weak)
        if m_ann is not None:
            return

        # Check for creation COP (cop_idx = 0):
        m_cre = self._match_cop(self.a[0], bs_weak)
        if m_cre is not None:
            new_k = k
            for _ in range(m_cre):
                new_k = t_mul(new_k, (None,), 0)
            # Normalization factor C = binomial(nm_sym + m - 1, m)
            # Represented via cached permutation gperm and factorial fac:
            # C = gperm(_nm + m_cre - 1, m_cre) / fac(m_cre)
            s_comb = fac(m_cre) / gperm(_nm + m_cre - 1, m_cre)
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

        :param obj: Expr, PF, dict, f_dict, or wrapper with `.val`.
        :param vac_modes: Iterable of mode names / BosonOps asserted to be in vacuum.
        :return: Collapsed PF, dict, or f_dict representation.
        """

        if hasattr(obj, 'val'):
            obj = obj.val

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
                return displacement * _nm
            return MultimodeNum(name, displacement)

        def _collapse_mmn2(n1, n2, p, d):
            in1 = n1 in vac_sets[1]
            in2 = n2 in vac_sets[1]
            if not in1 and not in2:
                return MultimodeNum2(n1, n2, p, d)
            if not p:
                if in1 and in2:
                    return (S.One + d) * _nm
                n_surv = n2 if in1 else n1
                return MultimodeNum(n_surv, 1) + d * _nm
            else:
                if in1 and in2:
                    return d * _nm
                if in1:
                    return MultimodeNum(n2) + d * _nm
                return -MultimodeNum(n1) + d * _nm
            
        _collapse_num = (lambda expr: sympify(expr).replace(MultimodeNum, _collapse_mmn)
                         .replace(MultimodeNum2, _collapse_mmn2))

        def _vac_pf(pf: PF) -> PF:
            vp = vac_pf(pf, vac_sets[0] | vac_sets[1])
            if not vp.table and vp.npf == S.Zero:
                return PF0
            applied = vp.n_apply(_collapse_num, on_npf=True)
            clean_table = {k: v for k, v in applied.table.items() if v != S.Zero}
            return PF(clean_table, applied.npf) if (clean_table or applied.npf) else PF0

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
                if step_op == S.Zero:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                acc = _collapse_num(_mexpand(step_op * acc))
                if acc == S.Zero:
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
                d = {k: pf for k, pf in d.items() if pf.table or pf.npf != S.Zero}
                if not d:
                    return dict()

            if not vac_sets[1]:
                res = {}
                for k, pf in d.items():
                    applied = pf.n_apply(_collapse_num, on_npf=True)
                    clean_table = {k_t: v for k_t, v in applied.table.items() if v != S.Zero}
                    if clean_table or applied.npf != S.Zero:
                        res[k] = PF(clean_table, applied.npf)
                return res if res else dict()

            pfs = []
            for k, pf in d.items():
                if not pf.table and pf.npf == S.Zero:
                    continue

                if pf.npf != S.Zero:
                    ck = _vac_tuple(k)
                    if ck != S.Zero:
                        c_npf = _collapse_num(pf.npf)
                        if c_npf != S.Zero:
                            pfs.append(PF({}, _collapse_num(ck * c_npf)))

                for (bs, fs), tail in pf.table.items():
                    bs_weak = tuple(b for b in bs if b.name in weak_names)
                    bs_strong = tuple(b for b in bs if b.name not in weak_names)

                    if not bs_weak:
                        ck = _vac_tuple(k)
                        if ck == S.Zero:
                            continue
                        sub_pf = PF({(bs_strong, fs): tail}, S.Zero)
                        applied = sub_pf.n_apply(_collapse_num, on_npf=True)
                        clean_table = {k_t: v for k_t, v in applied.table.items() if v != S.Zero}
                        if clean_table or applied.npf != S.Zero:
                            vp = PF(clean_table, applied.npf)
                            pfs.append(vp.n_apply(lambda t, c=ck: _collapse_num(c * t)))
                    else:
                        for s_comb, new_k in self._combine_weak_with_tuple(k, bs_weak):
                            ck = _vac_tuple(new_k)
                            if ck == S.Zero:
                                continue
                            sub_pf = PF({(bs_strong, fs): tail}, S.Zero)
                            applied = sub_pf.n_apply(_collapse_num, on_npf=True)
                            clean_table = {k_t: v for k_t, v in applied.table.items() if v != S.Zero}
                            if clean_table or applied.npf != S.Zero:
                                vp = PF(clean_table, applied.npf)
                                pfs.append(vp.n_apply(lambda t, c=ck, s=s_comb:
                                                      _collapse_num(cancel(s * c * t))))

            return PF.add(*pfs) if pfs else dict()

        def _vac_f_dict(fd: f_dict):
            if len(vac_sets[1]) == 2:
                return dict()
            if not vac_sets[1]:
                res = f_dict(b_left=fd.b_left)
                for b, d in fd.items():
                    vd = _vac_dict(d)
                    if isinstance(vd, dict) and vd:
                        res[b] = vd
                    elif isinstance(vd, PF) and (vd.table or vd.npf != S.Zero):
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
                    b_partner = self._od(b)
                    d_partner = swapped_fd.get(b_partner)
                    if d_partner:
                        vp = _vac_dict(d_partner)
                        if isinstance(vp, PF) and (vp.table or vp.npf != S.Zero):
                            fds.append(f_dict({b_partner: {(None,): vp}},
                                              b_left=swapped_fd.b_left))
                        elif isinstance(vp, dict) and vp:
                            fds.append(f_dict({b_partner: vp},
                                              b_left=swapped_fd.b_left))
                else:
                    vp = _vac_dict(d)
                    if isinstance(vp, PF) and (vp.table or vp.npf != S.Zero):
                        fds.append(f_dict({b: {(None,): vp}}, b_left=fd.b_left))
                    elif isinstance(vp, dict) and vp:
                        fds.append(f_dict({b: vp}, b_left=fd.b_left))

            if not fds:
                return dict()
            combined = self.d_add(*fds)
            cleaned = f_dict(b_left=combined.b_left)
            for b, sub in combined.items():
                for k, pf in sub.items():
                    if pf.table or pf.npf != S.Zero:
                        cleaned.setdefault(b, {})[k] = pf
            return cleaned if cleaned else dict()

        if isinstance(obj, f_dict):
            res = _vac_f_dict(obj)
        elif isinstance(obj, dict):
            res = _vac_dict(obj)
        elif isinstance(obj, PF):
            res = _vac_pf(obj)
        elif isinstance(obj, Expr):
            res = _vac_pf(pattern_form(obj, as_dict=None))
        else:
            res = _vac_pf(pattern_form(sympify(obj), as_dict=None))

        return res

    def eval_coherent(self, obj, order: int = 0):
        """
        Evaluates an operator representation under coherent state assertion
        for the strong pump field.

        :param obj: Expr, PF, dict, f_dict, or wrapper with `.val`.
        :param order: Target perturbative expansion order.
        """

        if hasattr(obj, 'val'):
            obj = obj.val

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
                        if child.table or child.npf != S.Zero:
                            res[k] = child
                    elif child:
                        res[k] = child
                return res if res else dict()
            elif isinstance(item, PF):
                return self.pe.cohr_pf(item, sub_dict, order)
            elif isinstance(item, Expr):
                return self.pe.cohr_pf(pattern_form(item, as_dict=None), sub_dict, order)
            else:
                return self.pe.cohr_pf(pattern_form(sympify(item), as_dict=None),
                                       sub_dict, order)

        return _apply_cohr(obj)

class TWMSolver(WMProcessor):
    """
    Three-wave mixing quantum multimode parametric process perturbation solver

    Ordering principle: The strong field and commutative parameters
    are extracted to the end, ordered in pat-form;
    the multimode patterns are left-multiplied to multimode number operators;
    the annihilation operator of single weak field with larger index number
    is left-multipied to the rest of the expression (multi-mode part),
    and the other operators are ordered as such that the commutation
    between singlemode and multimode operators is avoided.

    The above means that: a0, a1, a2† are right/left-multiplied to the expression
    in the down-conversion/phase transfer case,
    while their conjugates are left/right-multiplied.

    Under this ordering principle, we take the time differential equation
    in the following form:

    d/dt a0 = -i χ a1 a2 exp(-i Δk t),
    d/dt a1 = -i χ a0 a2† exp(i Δk t),
    d/dt a2 = -i χ a1† a0 exp(i Δk t)

    :param dc: whether the process is down-conversion (True)
    or state transfer (False); a2 is strong in the state transfer case
    :param mismatch: Phase mismatch of the process, assumed same for all modes,
    k0 - k1 - k2
    """

    def __init__(self, dc=True, mismatch=False):
        super().__init__(dc=dc, mismatch=mismatch)
        self.half_p_factor = exp(I * self.pm * t_ * S.Half)
        if dc:
            # Mark multiplication directions:
            # a_s/a_i right-multiplied (b_left=False),
            # a_s†/a_i† left-multiplied (b_left=True)
            self.sol = [({(None,): PF({((a_p,), ()): S.One})},
                         f_dict({a_s: {(None,): PF0}, Dagger(a_i): {(None,): PF0}}, b_left=False),
                         f_dict({Dagger(a_s): {(None,): PF0}, a_i: {(None,): PF0}}))]
            self.gc = S.Half*(4*nlc**2*self.nd-self.pm**2)**S.Half
            self.update_pe({a_p.name: S.NegativeOne})
        else:
            self.sol = [(f_dict({a_p: {(None,): PF0}, a_s: {(None,): PF0}}),
                         f_dict({a_p: {(None,): PF0}, a_s: {(None,): PF0}}),
                         {(None,): PF({((a_i,), ()): S.One})})]
            self.gc = S.Half*(4*nlc**2*self.nd+self.pm**2)**S.Half
            self.update_pe({a_i.name: S.NegativeOne})
        self.pe.register(MultimodeNum)
        self.pe.register(MultimodeNum2)
        self.pe.register(MultimodeCOP)
        # d/dt a_drive / (-i χ) for the last solved order
        self.dd_last = {(None,): PF0}
        # Standard expansion of an expression up to the given order
        self.expand = (lambda pf, order:
                       self.pe.expand_pf(pf, order).n_apply(lambda t:
                                                         t.subs({t_: _g/self.gc})))
        # Homogeneous solutions to the time differential equations
        @cache
        def _fdf0(n0, mode):
            if n0:
                if mode == 0:
                    return PF({((), ()): dsd2(self.gc*t_, dict(), False,
                                              S.One, I*self.pm*S.Half/self.gc)})
                return PF({((), ()): dsd2(self.gc*t_, dict(), self.dc,
                                          S.One, -I*self.pm*S.Half/self.gc)})
            elif self.dc:
                return PF({((a_p,), ()): dsd2(self.gc*t_, dict(), True,
                                                 S.Zero, -I*nlc/self.gc)})
            elif mode == 0:
                return PF({((a_i,), ()): dsd2(self.gc*t_, dict(), False,
                                                 S.Zero, -I*nlc/self.gc)})
            else:
                return PF({((Dagger(a_i),),()): dsd2(self.gc*t_, dict(), False,
                                                        S.Zero, -I*nlc/self.gc)})
        self.fdf0 = _fdf0

    def get_sol(self, order: int):
        if order < -1:
            raise ValueError('The order should be at least -1')
        i = order + 1
        if len(self.sol) > i:
            return self.sol[i]
        last_sol = self.get_sol(order-1)
        dc = self.dc
        gc_m2 = self.gc**(-2)
        if i & 1:
            def _d2ts(mode, n0, b):
                s1 = S.NegativeOne if mode == 0 else S.One
                s3 = S.NegativeOne if dc else S.One
                for k in term1[b].keys() | term2[b].keys() | last_sol[mode][b].keys():
                    pfs = []
                    if k in term1[b]:
                        pfs.append(term1[b][k].n_apply(lambda t: s1*t))
                    if k in term2[b]:
                        pfs.append(term2[b][k].n_apply(lambda t: -t))
                    if k in last_sol[mode][b]:
                        pfs.append(last_sol[mode][b][k].n_apply(lambda t: s3*t*self.nd))
                    rhs = self.expand(PF.add(*pfs).n_apply(lambda t:
                                                           Mul(nlc**2, t, gc_m2)), order)
                    s0 = rhs.n_apply(lambda t: self.n_simp(dsd2(_g, self.p2s_group(t), self.dc,
                                                                subs=self.gc*t_)))
                    self.sol[i][mode][b][k] = (PF.add(s0, self.fdf0(n0, mode))
                                               if len(k) == 1 else s0)
            if dc:
                self.sol.append((last_sol[0],
                                 f_dict({a_s: dict(), Dagger(a_i): dict()}, b_left=False),
                                 f_dict({Dagger(a_s): dict(), a_i: dict()})))
                pd = self.d_adjoint(last_sol[0])
                term1 = self.e_mul(self.e_mul(last_sol[0], pd), last_sol[1])
                term2 = self.e_mul(self.dd_last, self.d_adjoint(last_sol[2]))
                _d2ts(1, True, a_s)
                _d2ts(1, False, Dagger(a_i))
                term1 = self.e_mul(last_sol[2], self.e_mul(pd, last_sol[0]))
                term2 = self.e_mul(self.d_adjoint(last_sol[1]), self.dd_last)
                _d2ts(2, False, Dagger(a_s))
                _d2ts(2, True, a_i)
            else:
                self.sol.append((f_dict({a_p: dict(), a_s: dict()}),
                                 f_dict({a_p: dict(), a_s: dict()}),
                                 last_sol[2]))
                id = self.d_adjoint(last_sol[2])
                term1 = self.e_mul(last_sol[0], self.e_mul(id, last_sol[2]))
                term2 = self.e_mul(last_sol[1], self.dd_last)
                _d2ts(0, True, a_p)
                _d2ts(0, False, a_s)
                term1 = self.e_mul(last_sol[0], self.d_adjoint(self.dd_last))
                term2 = self.e_mul(last_sol[1], self.e_mul(last_sol[2], id))
                _d2ts(1, True, a_s)
                _d2ts(1, False, a_p)
        else:
            def _d1ts():
                b = self.sol[0][0][(None,)] if dc else self.sol[0][2][(None,)]
                result = dict()
                for k, v in self.dd_last.items():
                    rhs = self.expand(v.n_apply(lambda t: -I*nlc*t/self.gc), order)
                    s0 = rhs.n_apply(lambda t: self.n_simp(dsd1(_g, self.p2s_group(t),
                                                                subs=self.gc*t_))) 
                    result[k] = PF.add(s0, b) if len(k) == 1 else s0
                return result
            if dc:
                self.dd_last = self.e_mul(last_sol[1], last_sol[2])
                self.sol.append((_d1ts(), *last_sol[1:]))
            else:
                self.dd_last = self.e_mul(self.d_adjoint(last_sol[1]), last_sol[0])
                self.sol.append((*last_sol[:2], _d1ts()))
        return self.sol[i]

    def sol_expr(self, mode: int, order: int):
        """
        Get the expression of the solution for the given mode and order.

        The order is consistent with the order of the perturbative expansion,
        so the classical solution with zero input is considered as order -1,
        and the parametirc amplifier/oscillator solution with undepleted drive
        is considered as order 0.
        """
        s0 = self.sol[0]
        if len(s0[mode]) == 1:
            p = S.One
        elif mode == 0:
            p = self.half_p_factor**S.NegativeOne
        else:
            p = self.half_p_factor
        if len(s0[mode]) == 1 or mode != 1 or len(s0[0]) == 2:  # driving field
            return p*self.to_expr(self.get_sol(order)[mode])
        return p*self.to_expr(self.get_sol(order)[mode])


class FWMSolver(WMProcessor):
    """
    Four-wave mixing quantum multimode parametric process perturbation solver.

    The driving pump field a_0 is strong and treated parametrically.
    Solutions are obtained up to the first correction to the pump field (order 1),
    including self-phase modulation and cross-phase modulation pump depletion.

    Because the solution is capped at the first pump correction, results are
    computed and stored directly upon initialization rather than solved recursively.
    """

    def __init__(self):
        super().__init__(dc=True, mismatch=True)
        self.a = (MultimodeCOP(a_s.name, a_i.name, annih1=0),
                  MultimodeCOP(a_s.name, a_i.name))
        self.hpm = -I*self.pm*S.Half + I*nlc*self.nd
        self.gc = S.Half*(self.pm * (-self.pm + 4*nlc*self.nd))**S.Half
        self.update_pe({a_p.name: -S.Half})
        self.expand = (lambda pf, order:
                       self.pe.expand_pf(pf, order).n_apply(lambda t:
                                                             t.subs({t_: _g/self.gc})))
        self._solve()

    def _solve(self):
        # Order -1: Initial operators at t=0
        sol_m1 = (
            {(None,): PF({((a_p,), ()): S.One})},
            f_dict({a_s: {(None,): PF0}, Dagger(a_i): {(None,): PF0}}, b_left=False),
            f_dict({Dagger(a_s): {(None,): PF0}, a_i: {(None,): PF0}}, b_left=True)
        )

        # Order 0: Undepleted pump and parametric weak field evolution
        fwm0_0 = {(None,): PF({((a_p,), ()): S.One})}
        fwm0_1 = f_dict({
            a_s: {(None,): PF({((), ()): dsd2(self.gc*t_, dict(),
                                              True, S.One, self.hpm/self.gc)})},
            Dagger(a_i): {(None,): PF({((a_p**2,), ()): dsd2(self.gc*t_, dict(),
                                                             True, S.Zero, -I*nlc/self.gc)})}
        }, b_left=False)
        fwm0_2 = f_dict({
            Dagger(a_s): {(None,): PF({((a_p**2,), ()): dsd2(self.gc*t_, dict(),
                                                             True, S.Zero, -I*nlc/self.gc)})},
            a_i: {(None,): PF({((), ()): dsd2(self.gc*t_, dict(),
                                              True, S.One, self.hpm/self.gc)})}
        }, b_left=True)
        sol_0 = (fwm0_0, fwm0_1, fwm0_2)

        # Order 1: First correction to pump (pump depletion)
        xpm_c = {(None,): PF({((a_p,), ()): S.Half*MultimodeNum2(a_s.name, a_i.name)})}
        coeff = {(None,): PF({((a_p,), ()): S.Half})}
        xpm_t1 = self.e_mul(sol_0[1], self.d_adjoint(sol_0[1]))
        xpm_t2 = self.e_mul(self.d_adjoint(sol_0[2]), sol_0[2])
        xpm_t = self.d_add(xpm_t1, xpm_t2)
        dd2 = self.e_mul(self.e_mul(sol_0[1], self.d_adjoint(sol_0[0])), sol_0[2])
        fwm1p = dict()
        for k, v in self.d_add(dd2, xpm_c, self.e_mul(coeff, xpm_t)).items():
            rhs = self.pe.expand_pf(v.n_apply(lambda t:
                                              -2*I*nlc*t.subs({t_: _g/self.gc})/self.gc), 1)
            s0 = rhs.n_apply(lambda t: self.n_simp(dsd1(_g, self.p2s_group(t), subs=self.gc*t_)))
            fwm1p[k] = PF.add(sol_0[0][(None,)], s0) if len(k) == 1 else s0

        sol_1 = (fwm1p, sol_0[1], sol_0[2])
        self.sol = [sol_m1, sol_0, sol_1]

    def get_sol(self, order: int):
        """
        Get the solution tuple for the given order (-1, 0, or 1).
        """
        if order < -1 or order > 1:
            raise ValueError(f"Order {order} out of range for"
                             " FWMSolver (supports orders -1, 0, 1)")
        return self.sol[order + 1]

    def sol_expr(self, mode: int, order: int):
        """
        Get the expression of the solution for the given mode and order (up to order 1).
        For the pump mode (mode 0), includes the SPM phase factor
        exp(-i * χ * N_0 * t) for order >= 0.
        """
        sol_dict = self.get_sol(order)[mode]
        expr = self.to_expr(sol_dict)
        if mode == 0 and order >= 0:
            return expr * exp(-I * nlc * self.nd * t_)
        return expr
    def _zq(self, obj):
        """
        Assumes that the phase mismatch exactly compensates
        the self-phase modulation of the strong field,
        and evaluates the expression accordingly.

        :param obj: Expr, PF, dict, f_dict, or WmWrapper.
        """

        if hasattr(obj, 'val'):
            obj = obj.val

        return d_apply(obj, lambda x: nd_simp(x.subs({self.pm: 2*nlc*self.nd}), self.nd))


TWM = TWMSolver
FWM = FWMSolver
