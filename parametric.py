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
from op_patterns.boson_pattern import BosonNum
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from perturb_eval import PF, PE
from dsolve_pm import group_terms, dsd1, dsd2, _ps
from collections import defaultdict
from itertools import product
from functools import cached_property, cache, lru_cache
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

    def swap_exps(self, exp1, exp2, order=0, left_p=True):
        """
        Result for rewriting self ** exp1 * self† ** exp2
        into Σ l(N) * self† ** e2 * self ** e1 * r(N)
        
        l(N) / r(N) is 1 if left_p is set True / False

        In other words, the 'left_p' parameter dictates whether the exponents of 
        MultimodeCOP is left- or right- multiplied to the number operators
        """
        order = min(order, exp1, exp2)
        m, n = (exp2, exp1) if left_p else (exp1, exp2)
        s = 1 if (self.annih1 & 1) else -1
        p =  1 if ((self.parity + left_p) & 1) else -1
        c0 = []
        c1 = []
        for _ in range(order):
            m1 = m - 1
            c0.append(s*m*(MultimodeNum2(*self.names, self.parity, s*p*m1)))
            c1.append(-2*p*m)
            m = m1
        result = [Mul(Dagger(self)**exp2, self**exp1)]
        factors = []
        sub = {_mcc_indices[order & 1]: n}
        for o in range(1, order + 1):
            e1, e2 = exp1 - o, exp2 - o
            for i, term in mm_comm_dummy(o).items():
                factors.append(term.subs(sub))
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
    Σ(N_1 + N_2 + 1) + displacement * (n_modes + 1).
    """

    boson_num_only = True

    def __new__(cls, name1, name2, parity=0, displacement=0):
        name1 = sympify(name1)
        name2 = sympify(name2)
        if name1 == name2:
            raise ValueError('The two operators must be in different modes')
        parity = bool(parity)
        if name2 is None or name1 is not None and name1.compare(name2) == 1:
            return (-1)**parity*HermitianOperator.__new__(cls, name2, name1, parity, displacement)
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
    if e.is_Add:
        return Add(*(mmn_shift(arg, mmop, pd) for arg in e.args))
    if e.is_Mul:
        return Mul(*(mmn_shift(arg, mmop, pd) for arg in e.args))
    if e.is_Pow:
        return mmn_shift(e.base, mmop, pd)**e.exp
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

def bpswap(a: BosonOp, p: PF, s, to_left=True):
    def mmn_shift(mmn):
        if mmn.name == a.name:
            if a.is_annihilation ^ to_left:
                return mmn+1
            return mmn-1
        return mmn
    def mmn2_shift(mmn2):
        n1, n2 = mmn2.args[:2]
        if a.name == n2:
            if a.is_annihilation ^ to_left:
                return mmn2+1
            return mmn2-1
        if a.name == n1:
            if mmn2.parity ^ a.is_annihilation == to_left:
                return mmn2-1
            return mmn2+1
        return mmn2
    return p.n_apply(lambda e: s*e.replace(MultimodeNum, mmn_shift)
                                  .replace(MultimodeNum2, mmn2_shift))

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

# Dummy symbols for:
# the MultimodeNum of the signal field
# the MultimodeNum2 of the process
# the BosonNum of the driving field
# the total number of modes
_n1, _n2, _nd, _nm = symbols('n1 n2 nd nm', Dummy=True, nonnegative=True)
_g = Dummy('g', real=True)  # Dummy symbol for the parametric gain

PF0 = PF()  # Singleton null PatternForm

# Time / nonlinear coefficient symbols: t, χ
t_ = Dummy('t', real=True)
nlc = Symbol('χ', real=True, nonnegative=True)
# Field names: a0 - pump (high frequency), a1, a2 - signal and idler (low frequency)
a_p = BosonOp('a_0')
a_s = BosonOp('a_1')
a_i = BosonOp('a_2')
pm = Symbol('Δk', real=True)

class TWMSolver:
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
        self.dc = bool(dc)
        self.pm = pm if mismatch else S.Zero
        self.half_p_factor = exp(I * self.pm * t_ * S.Half)
        if dc:
            # driving field number operator
            self.nd = BosonNum(a_p.name)
            self.a = (MultimodeCOP(a_s.name, a_i.name, annih1=0),
                      MultimodeCOP(a_s.name, a_i.name))
            self.sol = [({(None,): PF({((a_p,), ()): S.One})},
                         {a_s: {(None,): PF0}, Dagger(a_i): {(None,): PF0}},
                         {Dagger(a_s): {(None,): PF0}, a_i: {(None,): PF0}})]
            # gc = g/t
            self.gc = (nlc**2*self.nd-self.pm**2/4)**S.Half
            pe = PE({a_p.name: S.NegativeOne, nlc: S.One})
        else:
            self.nd = BosonNum(a_i.name)
            self.a = (MultimodeCOP(a_p.name, a_s.name, 1, 0),
                      MultimodeCOP(a_p.name, a_s.name, 1))
            self.sol = [({a_p: {(None,): PF0}, a_s: {(None,): PF0}},
                         {a_p: {(None,): PF0}, a_s: {(None,): PF0}},
                         {(None,): PF({((a_i,), ()): S.One})})]
            self.gc = (nlc**2*self.nd+self.pm**2/4)**S.Half
            pe = PE({a_i.name: S.NegativeOne, nlc: S.One})
        pe.register(MultimodeNum)
        pe.register(MultimodeNum2)
        pe.register(MultimodeCOP)
        # d/dt a_drive / (-i χ) for the last solved order
        self.dd_last = {(None,): PF0}
        # Standard expansion of an expression up to the given order
        self.expand = (lambda pf, order:
                       pe.expand_pf(pf, order).n_apply(lambda t:
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
        
    def to_expr(self, x, b_left=True):
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
            if b_left:
                return Add(*(self.to_expr(b)*self.to_expr(e) for b, e in x.items()))
            return Add(*(self.to_expr(e)*self.to_expr(b) for b, e in x.items()))
        raise ValueError('Got unexpected format')

    def d_adjoint(self, d: dict):
        """
        Takes adjoint for the dict representation of field solutions
        """
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

    def d_mul(self, d0, d1, d2, mid=None):
        if isinstance(mid, MultimodeNum):
            for (k2, e2) in d2.items():
                pd = _pd(k2)
                for (k1, e1) in d1.items():
                    k = t_mul(k1, k2)
                    new = e1.n_apply(lambda t: mmn_shift(mid*t, self.a[0], pd))
                    if k in d0:
                        d0[k] = PF.add(d0[k], new.mul_ind(e2))
                    else:
                        d0[k] = new.mul_ind(e2)
        else:
            for (k2, e2) in d2.items():
                pd = _pd(k2)
                for (k1, e1) in d1.items():
                    k = t_mul(k1, k2, mid)
                    new = e1.n_apply(lambda t: mmn_shift(t, self.a[0], pd))
                    if k in d0:
                        d0[k] = PF.add(d0[k], new.mul_ind(e2))
                    else:
                        d0[k] = new.mul_ind(e2)
    
    def e_mul(self, p1: dict, p2: dict):
        """
        BosonOp is right-multiplied to p1 and left-multiplied to p2 by default
        
        if both dict contain BosonOps, this will generate an intermidiate dict
        including MultimodeNums (to be further collected as MultimodeNum2)
        """
        result = dict()
        if isinstance(next(iter(p1)), BosonOp):
            if isinstance(next(iter(p2)), BosonOp):
                for (b1, d1), (b2, d2) in product(p1.items(), p2.items()):
                    if b1 == Dagger(b2):
                        mn = (MultimodeNum(b1.name, 1) if b1.is_annihilation
                              else MultimodeNum(b1.name))
                        self.d_mul(result, d1, d2, mn)
                    elif b1.compare(b2) == 1:
                        self.d_mul(result, d1, d2, b2.is_annihilation)
                    else:
                        self.d_mul(result, d1, d2, b1.is_annihilation)
            else:
                for b, d in p1.items():
                    result[b] = dict()
                    self.d_mul(result[b], d, p2)
        elif isinstance(next(iter(p2)), BosonOp):
            for b, d in p2.items():
                result[b] = dict()
                self.d_mul(result[b], p1, d)
        else:
            self.d_mul(result, p1, p2)
        return result

    def bdswap(self, a: BosonOp, d, to_left=True):
        def _od(b: BosonOp):
            n1, n2, p = self.a[0].args[:3]
            if b.name == n1:
                return BosonOp(n2, b.is_annihilation==p)
            if b.name == n2:
                return BosonOp(n1, b.is_annihilation==p)
            return b
        def btswap(a: BosonOp, i, exps):
            yield S.One, a, exps
            if to_left:
                for j, e in enumerate(exps):
                    if i ^ (j & 1) ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                        for s, b, nes in btswap(_od(a), i, exps[:j]):
                            yield S.NegativeOne**int(a.is_annihilation)*s*e, b, (*nes, e-1, *exps[j+1:])
            else:
                for j, e in enumerate(exps):
                    ic = i ^ (j & 1)
                    if ic ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                        for s, b, nes in btswap(_od(a), not ic, exps[j+1:]):
                            yield S.NegativeOne**int(not a.is_annihilation)*s*e, b, (*exps[:j], e-1, *nes)
        new_dict = {a: dict(), _od(a): dict()}
        for t, p in d.items():
            i, *exps = t
            for s, b, nes in btswap(a, i, exps):
                key = _tdz(i, nes)
                if key in new_dict[b]:
                    new_dict[b][_tdz(i, nes)] = PF.add(new_dict[b][_tdz(i, nes)], bpswap(b, p, s, to_left))
                else:
                    new_dict[b][_tdz(i, nes)] = bpswap(b, p, s, to_left)
        return new_dict
                
    def n_simp(self, expr):
        n1, n2 = self.a[0].names
        def mmn_d(n, d):
            if n == a_s.name:
                return _n1 + d*_nm
            if self.dc:
                return _n2 - _n1 + (d-1) * _nm
            return _n1 - _n2 + d * _nm
        def combine_n(expr):
            if expr.is_Add and any(any(isinstance(f, MultimodeNum)
                                       for f in Mul.make_args(arg))
                                   for arg in expr.args):
                r = expr.replace(MultimodeNum, mmn_d).replace(MultimodeNum2,
                                                              lambda *args: _n2+args[-1]*_nm)
                c_m = r.subs({_n2: 0})
                c = c_m.subs({_nm: 0})
                c_n2 = ((r.subs({_nm: 0}) - c).expand() / _n2).doit()
                d =(((c_m - c) / _nm).expand() / c_n2).doit()
                # non-integer displacement is also legal here
                if d.is_number:
                    expr = (c_n2 * _n2 + c).factor().subs({_n1: MultimodeNum(a_s.name),
                                                           _n2: MultimodeNum2(n1, n2, not self.dc, d)})
                else:
                    warnings.warn('Failed to combine the weak-field multimode number operators')
            if expr.is_Mul:
                n_part = []
                other = []
                for arg in Mul.make_args(expr.subs({self.nd: _nd})):
                    if arg.has(_nd):
                        n_part.append(arg)
                    else:
                        other.append(arg)
                expr = Mul(*other, Mul(*n_part).subs({_nd: self.nd}))

            expr = _ps(expr)
            return expr
        return bottom_up(expr, combine_n)
                            
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
                    rhs = self.expand(PF.add(*pfs).n_apply(lambda t: Mul(nlc**2, t, gc_m2)), order)
                    s0 = rhs.n_apply(lambda t: self.n_simp(dsd2(_g, self.p2s_group(t), self.dc,
                                                                subs=self.gc*t_)))
                    self.sol[i][mode][b][k] = PF.add(s0, self.fdf0(n0, mode)) if len(k) == 1 else s0
            if dc:
                self.sol.append((last_sol[0],
                                 {a_s: dict(), Dagger(a_i): dict()},
                                 {Dagger(a_s): dict(), a_i: dict()}))
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
                self.sol.append(({a_p: dict(), a_s: dict()},
                                 {a_p: dict(), a_s: dict()},
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
        return p*self.to_expr(self.get_sol(order)[mode], False)

    def p2s_group(self, expr):
        """
        Group the functions of g and process product-to-sum to
        hyperbolic/trangular functions
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

def sep_weak(expr):
    """
    Separate the weak field multimode number operators from the given expr
    """
    result = dict()
    for term in Add.make_args(expr):
        other = []
        mmn = []
        for factor in Mul.make_args(term):
            if isinstance(factor, (MultimodeNum, MultimodeNum2)):
                mmn.append(factor)
            else:
                other.append(factor)
        result[tuple(mmn)] = Mul(*other)
    return result