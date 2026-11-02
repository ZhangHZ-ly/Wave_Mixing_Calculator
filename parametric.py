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
from sympy.core.containers import Dict
from sympy.functions.elementary.complexes import Abs, conjugate
from op_patterns.boson_pattern import BosonNum, NumShift
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from perturb_eval import PF, PE, bsc, pattern_form, cohr_avr
from dsolve_pm import group_terms, dsd1, dsd2, d_p2s, _ps
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
    Σ(N_1 + N_2 + 1) + displacement * (n_modes + 1).
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

def bpswap(a: BosonOp, p: PF, s, to_left=True):
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

def d_apply(e, f):
    if isinstance(e, f_dict):
        return f_dict({k: d_apply(v, f) for k, v in e.items()}, b_left=e.b_left)
    if isinstance(e, dict):
        return {k: d_apply(v, f) for k, v in e.items()}
    if isinstance(e, PF):
        try:
            return e.n_apply(f)
        except (TypeError, AttributeError):
            pass
    return f(e)

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
        'N_t': PF({((), ()): MultimodeNum2(a_s.name, a_i.name, 0)}),
    },
    False: {
        'a_0': f_dict({a_p: {(None,): PF_ONE}}, b_left=True),
        'a_1': f_dict({a_s: {(None,): PF_ONE}}, b_left=True),
        'a_2': PF({((a_i,), ()): S.One}),
        'A': {(True, 1): PF_ONE},
        'N_t': PF({((), ()): MultimodeNum2(a_p.name, a_s.name, 1)}),
    },
}

class WM_Processor:
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
        if isinstance(mismatch, (Expr, Symbol)):
            self.pm = mismatch
        else:
            self.pm = pm if mismatch else S.Zero
        self.sol = []

    def update_pe(self, orders: dict = None):
        """Update self.pe assumptions with custom symbol/mode orders."""
        assumptions = dict(self.base_assumptions)
        if orders:
            for k, v in orders.items():
                if v is not None:
                    assumptions[k] = sympify(v)
        self.pe.assumptions = Dict(assumptions)
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
                        for nk, nv in self.bdswap(k, v, to_left=b_left).items():
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
        Multiplies two dict representations of operators.
        
        if both dict contain BosonOps, this will generate an intermediate dict
        including MultimodeNums (to be further collected as MultimodeNum2)
        
        By default, single-mode weak-field operators are assumed to be
        at the edges. If a multiplication causes a single-mode operator
        to be trapped in the middle, bdswap is automatically invoked
        to swap it to the correct boundary.
        """
        if not p1 or not p2:
            return dict()
        is_f1 = isinstance(p1, f_dict)
        is_f2 = isinstance(p2, f_dict)
        result = dict()
        if is_f1 and is_f2:
            if self.f_dict_parity(p1) == self.f_dict_parity(p2):
                raise ValueError('Cannot use multimode operators to represent the product')
            if p1.b_left:
                new_p1 = f_dict(b_left=False)
                for b, d in p1.items():
                    for nb, nd in self.bdswap(b, d, to_left=False).items():
                        if nd: new_p1[nb] = self.d_add(new_p1.get(nb, dict()), nd)
                p1 = new_p1
            if not p2.b_left:
                new_p2 = f_dict(b_left=True)
                for b, d in p2.items():
                    for nb, nd in self.bdswap(b, d, to_left=True).items():
                        if nd: new_p2[nb] = self.d_add(new_p2.get(nb, dict()), nd)
                p2 = new_p2
            for (b1, d1), (b2, d2) in product(p1.items(), p2.items()):
                if b1 == Dagger(b2):
                    mn = (MultimodeNum(b1.name, 1) if b1.is_annihilation else MultimodeNum(b1.name))
                    self.d_mul(result, d1, d2, mn)
                elif b1.compare(b2) == 1:
                    self.d_mul(result, d1, d2, b2.is_annihilation)
                else:
                    self.d_mul(result, d1, d2, b1.is_annihilation)
        elif is_f1:
            if p1.b_left:
                result = f_dict(b_left=True)
                for b, d in p1.items():
                    result[b] = dict()
                    self.d_mul(result[b], d, p2)
            else:
                result = f_dict(b_left=False)
                for b, d1 in p1.items():
                    for nb, p2_prime in self.bdswap(b, p2, to_left=False).items():
                        self.d_mul(result.setdefault(nb, dict()), d1, p2_prime)
        elif is_f2:
            if not p2.b_left:
                result = f_dict(b_left=False)
                for b, d in p2.items():
                    result[b] = dict()
                    self.d_mul(result[b], p1, d)
            else:
                result = f_dict(b_left=True)
                for b, d2 in p2.items():
                    for nb, p1_prime in self.bdswap(b, p1, to_left=True).items():
                        self.d_mul(result.setdefault(nb, dict()), p1_prime, d2)
        else:
            self.d_mul(result, p1, p2)

        return result

    def bdswap(self, a: BosonOp, d, to_left=True):
        """
        Swaps the order of the operator a and the dict d
        The direction of the swap (for a) is determined by the to_left parameter.
        """
        def btswap(a: BosonOp, i, exps):
            yield S.One, a, exps
            if to_left:
                for j, e in enumerate(exps):
                    if i ^ (j & 1) ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                        for s, b, nes in btswap(self._od(a), i, exps[:j]):
                            yield S.NegativeOne**int(a.is_annihilation)*s*e, b, (*nes, e-1, *exps[j+1:])
            else:
                for j, e in enumerate(exps):
                    ic = i ^ (j & 1)
                    if ic ^ (not self.dc and a.name==a_s.name) != a.is_annihilation:
                        for s, b, nes in btswap(self._od(a), not ic, exps[j+1:]):
                            yield S.NegativeOne**int(not a.is_annihilation)*s*e, b, (*exps[:j], e-1, *nes)
        new_dict = f_dict({a: dict(), self._od(a): dict()}, b_left=to_left)
        for t, p in d.items():
            i, *exps = t
            for s, b, nes in btswap(a, i, exps):
                key = _tdz(i, nes)
                if key in new_dict[b]:
                    new_dict[b][_tdz(i, nes)] = PF.add(new_dict[b][_tdz(i, nes)], bpswap(b, p, s, to_left))
                else:
                    new_dict[b][_tdz(i, nes)] = bpswap(b, p, s, to_left)
        return new_dict

    def nd_simp(self, expr):
        """
        Simplifies expressions involving powers of self.nd by:
        1. Extracting common factors of self.nd from Add expressions inside Mul;
        2. Merging powers of self.nd via substitution with commutative dummy symbol _nd;
        """
        def is_pure_nd_power(a):
            if a == _nd:
                return True, S.One
            if a.is_Pow and a.base == _nd:
                return True, a.exp
            return False, S.Zero

        def simp_mul(e):
            if e.is_Mul and e.has(self.nd):
                expr_sub = e.subs({self.nd: _nd})
                if not expr_sub.is_Mul:
                    return expr_sub.subs({_nd: self.nd})
                args = []
                for a in Mul.make_args(expr_sub):
                    if a.is_Add and a.has(_nd):
                        args.extend(Mul.make_args(factor_terms(a, fraction=True)))
                    elif a.is_Pow and a.base.is_Add and a.base.has(_nd):
                        args.extend(arg** a.exp
                                    for arg in Mul.make_args(factor_terms(a.base, fraction=True)))
                    else:
                        args.append(a)

                nd_powers = []
                add_factors = []
                other_nd = []
                other = []
                for arg in args:
                    is_p, p_exp = is_pure_nd_power(arg)
                    if is_p:
                        nd_powers.append(p_exp)
                    elif arg.is_Add and arg.has(_nd):
                        add_factors.append(arg)
                    elif arg.has(_nd):
                        other_nd.append(arg)
                    else:
                        other.append(arg)

                total_nd_exp = Add(*nd_powers) if nd_powers else S.Zero
                if total_nd_exp != S.Zero and add_factors:
                    target_idx = 0
                    if len(add_factors) > 1:
                        for i, af in enumerate(add_factors):
                            if total_nd_exp > 0 and any(t.has(_nd) and
                                                        any(is_pure_nd_power(f)[0]
                                                            and is_pure_nd_power(f)[1] < 0
                                                            for f in Mul.make_args(t))
                                                            for t in af.args):
                                target_idx = i
                                break
                            elif total_nd_exp < 0 and any(t.has(_nd)
                                                          and any(is_pure_nd_power(f)[0]
                                                                  and is_pure_nd_power(f)[1] > 0
                                                                  for f in Mul.make_args(t))
                                                                  for t in af.args):
                                target_idx = i
                                break
                    nd_factor = _nd**total_nd_exp
                    first_add = add_factors[target_idx]
                    new_terms = [Mul(term, nd_factor) for term in first_add.args]
                    add_factors[target_idx] = Add(*new_terms)
                    total_nd_exp = S.Zero

                n_factors = list(other_nd)
                if total_nd_exp != S.Zero:
                    n_factors.append(_nd**total_nd_exp)
                n_factors.extend(add_factors)

                n_part = Mul(*n_factors).subs({_nd: self.nd})
                return Mul(*other, n_part)
            return e
        return bottom_up(expr, simp_mul)

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
                                                           _n2: MultimodeNum2(n1, n2, not self.dc, d)})
                else:
                    warnings.warn('Failed to combine the weak-field multimode number operators')
            if expr.is_Mul:
                expr = self.nd_simp(expr)
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
        return self.nd_simp(expr)

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
                res = PF.add(*(self._smk(k).mul_ind(self.to_single_mode(v)) for k, v in obj.items()))
            else:
                res = PF.add(*(self.to_single_mode(v).mul_ind(self._smk(k)) for k, v in obj.items()))
            return res.n_apply(self.nd_simp)
        elif isinstance(obj, PF):
            return obj.n_apply(lambda t: self._sm_expr(t))
        elif isinstance(obj, Expr):
            return pattern_form(self._sm_expr(obj), as_dict=None)

        return self.to_single_mode(sympify(obj))

    def eval_vacuum(self, obj, vac_modes, single_mode=False):
        """
        Evaluates the partial or full expectation value of an operator expression
        under vacuum input assertions for the specified weak-field mode(s).

        Uses MultimodeCOP.swap_exps to exchange powers of multimode canonical
        operators and eagerly collapses generated MultimodeCOP, MultimodeNum,
        and MultimodeNum2 operators using the active vacuum mode presets.

        :param obj: Expr, PF, dict, f_dict, or wrapper with `.val`.
        :param vac_modes: Iterable of mode names / BosonOps asserted to be in vacuum.
        :param single_mode: Whether single-mode processing (n_m = 1) is enabled.
        :return: Collapsed PF, dict, or f_dict representation.
        """
        if hasattr(obj, 'val'):
            obj = obj.val

        vac_set = {
            sympify(m.name if hasattr(m, 'name') else m)
            for m in (vac_modes or ())
        }
        if not vac_set:
            return self.to_single_mode(obj) if single_mode else obj

        nm_sym = S.One if single_mode else _nm
        weak_names = set(self.a[0].names)
        weak_vac = vac_set & weak_names
        base_d = S.One if self.dc else S.Zero

        def _collapse_mmn(x):
            if x.name in vac_set:
                return x.displacement * nm_sym
            if len(weak_vac) == 1 and x.name in weak_names and x.displacement != base_d:
                return MultimodeNum(x.name, base_d) + (x.displacement - base_d) * nm_sym
            return x

        def _collapse_mmn2(x):
            n1, n2 = x.names
            p = x.parity
            d = x.displacement
            in1 = n1 in vac_set
            in2 = n2 in vac_set
            if not in1 and not in2:
                return x
            if not p:
                if in1 and in2:
                    return (1 + d) * nm_sym
                n_surv = n2 if in1 else n1
                return MultimodeNum(n_surv, 1) + d * nm_sym
            else:
                if in1 and in2:
                    return d * nm_sym
                if in1:
                    return MultimodeNum(n2, 0) + d * nm_sym
                return -MultimodeNum(n1, 0) + d * nm_sym

        def _collapse_num(expr):
            if not isinstance(expr, Expr) or expr.is_number:
                return expr
            res = expr.replace(
                lambda x: isinstance(x, (BosonNum, MultimodeNum, MultimodeNum2)),
                lambda x: (
                    (S.Zero if x.name in vac_set else x) if isinstance(x, BosonNum)
                    else (_collapse_mmn(x) if isinstance(x, MultimodeNum)
                          else _collapse_mmn2(x))
                )
            )
            return self._sm_expr(res) if single_mode else self.n_simp(res)

        def _vac_pf(pf: PF) -> PF:
            if not pf.table and pf.npf == S.Zero:
                return PF0
            new_table = {}
            for (bs, fs), tail in pf.table.items():
                if any(b.name in vac_set for b in bs):
                    continue
                c_tail = _collapse_num(tail)
                if c_tail != S.Zero:
                    new_table[(bs, fs)] = c_tail
            new_npf = _collapse_num(pf.npf) if pf.npf != S.Zero else S.Zero
            return PF(new_table, new_npf) if (new_table or new_npf != S.Zero) else PF0

        tuple_cache = {}

        def _vac_tuple(k: tuple):
            if k in tuple_cache:
                return tuple_cache[k]
            if len(k) == 1:
                tuple_cache[k] = S.One
                return S.One
            if not self.dc and len(weak_vac) == 2:
                tuple_cache[k] = S.Zero
                return S.Zero

            if self.dc:
                ann_idx = True
            elif a_s.name in weak_vac:
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
            if not weak_vac:
                res = {}
                for k, pf in d.items():
                    vp = _vac_pf(pf)
                    if vp.table or vp.npf != S.Zero:
                        res[k] = vp
                return res if res else PF0
            pfs = []
            for k, pf in d.items():
                ck = _vac_tuple(k)
                if ck == S.Zero:
                    continue
                vp = _vac_pf(pf)
                if not vp.table and vp.npf == S.Zero:
                    continue
                pfs.append(vp.n_apply(lambda t, c=ck: _collapse_num(c * t)))
            return PF.add(*pfs) if pfs else PF0

        def _vac_f_dict(fd: f_dict):
            if len(weak_vac) == 2:
                return PF0
            if not weak_vac:
                res = f_dict(b_left=fd.b_left)
                for b, d in fd.items():
                    vd = _vac_dict(d)
                    if isinstance(vd, dict) and vd:
                        res[b] = vd
                    elif isinstance(vd, PF) and (vd.table or vd.npf != S.Zero):
                        res[b] = {(None,): vd}
                return res if res else PF0

            fds = []
            for b, d in fd.items():
                if not d:
                    continue
                if b.name in vac_set:
                    if (b.is_annihilation and not fd.b_left) or (not b.is_annihilation and fd.b_left):
                        continue
                    swapped_fd = self.bdswap(b, d, to_left=not b.is_annihilation)
                    b_partner = self._od(b)
                    d_partner = swapped_fd.get(b_partner)
                    if d_partner:
                        vp = _vac_dict(d_partner)
                        if isinstance(vp, PF) and (vp.table or vp.npf != S.Zero):
                            fds.append(f_dict({b_partner: {(None,): vp}}, b_left=swapped_fd.b_left))
                else:
                    vp = _vac_dict(d)
                    if isinstance(vp, PF) and (vp.table or vp.npf != S.Zero):
                        fds.append(f_dict({b: {(None,): vp}}, b_left=fd.b_left))

            if not fds:
                return PF0
            combined = self.d_add(*fds)
            cleaned = f_dict(
                {
                    b: {k: pf for k, pf in sub.items() if pf.table or pf.npf != S.Zero}
                    for b, sub in combined.items()
                },
                b_left=combined.b_left,
            )
            cleaned = f_dict({b: sub for b, sub in cleaned.items() if sub}, b_left=combined.b_left)
            return cleaned if cleaned else PF0

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

        if single_mode and not isinstance(res, PF):
            res = self.to_single_mode(res)
        return res

    def eval_coherent(self, obj, order: int, alpha=None, gain_sym=None, return_both=False):
        """
        Evaluates an operator expression (in time t_ representation) under the
        assertion that the strong driving field input is in a coherent state |alpha>,
        using `coherent_average_terms` from `perturb_eval.state_eval` up to `order`
        and substituting `self.nd` with `|alpha|^2`.

        :param obj: Expr, PF, dict, f_dict, or wrapper with `.val`.
        :param order: Target perturbative expansion order.
        :param alpha: Complex coherent amplitude Symbol (defaults to alpha_0 / alpha_2).
        :param gain_sym: Symbol used for classical parametric gain g when return_both=True.
        :param return_both: If True, returns (res_t, res_g) simultaneously.
        :return: Evaluated representation in t_ (or (res_t, res_g) if return_both=True).
        """
        if hasattr(obj, 'val'):
            obj = obj.val

        strong_name = self.nd.name
        if alpha is None:
            alpha = Symbol('alpha_0') if self.dc else Symbol('alpha_2')
        if gain_sym is None:
            gain_sym = Symbol('g', real=True)
        alpha_sq = Abs(alpha) ** 2
        nd_step = max(1, abs(int(self.pe.leading_order(self.nd)[1])))

        def _cohr_tail(norm_tail, rem_order):
            if norm_tail == S.Zero:
                return (S.Zero, S.Zero)
            by_ord = defaultdict(list)
            for t_sub, t_ord in self.pe.expand_expr(norm_tail, rem_order):
                if t_sub != S.Zero and t_ord <= rem_order:
                    by_ord[t_ord].append(t_sub)
            if not by_ord:
                return (S.Zero, S.Zero)

            ca_parts = []
            for t_ord, terms in by_ord.items():
                sub_expr = Add(*terms)
                if sub_expr == S.Zero:
                    continue
                if not sub_expr.has(self.nd):
                    ca_parts.append(sub_expr)
                else:
                    lmax = max(0, int((rem_order - t_ord) // nd_step))
                    sub_comm = sub_expr.subs({self.nd: _nd})
                    ca_term = Add(*cohr_avr(sub_comm, (_nd,), lmax)).subs({_nd: self.nd})
                    ca_parts.append(ca_term)

            if not ca_parts:
                return (S.Zero, S.Zero)
            ca_raw = Add(*ca_parts)
            ca_trunc = Add(*(e[0] for e in self.pe.expand_expr(ca_raw, rem_order)))
            if ca_trunc == S.Zero:
                return (S.Zero, S.Zero)

            if rem_order >= 0 or ca_trunc.has(t_):
                ca_g_op = self.n_simp(d_p2s(_g, self.p2s_group(ca_trunc.subs({t_: _g / self.gc}))))
            else:
                ca_g_op = self.n_simp(ca_trunc)

            expr_t = bottom_up(self.nd_simp(ca_g_op.subs({_g: self.gc * t_})).subs({self.nd: alpha_sq}), _ps)
            if return_both:
                expr_g = bottom_up(self.nd_simp(ca_g_op).subs({_g: gain_sym, self.nd: alpha_sq}), _ps)
            else:
                expr_g = S.Zero
            return (expr_t, expr_g)

        def _cohr_pf(pf: PF, pf_order: int):
            if not pf.table and pf.npf == S.Zero:
                return (PF0, PF0)
            table_t = {}
            table_g = {}
            for (bs, fs), tail in pf.table.items():
                b_strong = None
                bs_rest_list = []
                for b in bs:
                    if b.name == strong_name:
                        b_strong = b
                    else:
                        bs_rest_list.append(b)
                bs_rest = tuple(bs_rest_list)
                rest_ord = Add(*(self.pe.leading_order(x)[1] for x in (*bs_rest, *fs)))

                if b_strong is None:
                    b_ord = S.Zero
                    c_amp = S.One
                    norm_tail = tail
                elif b_strong.left_base.is_annihilation:
                    m = b_strong.exps[0]
                    b_ord = self.pe.leading_order(b_strong)[1]
                    c_amp = alpha ** m
                    norm_tail = NumShift(tail, -1, b_strong).doit()
                else:
                    m = b_strong.exps[0]
                    b_ord = self.pe.leading_order(b_strong)[1]
                    c_amp = conjugate(alpha) ** m
                    norm_tail = tail

                rem_order = pf_order - b_ord - rest_ord
                et, eg = _cohr_tail(norm_tail, rem_order)
                if et != S.Zero:
                    key = (bs_rest, fs)
                    val_t = table_t.get(key, S.Zero) + c_amp * et
                    if val_t != S.Zero:
                        table_t[key] = val_t
                    elif key in table_t:
                        del table_t[key]
                if return_both and eg != S.Zero:
                    key = (bs_rest, fs)
                    val_g = table_g.get(key, S.Zero) + c_amp * eg
                    if val_g != S.Zero:
                        table_g[key] = val_g
                    elif key in table_g:
                        del table_g[key]

            npf_t, npf_g = _cohr_tail(pf.npf, pf_order) if pf.npf != S.Zero else (S.Zero, S.Zero)
            pf_t = PF(table_t, npf_t) if (table_t or npf_t != S.Zero) else PF0
            pf_g = PF(table_g, npf_g) if (return_both and (table_g or npf_g != S.Zero)) else PF0
            return (pf_t, pf_g)

        def _cohr_dict(d: dict, d_order: int):
            res_t = {}
            res_g = {}
            for k, pf in d.items():
                pt, pg = _cohr_pf(pf, d_order)
                if pt.table or pt.npf != S.Zero:
                    res_t[k] = pt
                if return_both and (pg.table or pg.npf != S.Zero):
                    res_g[k] = pg
            out_t = res_t if res_t else PF0
            out_g = res_g if (return_both and res_g) else PF0
            return (out_t, out_g)

        def _cohr_f_dict(fd: f_dict, fd_order: int):
            res_t = f_dict(b_left=fd.b_left)
            res_g = f_dict(b_left=fd.b_left)
            for b, d in fd.items():
                b_ord = self.pe.leading_order(b)[1]
                dt, dg = _cohr_dict(d, fd_order - b_ord)
                if isinstance(dt, dict) and dt:
                    res_t[b] = dt
                if return_both and isinstance(dg, dict) and dg:
                    res_g[b] = dg
            out_t = res_t if res_t else PF0
            out_g = res_g if (return_both and res_g) else PF0
            return (out_t, out_g)

        if isinstance(obj, f_dict):
            rt, rg = _cohr_f_dict(obj, order)
        elif isinstance(obj, dict):
            rt, rg = _cohr_dict(obj, order)
        elif isinstance(obj, PF):
            rt, rg = _cohr_pf(obj, order)
        elif isinstance(obj, Expr):
            rt, rg = _cohr_pf(pattern_form(obj, as_dict=None), order)
        else:
            rt, rg = _cohr_pf(pattern_form(sympify(obj), as_dict=None), order)

        return (rt, rg) if return_both else rt


class TWMSolver(WM_Processor):
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
            self.nd = BosonNum(a_p.name)
            self.a = (MultimodeCOP(a_s.name, a_i.name, annih1=0),
                      MultimodeCOP(a_s.name, a_i.name))
            # Mark multiplication directions: a_s/a_i right-multiplied (False), a_s†/a_i† left-multiplied (True)
            self.sol = [({(None,): PF({((a_p,), ()): S.One})},
                         f_dict({a_s: {(None,): PF0}, Dagger(a_i): {(None,): PF0}}, b_left=False),
                         f_dict({Dagger(a_s): {(None,): PF0}, a_i: {(None,): PF0}}))]
            self.gc = S.Half*(4*nlc**2*self.nd-self.pm**2)**S.Half
            self.base_assumptions = {a_p.name: S.NegativeOne, nlc: S.One}
        else:
            self.nd = BosonNum(a_i.name)
            self.a = (MultimodeCOP(a_p.name, a_s.name, 1, 0),
                      MultimodeCOP(a_p.name, a_s.name, 1))
            self.sol = [(f_dict({a_p: {(None,): PF0}, a_s: {(None,): PF0}}),
                         f_dict({a_p: {(None,): PF0}, a_s: {(None,): PF0}}),
                         {(None,): PF({((a_i,), ()): S.One})})]
            self.gc = S.Half*(4*nlc**2*self.nd+self.pm**2)**S.Half
            self.base_assumptions = {a_i.name: S.NegativeOne, nlc: S.One}
        self.pe = PE(self.base_assumptions)
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
                    rhs = self.expand(PF.add(*pfs).n_apply(lambda t: Mul(nlc**2, t, gc_m2)), order)
                    s0 = rhs.n_apply(lambda t: self.n_simp(dsd2(_g, self.p2s_group(t), self.dc,
                                                                subs=self.gc*t_)))
                    self.sol[i][mode][b][k] = PF.add(s0, self.fdf0(n0, mode)) if len(k) == 1 else s0
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


class FWMSolver(WM_Processor):
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
        self.nd = BosonNum(a_p.name)
        self.a = (MultimodeCOP(a_s.name, a_i.name, annih1=0),
                  MultimodeCOP(a_s.name, a_i.name))
        self.hpm = -I*self.pm*S.Half + I*nlc*self.nd
        self.gc = S.Half*(self.pm * (-self.pm + 4*nlc*self.nd))**S.Half
        self.base_assumptions = {a_p.name: -S.Half, nlc: S.One}
        self.pe = PE(self.base_assumptions)
        self.pe.register(MultimodeNum)
        self.pe.register(MultimodeNum2)
        self.pe.register(MultimodeCOP)
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
            a_s: {(None,): PF({((), ()): dsd2(self.gc*t_, dict(), True, S.One, self.hpm/self.gc)})},
            Dagger(a_i): {(None,): PF({((a_p**2,), ()): dsd2(self.gc*t_, dict(), True, S.Zero, -I*nlc/self.gc)})}
        }, b_left=False)
        fwm0_2 = f_dict({
            Dagger(a_s): {(None,): PF({((a_p**2,), ()): dsd2(self.gc*t_, dict(), True, S.Zero, -I*nlc/self.gc)})},
            a_i: {(None,): PF({((), ()): dsd2(self.gc*t_, dict(), True, S.One, self.hpm/self.gc)})}
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
            rhs = self.pe.expand_pf(v.n_apply(lambda t: -2*I*nlc*t.subs({t_: _g/self.gc})/self.gc), 1)
            s0 = rhs.n_apply(lambda t: self.n_simp(dsd1(_g, self.p2s_group(t), subs=self.gc*t_)))
            fwm1p[k] = PF.add(sol_0[0][(None,)], s0) if len(k) == 1 else s0

        sol_1 = (fwm1p, sol_0[1], sol_0[2])
        self.sol = [sol_m1, sol_0, sol_1]

    def get_sol(self, order: int):
        """
        Get the solution tuple for the given order (-1, 0, or 1).
        """
        if order < -1 or order > 1:
            raise ValueError(f"Order {order} out of range for FWMSolver (supports orders -1, 0, 1)")
        return self.sol[order + 1]

    def sol_expr(self, mode: int, order: int):
        """
        Get the expression of the solution for the given mode and order (up to order 1).
        For the pump mode (mode 0), includes the SPM phase factor exp(-i * χ * N_0 * t) for order >= 0.
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

        return d_apply(obj, lambda x: self.nd_simp(x.subs({self.pm: 2*nlc*self.nd})))


TWM = TWMSolver
FWM = FWMSolver
