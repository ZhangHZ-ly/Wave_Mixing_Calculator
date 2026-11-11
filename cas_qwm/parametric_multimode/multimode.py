from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.numbers import Integer
from sympy.core.symbol import Dummy
from sympy.core.mul import Mul
from sympy.core.containers import Dict
from sympy.concrete.summations import Sum
from sympy.physics.quantum.operator import Operator, HermitianOperator
from ..op_pattern import BosonNum
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from functools import cached_property, lru_cache

__all__ = ['MultimodeCOP', 'MultimodeNum', 'MultimodeNum2']

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
        return Mul(BosonOp(self.names[0], self.annih1),
                   BosonOp(self.names[1], self.annih2))

    def swap_exps(self, exp1, exp2, order=None, left_p=True):
        """
        Yields terms of [self ** exp1, self† ** exp2]
        with form Σ l(N) * self† ** e2 * self ** e1 * r(N)
        
        l(N) / r(N) is 1 if left_p is set True / False

        In other words, the 'left_p' parameter dictates whether the exponents of 
        MultimodeCOP is left- or right- multiplied to the number operators
        """
        order = min(exp1, exp2) if order is None else min(order, exp1, exp2)
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
                    yield Dagger(self)**e2, self**e1, Mul(*factors)
                else:
                    yield Mul(*factors), Dagger(self)**e2, self**e1
                factors.clear()

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
    @property
    def bn_coeffs(self):
        return Dict({self.name: S.One})
    
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

    Note: For the degenerate mode with parity=0, the commutator [a, a†] is counted twice,
    so the exact expression for the sum is in fact Σ(N_1 + N_2 + 1) + 1.
    """

    boson_num_only = True

    def __new__(cls, name1, name2, parity=0):
        name1 = sympify(name1)
        name2 = sympify(name2)
        if name1 == name2:
            raise ValueError('The two operators must be in different modes')
        parity = S.Zero if not bool(parity) else S.One
        if name2 is None or name1 is not None and name1.compare(name2) == 1:
            return (-1)**parity*HermitianOperator.__new__(cls, name2, name1, parity)
        return HermitianOperator.__new__(cls, name1, name2, parity)
    
    @property
    def names(self):
        return self.args[:2]
    @property
    def parity(self):
        return self.args[2]
    @cached_property
    def mode_expr(self):
        a_1, a_2 = self.names
        return (BosonNum(a_2) - BosonNum(a_1) if self.parity
                else BosonNum(a_1) + BosonNum(a_2) + 1)
    @property
    def bn_coeffs(self):
        a_1, a_2 = self.names
        return Dict({a_1: S.NegativeOne**self.parity, a_2: S.One})
    
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
