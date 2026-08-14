import sympy.physics.quantum.boson as boson
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.operator import Operator, HermitianOperator
from sympy.core.symbol import Dummy
from sympy.core.singleton import S
from sympy.core.numbers import Integer
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.containers import Tuple, Dict
from sympy.core.sympify import sympify
from .pattern_exponents import PatExps
from collections.abc import Sequence
from functools import cached_property, lru_cache

__all__ = ['BosonNum', 'BosonPat', 'NumShift', 'apply_num_shift']

boson.BosonOp._eval_adjoint = lambda self: BosonOp(self.name, not self.is_annihilation)
boson.BosonOp._eval_power = lambda self, exp: (BosonPat(*self.args, exp)
                                               if exp.is_commutative and exp.is_real
                                               else Operator._eval_power(self, exp))

@lru_cache(maxsize=None)
def fac(x:int|Integer):
    if x < 0:
        raise ValueError('Expect nonnegative input')
    if x == 0:
        return S.One if x is S.Zero else 1
    return x * fac(x-1)

@lru_cache
def ffc(x:Integer, y):
    if x < 0 or y < 0:
        raise ValueError('Expect nonnegative Integer input')
    if y == 0:
        return S.One
    return ffc(x, y - 1) * (x - y + 1)

class BosonPat(Operator):
    @classmethod
    def default_args(self):
        return ('a', True)

    def __new__(cls, *args, eval_exp=True, **hints):
        if isinstance(args[0], cls) and 'merge_ends' in hints:
            if len(args) == 1:
                return args[0]
            if isinstance(args[0], BosonOp):
                la = args[0].args[1]
                pe = PatExps(S.One)
            elif isinstance(args[0], BosonNum):
                la = S.Zero
                pe = PatExps(S.One, S.One)
            else:
                la = args[0].args[1]
                pe =  args[0].args[2]
            if len(args) == 2:
                if isinstance(args[1], (PatExps, Sequence)):
                    new_args = (args[0].name, la, pe.cat(args[1], hints['merge_ends']))
            else:
                new_args = (args[0].name, la, pe.cat(args[1:], hints['merge_ends']))
            if not new_args[2].args:
                return S.One
            if new_args[2].args == (1,):
                return BosonOp(args[0].name, new_args[1])
            if not bool(new_args[1]) and new_args[2].args == (1, 1):
                return BosonNum(args[0].name)
            return Operator.__new__(cls, *new_args)
        if len(args) == 1 or len(args) == 2:
            return BosonOp(*args)
        if len(args) == 3:
            exponents = PatExps(args[2])
        else:
            exponents = PatExps(args[2:])
        if eval_exp:
            flip, new_exponents = exponents.in_new()
        else:
            flip, new_exponents = exponents.drop_zeros()
        if not new_exponents.args:
            return S.One
        left = bool(Integer(args[1])) ^ flip
        if new_exponents.args == (1,):
            return BosonOp(args[0], left)
        if not left and new_exponents.args == (1, 1):
            return BosonNum(args[0])
        return Operator.__new__(cls, args[0], left, new_exponents)

    @property
    def name(self):
        return self.args[0]
    @cached_property
    def left_base(self):
        return BosonOp(*self.args[:2])
    @cached_property
    def right_base(self):
        is_annihilation = (not self.args[1]) ^ (len(self.exps) & 1)
        return BosonOp(self.name, is_annihilation)
    @property
    def exps(self):
        if isinstance(self, BosonOp):
            return (S.One,)
        return self.args[2].args
    @cached_property
    def parity_diff(self):
        """
        number of creation operators minus annihilation operators
        """
        if isinstance(self, BosonOp):
            return S.NegativeOne if self.is_annihilation else S.One
        return self.args[2].parity_diff(self.args[1])
    
    def _eval_is_hermitian(self):
        if self.parity_diff == 0:
            return True
        from sympy.core.relational import Ne
        if Ne(self.parity_diff, 0) == True:
            return False

    def _eval_adjoint(self):
        new_left_base = self.right_base._eval_adjoint()
        new_exponents = [Dagger(exp) for exp in self.exps]
        return BosonPat(*new_left_base.args, new_exponents, eval_exp=False)
    
    def _eval_power(self, exp):
        if len(self.exps) == 1:
            return BosonPat(self.name, self.args[1], self.exps[0]*exp)
    
    def _eval_commutator_FermionPat(self, other, **hints):
        return S.Zero
    
    def _eval_anticommutator_FermionPat(self, other, **hints):
        return 2 * self * other
    
    def _eval_commutator_BosonPat(self, other, **hints):
        if self.name == other.name:
            if self.exps == (1, 1) and other.exps == (1,):
                return -other if other.is_annihilation else other
            if self.exps == (1,) and other.exps == (1, 1):
                return self if self.is_annihilation else self
            if self.is_hermitian and other.is_hermitian:
                return S.Zero
            return (BosonPat(self, other.exps, eval_exp=False,
                             merge_ends=(self.right_base==other.left_base))
                    - BosonPat(other, self.exps, eval_exp=False,
                               merge_ends=(other.right_base==self.left_base)))
        elif 'independent' in hints and hints['independent']:
            return S.Zero
        
    def _eval_anticommutator_BosonPat(self, other, **hints):
        if self.name == other.name:
            if self.is_hermitian and other.is_hermitian:
                return 2 * BosonPat(self, other.exps, eval_exp=False,
                                    merge_ends=(self.right_base==other.left_base))
            return (BosonPat(self, other.exps, eval_exp=False,
                             merge_ends=(self.right_base==other.left_base))
                    + BosonPat(other, self.exps, eval_exp=False,
                               merge_ends=(other.right_base==self.left_base)))
        elif 'independent' in hints and hints['independent']:
            return 2 * self * other

    def _eval_rewrite_as_BosonOp(self, base, *args, **kwargs):
        new_args = []
        ia = self.left_base.args[1]
        for n in self.exps:
            if n is S.Zero:
                continue
            if n is S.One:
                new_args.append(BosonOp(base, ia))
            else:
                new_args.append(Pow(BosonOp(base, ia),
                                    n, evaluate=False))
            ia = not ia
        return Mul(*new_args, evaluate=False)

    def num_repr(self, *args, left_p=True):
        """
        Represents the pattern in number operators

        :param left_p: Whether the power of canonical operator is
        left-multiplied to the number-operators
        """
        if isinstance(self, BosonNum):
            return (S.One, self) if left_p else (self, S.One)
        if len(self.exps) == 1:
            return (self, S.One) if left_p else (S.One, self)
        if args:
            base, ia, exps = args
        else:
            base, ia, exps = self.args
        from sympy.functions.combinatorial.factorials import ff
        n = Dummy()
        if left_p:
            exps = reversed(exps.args)
            bia = ia_temp = bool((not ia) ^ (len(self.exps) & 1))
        else:
            exps = exps.args
            bia = ia_temp = bool(ia)

        exp_temp = S.Zero
        factors = []
        for exp in exps:
            if exp_temp is S.Zero:
                ia_temp = bia
                exp_temp = exp
            elif bia == ia_temp:
                exp_temp += exp
            else:
                d = exp_temp - exp
                if (d > 0) == True:
                    a1 = n + exp + d if ia_temp ^ left_p else n - d
                    factors.append(ff(a1, exp))
                    exp_temp = d
                else:
                    a1 = n + exp_temp if ia_temp ^ left_p else n
                    factors.append(ff(a1, exp_temp))
                    exp_temp = - d
                    ia_temp = not ia_temp
            bia = not bia
        if left_p:
            # factors.reverse()
            num_prod = Mul(*factors).xreplace({n: BosonNum(base)})
            return BosonPat(base, ia_temp, exp_temp, eval_exp=False), num_prod
        else:
            num_prod = Mul(*factors).xreplace({n: BosonNum(base)})
            return num_prod, BosonPat(base, ia_temp, exp_temp, eval_exp=False)

    def _eval_rewrite_as_BosonNum(self, *args, **kwargs):
        left_p = kwargs.get('left_p', True) 
        f1, f2 = self.num_repr(*args, left_p=left_p)
        return f1 * f2

    def to_normal_order(self, via_dict=False):
        """
        Normal-ordered form of the pattern

        Can lead to a wrong result if any non-integer exponent is included

        :param via_dict: Applies a faster method that only works for
        known nonnegative integer exponents
        """
        bia = bool(self.left_base.is_annihilation)  # base is annihilation
        ea = ec = S.Zero  # exponent of annihilation/creation operators
        if via_dict:
            from sympy.core.add import Add
            p = {S.Zero: S.One}
            for exp in self.exps:
                if bia:
                    ea += exp
                elif ea is S.Zero:
                    ec += exp
                else:
                    new = dict()
                    for n, coeff in p.items():
                        ean = ea-n
                        for i in range(min(ean, exp)+1):
                            ni = n + i
                            new[ni] = new.get(ni, 0) + coeff*ffc(ean, i)*ffc(exp, i)/fac(i)
                    ec += exp
                    p = new
                bia = not bia
            return Add(*(c*BosonPat(self.name, 0, ec-k, ea-k) for k, c in p.items()))
        
        from sympy.concrete.summations import Sum
        from sympy.functions.combinatorial.factorials import binomial, ff
        limits, coeffs = [], []
        for exp in self.exps:
            if bia:
                ea += exp
            elif ea is S.Zero:
                ec += exp
            else:
                i = Dummy(nonnegative=True, integer=True)
                # In order to allow doit, some zero term need to be counted
                limits.append((i, 0, exp))
                coeffs.append(ff(ea, i))
                coeffs.append(binomial(exp, i))
                ec += exp - i
                ea -= i
            bia = not bia
        
        term = BosonPat(self.name, 0, ec, ea, eval_exp = False)
        return Sum(Mul(*coeffs, term), *limits)

    def _print_contents_latex(self, printer, *args):
        return printer._print(self.rewrite(BosonOp))
    def _print_contents(self, printer, *args):
        if self.left_base.is_annihilation:
            return r'BP(%s; %s)' % (str(self.name), ', '.join(str(e) for e in self.exps))
        else:
            return r'BP(D(%s); %s)' % (str(self.name), ', '.join(str(e) for e in self.exps))

boson.BosonOp.__bases__ = (BosonPat,)

class BosonNum(HermitianOperator, BosonPat):
    """
    Newly defined number operator class that supports diff function
    """

    _diff_wrt = True
    exps = (S.One, S.One)
    parity_diff = S.Zero
    boson_num_only = True

    @classmethod
    def default_args(self):
        return ("a",)
    
    def __new__(cls, mode, **hints):
        if hints.get('from_bp', False) and isinstance(mode, BosonPat):
            mode = mode.name
        return HermitianOperator.__new__(cls, mode)

    @property
    def name(self):
        return self.args[0]
    @cached_property
    def factors(self):
        a = BosonOp(self.args[0])
        return Dagger(a), a
    @property
    def left_base(self):
        return self.factors[0]
    @property
    def right_base(self):
        return self.factors[1]
    
    def __repr__(self):
        return f"N_({self.name})"

    def _eval_derivative(self, s: Expr, **hints):
        if s.is_commutative:
            return S.Zero
        if s == self:
            return S.One
        if 'independent' in hints and hints['independent'] and isinstance(s, BosonNum):
            return S.Zero
        return None
        
    def _eval_commutator_BosonNum(self, other, **hints):
        if 'independent' in hints and hints['independent']:
            return S.Zero
        return None
    
    def _eval_anticommutator_BosonNum(self, other, **hints):
        if self == other:
            return 2 * self ** 2
    
    def _sympystr(self, printer, *args):
        return printer._print(self.__repr__())
    def _print_contents_latex(self, printer, *args):
        s = str(self.name)
        if s == 'a':
            return r'\hat{N}' 
        if s == 'a_':
            return r'\hat{N\_{}}'
        if s and s[0] == 'a':
            if len(s) > 2 and s[1] == '_':
                return r'\hat{N}_{%s}' % ' '.join(s[2:].split('_'))
            return r'\hat{N}_{%s}' % ' '.join(s[1:].split('_'))
        return r'\hat{N}_{%s}' % s
    def _print_contents_pretty(self, printer, *args):
        return printer._print(self.__repr__())

def _base_d(op):
    if isinstance(op, BosonPat):
        base, d = op.name, op.parity_diff
    elif isinstance(op, Sequence):
        base, d = tuple(op)
        if not d.is_commutative or d.is_real == False:
            raise ValueError('Cannot shift with a non-commutative'
                ' or imaginary quantity')
    return base, d
class NumShift(Expr):
    """
    The doit() method of NumShift shifts the boson number operators in expr_num
    according to the elements of "shift" parameter,
    each of which should be given by a dict with {op: d} or Tuple(op, d) or Pow(op, d)

    Each BosonNum in the corresponding base will be shifted by direction*d
    for creation boson operator or -direction*d for annihilation;

    This is equivalent to reordering f(N) * op^d into
    op^d * g(N) (direction = 1) assuming independent modes and that d is commutative.
    """

    def __new__(cls, expr_num: Expr, *shift, **hints):
        if expr_num.is_commutative:
            return expr_num
        if isinstance(expr_num, cls):
            expr_num, direction, _shift = expr_num.args
        else:
            if shift and (shift[0] == 1 or shift[0] == -1):
                direction = shift[0]
                shift = shift[1:]
            else:
                direction = 1
            _shift = dict()

        scan_dict = hints.get('scan_dict', True)
        if (scan_dict or _shift or len(shift) > 1
            or not isinstance(shift[0], (Dict, dict, Tuple, tuple))):
            for s in shift:
                if isinstance(s, (dict, Dict)):
                    for a in s:
                        if not isinstance(a, BosonOp):
                            raise TypeError('Expect BosonOp, got %s' % a)
                        d = sympify(s[a])
                        if not d.is_commutative or d.is_real == False:
                            raise ValueError('Cannot shift with'
                                ' a non-commutative or imaginary quantity')
                        if a.is_annihilation:
                            _shift[a.name] = _shift.get(a, 0) + d
                        else:
                            _shift[a.name] = _shift.get(Dagger(a), 0) - d
                else:
                    try:
                        for op in s:
                            op, d = _base_d(op)
                            _shift[op] = _shift.get(op, 0) + d
                    except TypeError:
                        op, d = _base_d(s)
                        _shift[op] = _shift.get(op, 0) + d
            shift = Dict(_shift)
        else:
            shift = Dict(shift)

        obj = Expr.__new__(cls)
        obj._args = (expr_num, direction, shift)
        return obj
    
    @property
    def function(self):
        return self.args[0]
    @property
    def direction(self):
        return self.args[1]
    @property
    def shift(self):
        return self.args[2]

    def doit(self, **hints):
        sub = dict()
        for b in self.shift:
            n = BosonNum(b)
            if self.shift[b] != 0:
                sub[n] = n + self.direction * self.shift[b]
        f = self.function.doit(**hints) if hints.get('deep', True) else self.function
        return f.subs(sub)
    
    def shift_dict(self):
        return dict(self.shift)
    
    def _eval_commutator(self, other, **hints):
        return NumShift(self.function._eval_commutator(other, **hints),
                        self.direction, self.shift, scan_dict=False)
    
    def _eval_anticommutator(self, other, **hints):
        return NumShift(self.function._eval_anticommutator(other, **hints),
                        self.direction, self.shift, scan_dict=False)
    
    def _eval_adjoint(self):
        return NumShift(self.function._eval_adjoint(),
                        self.direction, self.shift, scan_dict=False)

def apply_num_shift(expr, depth=None, _current=0):
    if expr.is_Atom or isinstance(expr, (Operator, Dict)):
        return expr
    if depth is None or _current < depth:
        new_args = tuple(apply_num_shift(a, depth, _current+1)
                         for a in expr.args)
    else:
        new_args = expr.args
    if isinstance(expr, NumShift):
        if new_args == expr.args:
            new = expr.doit(deep=False)
        else:
            new = NumShift(*new_args, scan_dict=False).doit(deep=False)
    elif new_args == expr.args:
        new = expr
    else:
        new = expr.func(*new_args)
    return new
