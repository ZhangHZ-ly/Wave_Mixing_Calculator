from sympy.core.singleton import S
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.numbers import Integer
from sympy.core.expr import Expr
from sympy.physics.quantum import Operator
import sympy.physics.quantum.fermion as fermion
from sympy.physics.quantum.fermion import FermionOp
from .pattern_exponents import PatExps, is_nat_n
from collections.abc import Sequence
from functools import cached_property

__all__ = ['FermionPat', 'UidCommutator']

fermion.FermionOp._eval_adjoint = lambda self: FermionOp(self.name, not self.is_annihilation)
fermion.FermionOp._eval_power = lambda self, exp: (FermionPat(*self.args, exp) if exp.is_commutative
                                                   else Operator._eval_power(self, exp))

def check_diff(x: dict, y: dict):
    # 0 for equal, 1 for y - x, -1 for x - y
    xk = x.keys, yk = y.keys
    lx, ly = len(xk), len(yk)
    lt = len(xk | yk)
    if lt > lx:
        if lt > ly:
            return None
        else:
            direction = 1
    elif lt > ly:
        direction = -1
    else:
        direction = 0
    for key in xk & yk:
            r = (y[key]/x[key])
            if r < 0:
                return None
            if r == 1:
                continue
            if r > 1:
                if direction == -1:
                    return None
                else:
                    direction = 1
            elif direction == 1:
                return None
            else:
                direction = -1
    return direction
def new_uid_exp(collection: set[dict], new: dict):
    for old in tuple(collection):
        match check_diff(old, new):
            case None:
                collection.add(new)
            case 0:
                collection.remove(old)
                del old, new
                return
            case -1:
                collection.remove(old)
                for k in old:
                    old[k] -= new.get(k, 0)
                new_uid_exp(collection, old)
                del new
                return
            case 1:
                collection.remove(old)
                for k in new:
                    new[k] -= old.get(k, 0)
                new_uid_exp(collection, new)
                del old
                return
def uid_exp(*exps, sym_count=False):
    if sym_count:
        int_part = S.Zero
        uid_part = set()
        temp = dict()
        for exponent in exps:
            for a in Add.make_args(exponent):
                if a.is_Integer:
                    int_part += a
                else:
                    coeff, rest = a.as_coeff_Mul()
                    temp[rest] = temp.get(rest, 0) + coeff
            new_uid_exp(uid_part, temp.copy())
            temp.clear()
        return Add(int_part, *(Mul(n, k) for ns in uid_part for n, k in ns.items()))
    else:
        return Add(*exps)

class FermionPat(Operator):
    """
    Collects multiplication of fermionic operators from the same mode.

    Equivalent to ... * c^x * c†^y * c^z * ..., where the base of
    the left_most factor is c^(†) = FermionOp(name, left_base) 
    """

    @classmethod
    def default_args(self):
        return ('c', True)
    
    def __new__(cls, *args, eval_exp=True, **hints):
        if isinstance(args[0], cls) and 'merge_ends' in hints:
            if len(args) == 1:
                return args[0]
            if isinstance(args[0], FermionOp):
                pe = PatExps(S.One)
            else:
                pe = args[0].args[2]
            if len(args) == 2:
                if isinstance(args[1], (Sequence, PatExps)):
                    new_args = (args[0].name, args[0].args[1],
                                pe.cat(args[1], hints['merge_ends'], True))
            else:
                new_args = (args[0].name, args[0].args[1],
                            pe.cat(args[1:], hints['merge_ends'], True))
            if not isinstance(new_args[2], PatExps):
                return S.Zero
            if new_args[2].args == ():
                return S.One
            if new_args[2].args == (1,):
                return FermionOp(args[0].args, left)
            pd = new_args[2].parity_diff()
            if pd.is_integer == False:
                raise ValueError("Fermionic operators can only be raised to a"
                " positive integer power")
            if (pd < -1) == True or (pd > 1) == True:
                return S.Zero
            return Operator.__new__(cls, *new_args)
        if len(args) == 1 or len(args) == 2:
            return FermionOp(*args)
        if len(args) == 3:
            if args[2] is None:
                return S.Zero
            else:
                exponents = PatExps(args[2])
        else:
            exponents = PatExps(args[2:])
        if eval_exp:
            flip, new_exponents = exponents.in_new(f=True)
        else:
            flip, new_exponents = exponents.drop_zeros()
        if not isinstance(new_exponents, PatExps):
            return S.Zero
        left = bool(Integer(args[1])) ^ flip
        if new_exponents.args == ():
            return S.One
        if new_exponents.args == (1,):
            return FermionOp(args[0], left)
        pd = new_exponents.parity_diff()
        if pd.is_integer == False:
            raise ValueError("Fermionic operators can only be raised to a"
            " positive integer power")
        if (pd < -1) == True or (pd > 1) == True:
            return S.Zero
        return Operator.__new__(cls, args[0], left, new_exponents)
    
    @property
    def name(self):
        return self.args[0]
    @cached_property
    def left_base(self):
        return FermionOp(*self.args[:2])
    @cached_property
    def right_base(self):
        is_annihilation = (not self.args[1]) ^ (len(self.exps) & 1)
        return FermionOp(self.name, is_annihilation)
    @property
    def exps(self):
        if isinstance(self, FermionOp):
            return (S.One,)
        return self.args[2].args
    @cached_property
    def comm_symm(self):
        return Pow(S.NegativeOne, uid_exp(*self.exps, sym_count=True))

    def doit(self, **hints):
        flip, new_exponents = self.args[2].f_simp(**hints)
        return FermionPat(self.name,
                          self.left_base.is_annihilation ^ flip,
                          new_exponents, eval_exp=False)
    
    def _eval_is_hermitian(self):
        if self.comm_symm == 1:
            return True

    def _eval_adjoint(self):
        # not taking adjoint for the exponents
        # since any legal exponent must be a nonnegative integer
        new_left_base = self.right_base._eval_adjoint()
        new_exponents = list[self.exps].reverse()
        return FermionPat(*new_left_base.args, new_exponents, eval_exp=False)
    
    def _eval_power(self, exp):
        if exp == 0:
            return S.One
        if exp == 1:
            return self
        if exp.is_integer:
            if (exp > 1) == True:
                if self.comm_symm == -1:
                    return 0
                if self.comm_symm == 1:
                    return self
                if all(n.is_integer and n.is_positive for n in self.exps):
                    return S.Zero if bool(len(self.exps) & 1) else self
            if (exp > 2) == True:
                return FermionPat(self, *self.exps,
                                  merge_ends=not(len(self.exps) & 1),
                                  eval_exp=False)
        if exp.is_integer == False or (exp < 0) ==True:
            if self.exps == (1, 1):
                raise ValueError("Fermionic operators can only be raised to a"
                    " positive integer power")

    def _eval_commutator_FermionPat(self, other, **hints):
        if self.name == other.name:
            if self.exps == (1,) and other.exps == (1, 1):
                if self == other.left_base:
                    return -self
                else:
                    return self
            if self.is_hermitian and other.is_hermitian:
                return S.Zero
        elif 'independent' in hints and hints['independent']:
            if self.is_hermitian or other.is_hermitian:
                return S.Zero
            if self.comm_symm == -1 == other.comm_symm:
                return 2 * self * other
    
    def _eval_anticommutator_FermionPat(self, other, **hints):
        if self.name == other.name:
            if self.comm_symm == -1 and other.exps == (1, 1):
                return self
            if self.is_hermitian and other.is_hermitian:
                return 2 * FermionPat(self, other.exponents,
                                  merge_ends=(self.right_base == other.left_base))
        elif 'independent' in hints and hints['independent']:
            if self.is_hermitian or other.is_hermitian:
                return 2 * self * other
            if self.comm_symm == -1 == other.comm_symm:
                return S.Zero

    def _eval_rewrite_as_FermionOp(self, base, bia, *args, **hints):
        new_args = []
        for n in self.exps:
            if n is S.Zero:
                continue
            if n is S.One:
                new_args.append(FermionOp(base, bia))
            else:
                new_args.append(Pow(FermionOp(base, bia),
                                    n, evaluate=False))
            bia = not bia
        return Mul(*new_args, evaluate=False)

    def _print_contents_latex(self, printer, *args):
        return printer._print(self.rewrite(FermionOp))

    def _print_contents(self, printer, *args):
        if self.left_base.is_annihilation:
            return r'FP(%s; %s)' % (str(self.name), ', '.join(str(e) for e in self.exps))
        else:
            return r'FP(D(%s); %s)' % (str(self.name), ', '.join(str(e) for e in self.exps))

fermion.FermionOp.__bases__ = (FermionPat,)

def fp_separable(fp_fac: FermionPat, fp: FermionPat, from_left=True, force=False):
    """
    Whether the given FermionPat fp can be represented 
    as another FermionPat multiplied by fp_fac

    :param from_left: True/False for left-/right-multiplication
    :param force: Whether to assume that each arg in the FermionPats is legal,
    might lead to a false-True when the signs of symbols are not specified
    """
    if fp_fac.name != fp.name or len(fp.exps) < len(fp_fac.exps):
        return False
    if from_left:
        if not fp.left_base == fp_fac.left_base:
            return False
        es1, es2 = fp.exps, fp_fac.exps
        if all(es1[i] == es2[i] for i in range(len(es2))):
            le1, le2 = es1[len(es2)-1], es2[-1]
        else:
            return False
    elif not fp.right_base == fp_fac.right_base:
        return False
    else:
        es1, es2 = fp.exps, fp_fac.exps
        if all(es1[-i] == es2[-i] for i in range(1, len(es2))):
            le1, le2 = es1[-len(es2)], es2[0]
    d = le2 - le1
    if d.is_integer == False:
        raise ValueError("Fermionic operators can only be raised to a"
            " positive integer power")
    if (d < 0) == True:
        return False
    if force:
        dict1 = dict(le1.as_coefficients_dict())
        dict2 = dict(le2.as_coefficients_dict())
        check = check_diff(dict1, dict2)
        return (check == -1 or check == 0)
    else:
        return is_nat_n(d)

def m1_base(s):
    if s is S.One:
        return S.Zero
    if s is S.NegativeOne:
        return S.One
    if s.is_Pow and s.base is S.NegativeOne:
        return s.exp
class UidCommutator(Expr):
    """
    Unidentified commutator with FermionPat
    
    Can be either Commutator or Anticommutator
    """
    is_commutative = False

    __slots__ = ('comm_sign',)

    def __new__(cls, A, B, **hints):
        if not (A and B):
            return S.Zero
        if A == B:
            return S.Zero
        if A.is_commutative or B.is_commutative:
            return S.Zero

        ca, nca = A.args_cnc()
        cb, ncb = B.args_cnc()

        syma = Pow(S.NegativeOne, uid_exp(*(m1_base(a.comm_symm) for a in nca),
                                          sym_count=hints.get('sym_count', False)))
        symb = Pow(S.NegativeOne, uid_exp(*(m1_base(b.comm_symm) for b in ncb),
                                          sym_count=hints.get('sym_count', False)))
        if syma == 1 or symb == 1:
            from sympy.physics.quantum.commutator import Commutator
            obj = Commutator(Mul._from_args(nca), Mul._from_args(ncb))
        elif syma == -1 and symb == -1:
            from sympy.physics.quantum.anticommutator import AntiCommutator
            obj = AntiCommutator(Mul._from_args(nca), Mul._from_args(ncb))
        else:
            obj = Expr.__new__(cls, Mul._from_args(nca), Mul._from_args(ncb))
            if syma == -1:
                obj.comm_sign = symb
            elif symb == -1 or syma == symb:
                obj.comm_sign = syma
            else:
                obj.comm_sign = Pow(S.NegativeOne, syma.exp*symb.exp)
        c_part = ca + cb
        return Mul(*c_part, obj)

    def doit(self, **hints):
        """ Evaluate commutator """
        A = self.args[0]
        B = self.args[1]
        return (A*B - self.comm_sign*B*A).doit(**hints)

    def _eval_adjoint(self):
        from sympy.physics.quantum.dagger import Dagger
        return UidCommutator(Dagger(self.args[1]), Dagger(self.args[0]))

    def _sympyrepr(self, printer, *args):
        return "UComm(%s,%s)" % (printer._print(self.args[0]),
                                 printer._print(self.args[1]))
    def _print_contents_latex(self, printer, *args):
        return r"\operatorname{UComm}(%s, %s)" % (printer._print(self.args[0]),
                                 printer._print(self.args[1]))
    