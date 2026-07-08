from sympy.core.singleton import S
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import Function, Subs, Derivative
from sympy.core.containers import Tuple
from sympy.concrete.expr_with_limits import AddWithLimits
from op_patterns import BosonPat, FermionPat, check_num, redo_mul
from .sort_context import *
from .sort_context import clear_cache
import warnings

__all__ = ['is_pattern_form', 'pattern_form']

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
                yield PFTableProcessor()
            case a if a.is_Atom:
                if not self.expr.is_commutative:
                    yield PFTableProcessor(npf=a)
                yield PFTableProcessor({((), ()): a})
            case BosonPat():
                if v(self.expr):
                    yield PFLeaf((self.expr,), ())
                yield PFTableProcessor(npf=self.expr)
            case FermionPat():
                if v(self.expr):
                    yield PFTableProcessor({((), (self.expr,)): S.One})
                yield PFTableProcessor(npf=self.expr)
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
                yield PFTableProcessor({((), ()): a})
            case _:
                warnings.warn('Received unexpected expression')
                yield PFTableProcessor(npf=self.ls_apply())

    def result_ind(self) -> PFTableProcessor:
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
                    return PFTableProcessor({((), ()): expr})
                return PFTableProcessor(npf=expr)
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
                    return PFTableProcessor({((), ()): expr})
                return PFTableProcessor(npf=expr)
            case Derivative():
                pf = next(self.recur()).result_ind()
                e0 = pf.expr()
                if e0 == self.expr.expr:
                    expr = self.ls_apply()
                else:
                    expr = self.ls_apply(*self.expr.variable_count)
                if e0.npf is S.Zero:
                    return PFTableProcessor({((), ()): expr})
                return PFTableProcessor(npf=expr)
            case Add():
                pf = PFTableProcessor.add(*(r.result_ind() for r in self.recur()))
            case Pow():
                rs = self.recur()
                base = next(rs).result_ind()
                exp = next(rs).result_ind()
                if base.npf is not S.Zero:
                    return PFTableProcessor(npf=self.ls_apply(Pow(base.expr(), expr.expr())))
                if exp.npf is not S.Zero or not set(exp.table.keys()).issubset({((), ())}):
                    return PFTableProcessor(npf=self.ls_apply(Pow(base.expr(), exp.expr())))
                if set(base.table.keys()).issubset({((), ())}):
                    return PFTableProcessor({((), ()):self.ls_apply(Pow(base.expr(), exp.expr()))})
                e = exp.expr()
                if not e.is_Integer or e < 0:
                    return PFTableProcessor(npf=self.ls_apply(Pow(base.expr(), e)))
                pf = base.power_ind(e)
            case Mul():
                pf = PFTableProcessor({((), ()): S.One})
                for r in self.recur():
                    new = r.result_ind()
                    pf = pf.mul_ind(new)
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

    def result_res(self) -> tuple[PFTableProcessor, Expr]:
        """
        Pattern-formed result with residue due to non-commutativity among modes,
        operators reordered canonically

        Non-num_only args of Pow/Function will not be treated further;
        if the number operators in different modes appear
        in the args of a single Pow/Function,
        they will be temporarily considered as independent.
        """
        if isinstance(self.expr, (Function, Derivative, Subs)):
            if all(check_num(c.expr, self._cn_cache) for c in self.recur()):
                return PFTableProcessor({((), ()): self.ls_apply()}), S.Zero
            return PFTableProcessor(npf=self.ls_apply()), S.Zero
        if isinstance(self.expr, Add):
            pfs, rds = [], []
            for r in self.recur():
                pf, rd = r.result_res()
                pfs.append(pf)
                rds.append(rd)
            pf, res = PFTableProcessor.add(*pfs), Add(*rds)
        elif isinstance(self.expr, Pow):
            exp = self.expr.exp
            if not check_num(exp, self._cn_cache):
                return PFTableProcessor(npf=self.ls_apply()), S.Zero
            base_pf, base_rd = next(self.recur()).result_res()
            if base_pf.npf is not S.Zero:
                return PFTableProcessor(npf=self.ls_apply()), S.Zero
            if not (exp.is_Integer and exp > 1):
                if set(base_pf.table.keys()).issubset({((), ())}):
                    warnings.warn("Some residual table may be left uncalculated")
                    return PFTableProcessor({((), ()): Pow(base_pf[((), ())], exp)
                                                        * self.ls_apply(S.One).doit()}), S.Zero
                return PFTableProcessor(npf=self.ls_apply()), S.Zero
            pf, res = base_pf.power_res(exp, base_rd)
        elif isinstance(self.expr, Mul):
            pf, res = PFTableProcessor({((), ()): S.One}), S.Zero
            for r in self.recur():
                new, rd = r.result_res()
                if rd is not S.Zero:
                    new_res = Mul(pf.expr(), rd)
                pf, mr = pf.mul_res(new)
                res = Add(Mul(res, Add(new.expr(), mr)), mr, new_res)
        else:
            pf, res = next(self.recur()).result_res()
        if hasattr(self, 'limits'):
            if pf.table:
                c = self.ls_apply(S.One).doit()
                if c is not S.One:
                    for p, n in pf.table.items():
                        pf.table[p] = c * n
            if pf.npf is not S.Zero:
                pf.npf = self.ls_apply(pf.npf)
            if res is not S.Zero:
                res = self.ls_apply(res)
        return pf, res

class PFLeaf:
    __slots__ = ('bsc', 'fsc')
    def __init__(self, bs:tuple[BosonPat], fs:tuple[FermionPat]):
        self.bsc = BosonSortContext(bs)
        self.fsc = FermionSortContext(fs)

    def result_ind(self):
        bp, bt = self.bsc.result_ind()
        fp, ft = self.fsc.result_ind()
        if fp is None:
            return PFTableProcessor(dict())
        return PFTableProcessor({(bp, fp): bt * ft})
        
    def result_res(self):
        bp, bt, br = self.bsc.result_res()
        fp, ft, fr = self.fsc.result_res()
        if fp is None:
            return PFTableProcessor(dict()), (Mul(*bp, bt) + br) * fr
        table = {(bp, fp): bt * ft}
        residue = Add(Mul(*bp, bt, fr), Mul(*fp, ft, br), Mul(br, fr))
        return PFTableProcessor(table, cache=getattr(self.bsc, '_res_cache', None)), residue

def pattern_form(expr, as_dict=False, res=False, ignore=(), _s=True):
    """
    Returns the pattern-form result of the expr.
    
    :param as_dict: Whether to present the result as a dict (True),
    PFTableProcessor (None) or Expr (False)
    :param res: Whether to consider the non-independency among the operators
    and include the commutator residue
    :param ignore: Uncommtable types to be processed as if they are BosonNums
    :param _s: Whether to symplify the expression before reordering
    """
    pftn = PFTreeNode(expr, ignore=ignore, _s=_s)
    if res:
        result, residue = pftn.result_res()
        clear_cache()
        if as_dict is None:
            return result, residue
        if as_dict:
            return result.table, result.npf, residue
        return result.expr(), residue
    if as_dict is None:
        return pftn.result_ind()
    if as_dict:
        result = pftn.result_ind()
        return result.table, result.npf
    return pftn.result_ind().expr()
