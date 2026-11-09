from sympy.core.symbol import Symbol
import sympy.concrete.expr_with_limits as ewl
import sympy.core.function as function
import sympy.core.power as power
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.power import Pow
from sympy.core.function import Function
from .boson_pattern import *
from .fermion_pattern import *
from .tools import *

__all__ = ['BosonNum', 'BosonPat', 'NumShift', 'apply_num_shift',
           'FermionPat', 'UidCommutator',
           'extract_modes', 'check_num', 'in_boson_num',
           'mul_pattern', 'redo_mul']

def _safe_eval_adjoint(self):
    own = self.__class__.__dict__.get("_eval_adjoint", None)
    if own is not None and own is not _safe_eval_adjoint:
        return own(self)
    return self.func(*[arg._eval_adjoint() for arg in self.args])

function.Function._eval_adjoint = _safe_eval_adjoint
power.Pow._eval_adjoint = lambda self: Pow(*[arg._eval_adjoint() for arg in self.args])

def _adjoint_ewl_recursive(expr, exclude_vars=set()):
    
    if isinstance(expr, AddWithLimits):

        current_vars = set(v for v, *_ in expr.limits)
        all_exclude = exclude_vars | current_vars
        

        new_func = _adjoint_ewl_recursive(expr.function, all_exclude)

        new_limits = []
        for var, *lims in expr.limits:
            new_lims = []
            for lim in lims:
                new_lim = _adjoint_ewl_recursive(lim, all_exclude)
                new_lims.append(new_lim)
            new_limits.append((var, *new_lims))
        
        return AddWithLimits(new_func, *new_limits)
    
    if isinstance(expr, (Pow, Function)):
        return expr.func(*[_adjoint_ewl_recursive(arg, exclude_vars) for arg in expr.args])
    
    if expr.is_Add:
        return expr.func(*[_adjoint_ewl_recursive(arg, exclude_vars) for arg in expr.args])

    if expr.func.__name__ in ('Mul', 'Commutator'):
        return expr.func(*[_adjoint_ewl_recursive(arg, exclude_vars) for arg in reversed(expr.args)])
    
    if expr.atoms(Symbol) & exclude_vars:
        return expr
    
    return expr._eval_adjoint()

def _ewl_eval_adjoint(self: AddWithLimits):
    """
    _eval_adjoint method for AddWithLimits class
    """
    return _adjoint_ewl_recursive(self)

ewl.AddWithLimits._eval_adjoint = _ewl_eval_adjoint
