from __future__ import annotations
import ast
import keyword
import re
from typing import Any, Dict, List, Optional, Tuple

from op_patterns import BosonNum, BosonPat, FermionPat
from parametric import (
    t_, _g, nlc, pm,
    a_p, a_s, a_i, MultimodeNum, MultimodeNum2, MultimodeCOP
)
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
)
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.fermion import FermionOp

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Parametric gain operator symbol \hat{g} (non-commutative to preserve operator ordering)
g_hat = Symbol(r'\hat{g}', commutative=False)

RESERVED_NAMES = {
    # Constants
    'i', 'I', 'pi', 'PI', 'e', 'E',
    # Elementary functions
    'exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'sinh', 'cosh', 'tanh', 'coth',
    'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
    'sqrt', 'log', 'ln',
    # Quantum macros / operators
    'dag', 'Dagger', 'dagger', 'BosonNum', 'N_', 'BP', 'FP',
    'BosonPat', 'FermionPat', 'BosonOp', 'FermionOp',
    'MMN', 'MMN2', 'MMC', 'Pat', 'pattern',
}


def is_valid_variable_name(name: str) -> Tuple[bool, str]:
    """Validate whether name conforms to Python identifier rules and does not conflict with keywords or built-in functions/constants."""
    if not name or not isinstance(name, str):
        return False, "Identifier cannot be empty."
    name = name.strip()
    if not name.isidentifier():
        return False, f"'{name}' is not a valid Python identifier."
    if keyword.iskeyword(name):
        return False, f"'{name}' is a reserved Python keyword."
    if name in RESERVED_NAMES:
        return False, f"'{name}' conflicts with a built-in constant or mathematical function ({name})."
    return True, ""


def is_valid_python_name(name: str) -> bool:
    """Validate whether name conforms to Python identifier rules and does not conflict with keywords or built-ins."""
    return is_valid_variable_name(name)[0]


def expr_to_code_str(e: Expr, dc: bool = True) -> str:
    """
    Convert a SymPy quantum operator expression (from PF.expr() or TWMSolver.to_expr())
    into a valid Python expression string that can be parsed back by parse_expr.
    """
    if e == 0 or e == S.Zero:
        return "0"
    if e == 1 or e == S.One:
        return "1"

    s_Nt = Symbol('N_t', commutative=False)
    s_A = Symbol('A', commutative=False)
    s_dagA = Symbol('dag(A)', commutative=False)
    s_t = Symbol('t', real=True)
    s_g = Symbol('g', commutative=False)
    s_nlc = Symbol('nlc', real=True, nonnegative=True)
    s_pm = Symbol('pm', real=True)

    mmc_ann = MultimodeCOP(a_s.name, a_i.name) if dc else MultimodeCOP(a_p.name, a_s.name, 1)
    mmc_cre = Dagger(mmc_ann)

    def _rec(x):
        if isinstance(x, MultimodeCOP):
            if x == mmc_ann:
                return s_A
            if x == mmc_cre:
                return s_dagA
        if isinstance(x, MultimodeNum2):
            if x.displacement == 0:
                return s_Nt
            return Symbol(f'MMN2({x.displacement})', commutative=False)
        if isinstance(x, MultimodeNum):
            if x.displacement == 0:
                return Symbol(f'MMN({x.name})', commutative=False)
            return Symbol(f'MMN({x.name}, {x.displacement})', commutative=False)
        if isinstance(x, BosonNum):
            return Symbol(f'N_({x.name})', commutative=False)
        if isinstance(x, (BosonOp, FermionOp)):
            return Symbol(str(x.name), commutative=False) if x.is_annihilation else Symbol(f'dag({x.name})', commutative=False)
        if isinstance(x, (BosonPat, FermionPat)):
            factors = []
            is_ann = x.left_base.is_annihilation
            for exp_val in x.exps:
                op_sym = Symbol(str(x.name), commutative=False) if is_ann else Symbol(f'dag({x.name})', commutative=False)
                factors.append(op_sym ** exp_val if exp_val != 1 else op_sym)
                is_ann = not is_ann
            return Mul(*factors)
        if isinstance(x, Dagger):
            inner = _rec(x.args[0])
            return Symbol(f'dag({inner})', commutative=False)
        if x == t_:
            return s_t
        if x == g_hat or x == _g:
            return s_g
        if x == nlc:
            return s_nlc
        if x == pm:
            return s_pm
        if not getattr(x, 'args', None):
            return x
        return x.func(*(_rec(a) for a in x.args))

    return str(_rec(sympify(e)))

