from __future__ import annotations
import keyword
from typing import Tuple

from op_patterns import BosonNum, BosonPat, FermionPat
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
import re
from tokenize import NUMBER, NAME
from sympy.core.sympify import sympify
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    split_symbols_custom,
    standard_transformations,
    _token_splittable,
)
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.fermion import FermionOp


def custom_can_split(symbol: str) -> bool:
    """
    Predicate for whether a token name can be split into multiple single-letter symbols.
    Tokens containing digits (e.g. 'e1', 'a1', 'b2', 'x0') or underscores are treated
    as single alphanumeric variable/operator names with index/subscript, and must NEVER be split.
    """
    if any(c.isdigit() for c in symbol):
        return False
    return _token_splittable(symbol)


def split_scientific_variable(tokens, local_dict, global_dict):
    """
    Pre-pass to separate number followed by a variable starting with e/E (e.g. '2e1' -> 2 * e1)
    when e1 is a registered variable in local_dict, preventing Python tokenizer from treating
    it as scientific notation float.
    """
    result = []
    for tok in tokens:
        if tok[0] == NUMBER:
            m = re.match(r'^(\d+)([eE]\d+)$', tok[1])
            if m and (m.group(2) in local_dict or m.group(2).lower() in local_dict):
                num_part, var_part = m.group(1), m.group(2)
                result.append((NUMBER, num_part))
                result.append((NAME, var_part))
                continue
        result.append(tok)
    return result


TRANSFORMS = (
    split_scientific_variable,
) + standard_transformations + (
    split_symbols_custom(custom_can_split),
    implicit_multiplication,
    implicit_application,
    function_exponentiation,
    convert_xor,
)

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


def expr_to_code_str(e: Expr) -> str:
    """
    Convert a SymPy quantum operator expression (from single-mode or pattern form)
    into a valid Python expression string that can be parsed back by parse_expr.
    """
    if e == 0:
        return "0"
    if e == 1:
        return "1"

    def _rec(x):
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
        if isinstance(x, Dagger) or getattr(x, 'func', None) and x.func.__name__ == 'conjugate':
            inner = _rec(x.args[0])
            return Symbol(f'dag({inner})', commutative=False)
        if not getattr(x, 'args', None):
            return x
        return x.func(*(_rec(a) for a in x.args))

    return str(_rec(sympify(e)))

