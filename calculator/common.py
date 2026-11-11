from __future__ import annotations
import keyword
from typing import Tuple

from sympy.core.symbol import Symbol
import re
from tokenize import NUMBER, NAME, OP
from sympy.parsing.sympy_parser import (
    convert_xor,
    function_exponentiation,
    implicit_application,
    implicit_multiplication,
    split_symbols_custom,
    standard_transformations,
    _token_splittable,
)


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


def implicit_matmul(tokens, local_dict, global_dict):
    """Implicitly inserts '@' (multimode multiplication) between adjacent factors while preserving explicit '*'."""
    masked = [(OP, '\x00') if tok == (OP, '*') else tok for tok in tokens]
    return [
        (OP, '@') if tok == (OP, '*') else (OP, '*') if tok == (OP, '\x00') else tok
        for tok in implicit_multiplication(masked, local_dict, global_dict)
    ]


WM_TRANSFORMS = (
    split_scientific_variable,
) + standard_transformations + (
    split_symbols_custom(custom_can_split),
    implicit_matmul,
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
