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
    _group_parentheses,
    _apply_functions,
    _flatten,
    AppliedFunction,
    _token_callable,
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


def _implicit_multiplication_single(tokens, local_dict, global_dict):
    """
    Implicitly adds '@' (single-mode multiplication) tokens between adjacent factors
    separated only by spaces, while preserving explicit '*' for multi-mode multiplication.
    """
    result = []
    skip = False
    for tok, nextTok in zip(tokens, tokens[1:]):
        result.append(tok)
        if skip:
            skip = False
            continue
        if tok[0] == OP and tok[1] == '.' and nextTok[0] == NAME:
            skip = True
            continue
        if isinstance(tok, AppliedFunction):
            if isinstance(nextTok, AppliedFunction):
                result.append((OP, '@'))
            elif nextTok == (OP, '('):
                if tok.function[1] == "Function":
                    tok.function = (tok.function[0], 'Symbol')
                result.append((OP, '@'))
            elif nextTok[0] == NAME or nextTok[0] == NUMBER:
                result.append((OP, '@'))
        else:
            if tok == (OP, ')'):
                if isinstance(nextTok, AppliedFunction):
                    result.append((OP, '@'))
                elif nextTok[0] == NAME or nextTok[0] == NUMBER:
                    result.append((OP, '@'))
                elif nextTok == (OP, '('):
                    result.append((OP, '@'))
            elif (tok[0] == NAME and not _token_callable(tok, local_dict, global_dict)) or tok[0] == NUMBER:
                if isinstance(nextTok, AppliedFunction) or (nextTok[0] == NAME and _token_callable(nextTok, local_dict, global_dict)):
                    result.append((OP, '@'))
                elif nextTok == (OP, '('):
                    result.append((OP, '@'))
                elif nextTok[0] == NAME or nextTok[0] == NUMBER:
                    result.append((OP, '@'))
    if tokens:
        result.append(tokens[-1])
    return result


def implicit_multiplication_single(tokens, local_dict, global_dict):
    """
    Custom implicit multiplication for Page 3 (Wave Mixing):
    Factors separated by whitespace without '*' execute single-mode multiplication (mapped to '@').
    """
    res1 = _group_parentheses(implicit_multiplication_single)(tokens, local_dict, global_dict)
    res2 = _apply_functions(res1, local_dict, global_dict)
    res3 = _implicit_multiplication_single(res2, local_dict, global_dict)
    return _flatten(res3)


WM_TRANSFORMS = (
    split_scientific_variable,
) + standard_transformations + (
    split_symbols_custom(custom_can_split),
    implicit_multiplication_single,
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
