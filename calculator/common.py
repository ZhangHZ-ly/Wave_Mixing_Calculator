from __future__ import annotations
import keyword
from typing import Any, Dict, List, Tuple

import re
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
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


def _build_transforms(multiplication) -> tuple:
    return (split_scientific_variable,) + standard_transformations + (
        split_symbols_custom(custom_can_split),
        multiplication,
        implicit_application,
        function_exponentiation,
        convert_xor,
    )


TRANSFORMS = _build_transforms(implicit_multiplication)


def implicit_matmul(tokens, local_dict, global_dict):
    """Implicitly inserts '@' (multimode multiplication) between adjacent factors while preserving explicit '*'."""
    masked = [(OP, '\x00') if tok == (OP, '*') else tok for tok in tokens]
    return [
        (OP, '@') if tok == (OP, '*') else (OP, '*') if tok == (OP, '\x00') else tok
        for tok in implicit_multiplication(masked, local_dict, global_dict)
    ]


WM_TRANSFORMS = _build_transforms(implicit_matmul)

# Elementary functions exposed to expression input (wrapped per-workspace before use)
ELEMENTARY_FUNCS = {
    'exp': exp, 'sin': sin, 'cos': cos,
    'sinh': sinh, 'cosh': cosh, 'tanh': tanh,
    'sqrt': sqrt,
}
DAGGER_ALIASES = ('dag', 'Dagger', 'dagger')

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


def validated_name(defn: Dict[str, Any], label: str) -> str:
    """Return the stripped 'name' of a definition dict, raising ValueError if it is not a valid identifier."""
    name = (defn.get('name') or '').strip()
    valid, err_msg = is_valid_variable_name(name)
    if not valid:
        raise ValueError(f"Invalid {label} identifier: {err_msg}")
    return name


def order_by_refs(named_expressions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Order the saved-expression definitions so that each follows the definitions it references
    (`refs`: name in its source input -> definition id). A definition only references definitions
    that existed when it was saved, so the references are acyclic.
    """
    by_id = {ne['id']: ne for ne in named_expressions}
    ordered: List[Dict[str, Any]] = []
    visited = set()

    def visit(def_id: str) -> None:
        if def_id in visited or def_id not in by_id:
            return
        visited.add(def_id)
        for ref_id in by_id[def_id].get('refs', {}).values():
            visit(ref_id)
        ordered.append(by_id[def_id])

    for def_id in by_id:
        visit(def_id)
    return ordered


def symbol_assumptions(defn: Dict[str, Any]) -> Dict[str, bool]:
    """Map the (mutually prioritized) assumption flags of a symbol definition to sympy Symbol kwargs."""
    for flag, key in (('is_positive', 'positive'), ('is_nonnegative', 'nonnegative'),
                      ('is_integer', 'integer'), ('is_real', 'real')):
        if defn.get(flag):
            return {key: True}
    return {}


def safe_subs(expr, subs: Dict[Any, Any]):
    """Apply display-name substitutions, falling back to the original expression on failure."""
    if not subs:
        return expr
    try:
        return expr.subs(subs)
    except Exception:
        return expr
