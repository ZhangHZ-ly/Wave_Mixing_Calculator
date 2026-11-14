from __future__ import annotations
import re
from typing import Optional, Set, Tuple

from cas_qwm.op_pattern import BosonNum, BosonPat, FermionPat
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.combinatorial.factorials import FallingFactorial
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex

def _fermion_target_to_expr_str(target, name: str) -> str:
    """Format single-mode fermion result into a valid Python expression string for Tab 2."""
    if target == 0 or target == S.Zero:
        return "0"
    if target == 1 or target == S.One:
        return "1"
    if isinstance(target, FermionOp):
        return name if target.is_annihilation else f"dag({name})"
    if isinstance(target, FermionPat) and target.exps == (1, 1):
        if target.left_base.is_annihilation:
            return f"{name} * dag({name})"
        else:
            return f"dag({name}) * {name}"
    try:
        norm_test = 1 - FermionPat(name, False, 1, 1)
        if target == norm_test:
            return f"1 - dag({name}) * {name}"
    except Exception:
        pass
    return str(target)


def _has_unknowns(target, exp_symbols: Set[Symbol]) -> bool:
    """Check whether the simplified quantum expression contains any undetermined symbols (unknowns)."""
    if not exp_symbols:
        return False
    target_syms = target.free_symbols
    return any(
        s in target_syms or any(ts.name == s.name and (ts.is_integer or ts.is_nonnegative) for ts in target_syms)
        for s in exp_symbols
    )


_FF_NOTE = "Subscripts in (x)_k indicate the falling factorial (Pochhammer symbol): (x)_k = x(x-1)...(x-k+1), rather than operator powers."


def _check_normal_orderable(simplified, exp_symbols: Set[Symbol]) -> None:
    if _has_unknowns(simplified, exp_symbols):
        raise ValueError("Cannot perform normal ordering: undetermined symbols exist in the simplified expression.")


def eval_single_mode(mode_type: str, name: str, first_op: str, exp_str: str, action: str) -> Tuple[str, Optional[bool], str, Optional[str]]:
    """
    Evaluates single-mode quantum operator product (BosonPat or FermionPat).
    Letters in exp_str are treated as undetermined nonnegative integers (Symbol(..., integer=True, nonnegative=True)).
    Other non-alphanumeric characters are treated as separators.
    Normal ordering is not supported if the simplified expression contains unknowns.
    Returns: (latex_str, is_hermitian, expr_str, note)
    """
    is_boson = (mode_type == 'boson')
    name = (name or '').strip() or ('a' if is_boson else 'c')
    is_ann = (first_op == 'annihilation')

    # Extract tokens: a letter followed by letters, digits, or underscores is part of the symbol name.
    # Digits alone are integers. Any other character (including underscore not following a letter) is a separator.
    exps = []
    exp_symbols: Set[Symbol] = set()
    for tok in re.findall(r'[a-zA-Z][a-zA-Z0-9_]*|\d+', exp_str or ''):
        if tok.isdigit():
            exps.append(int(tok))
        else:
            sym = Symbol(tok, integer=True, nonnegative=True)
            exps.append(sym)
            exp_symbols.add(sym)

    if not exps or all(e == 0 for e in exps):
        orig = simp = norm = S.One
        is_hermitian = True
    elif is_boson:
        orig = BosonPat(name, is_ann, *exps, eval_exp=False)
        if isinstance(orig, (BosonOp, BosonNum)):
            simp = orig
            is_hermitian = isinstance(orig, BosonNum)
        else:
            simp = orig.rewrite(BosonNum)
            pd = orig.parity_diff
            is_hermitian = True if pd == 0 else (False if not pd.free_symbols else None)

        norm = None
        if action == 'normal_order':
            _check_normal_orderable(simp, exp_symbols)
            norm = orig.to_normal_order(via_dict=True).rewrite(BosonOp)
    else:
        orig = FermionPat(name, is_ann, *exps, eval_exp=False)
        simp = orig.doit()
        is_pair = isinstance(simp, FermionPat) and simp.exps == (1, 1)
        if simp == 0 or simp == 1 or is_pair:
            is_hermitian = True
        elif isinstance(simp, FermionOp):
            is_hermitian = False
        else:
            is_herm_val = simp.is_hermitian
            is_hermitian = True if is_herm_val is True else (False if is_herm_val is False else None)

        norm = None
        if action == 'normal_order':
            _check_normal_orderable(simp, exp_symbols)
            # c c† -> 1 - c† c ; every other simplified form is already normal ordered
            norm = 1 - FermionPat(name, False, 1, 1) if is_pair and simp.left_base.is_annihilation else simp

    target = orig if action == 'original' else (simp if action == 'patform' else norm)
    target_str = str(target) if is_boson else _fermion_target_to_expr_str(target, name)
    note = _FF_NOTE if target.has(FallingFactorial) else None
    return latex(target), is_hermitian, target_str, note
