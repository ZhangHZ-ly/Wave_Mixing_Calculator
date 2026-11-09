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
    for s in exp_symbols:
        if s in target_syms:
            return True
        for ts in target_syms:
            if ts.name == s.name and (ts.is_integer or ts.is_nonnegative):
                return True
    return False


def eval_single_mode(mode_type: str, name: str, first_op: str, exp_str: str, action: str) -> Tuple[str, Optional[bool], str, Optional[str]]:
    """
    Evaluates single-mode quantum operator product (BosonPat or FermionPat).
    Letters in exp_str are treated as undetermined nonnegative integers (Symbol(..., integer=True, nonnegative=True)).
    Other non-alphanumeric characters are treated as separators.
    Normal ordering is not supported if the simplified expression contains unknowns.
    Returns: (latex_str, is_hermitian, expr_str, note)
    """
    name = (name or '').strip()
    if not name:
        name = 'a' if mode_type == 'boson' else 'c'
    is_ann = (first_op == 'annihilation')

    # Extract tokens: a letter followed by letters, digits, or underscores is part of the symbol name.
    # Digits alone are integers. Any other character (including underscore not following a letter) is a separator.
    exp_tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9_]*|\d+', exp_str or '')
    exps = []
    exp_symbols: Set[Symbol] = set()
    for tok in exp_tokens:
        if tok.isdigit():
            exps.append(int(tok))
        else:
            sym = Symbol(tok, integer=True, nonnegative=True)
            exps.append(sym)
            exp_symbols.add(sym)

    if mode_type == 'boson':
        if not exps or all(e == 0 for e in exps):
            orig = patform = norm = S.One
            is_hermitian = True
        else:
            orig = BosonPat(name, is_ann, *exps, eval_exp=False)
            if isinstance(orig, BosonOp):
                patform = orig
                is_hermitian = False
            elif isinstance(orig, BosonNum):
                patform = orig
                is_hermitian = True
            else:
                patform = orig.rewrite(BosonNum)
                pd = orig.parity_diff
                if pd == 0 or pd is S.Zero:
                    is_hermitian = True
                elif not pd.free_symbols:
                    is_hermitian = False
                else:
                    is_hermitian = None

            if action == 'normal_order':
                if _has_unknowns(patform, exp_symbols):
                    raise ValueError("Cannot perform normal ordering: undetermined symbols exist in the simplified expression.")
                norm = orig.to_normal_order(via_dict=True).rewrite(BosonOp)
            else:
                norm = None

        target = orig if action == 'original' else (patform if action == 'patform' else norm)
        has_ff = target.has(FallingFactorial)
        note = "Subscripts in (x)_k indicate the falling factorial (Pochhammer symbol): (x)_k = x(x-1)...(x-k+1), rather than operator powers." if has_ff else None
        return latex(target), is_hermitian, str(target), note

    else:  # fermion
        if not exps or all(e == 0 for e in exps):
            orig = simp = norm = S.One
            is_hermitian = True
        else:
            orig = FermionPat(name, is_ann, *exps, eval_exp=False)
            simp = orig.doit()
            if simp == 0 or simp is S.Zero:
                is_hermitian = True
            elif simp == 1 or simp is S.One:
                is_hermitian = True
            elif isinstance(simp, FermionOp):
                is_hermitian = False
            elif isinstance(simp, FermionPat) and simp.exps == (1, 1):
                is_hermitian = True
            else:
                is_herm_val = simp.is_hermitian
                if is_herm_val is True:
                    is_hermitian = True
                elif is_herm_val is False:
                    is_hermitian = False
                else:
                    is_hermitian = None

            if action == 'normal_order':
                if _has_unknowns(simp, exp_symbols):
                    raise ValueError("Cannot perform normal ordering: undetermined symbols exist in the simplified expression.")
                if simp == 0 or simp is S.Zero:
                    norm = S.Zero
                elif simp == 1 or simp is S.One:
                    norm = S.One
                elif isinstance(simp, FermionOp):
                    norm = simp
                elif isinstance(simp, FermionPat) and simp.exps == (1, 1):
                    if simp.left_base.is_annihilation:
                        # c c† -> 1 - c† c
                        norm = 1 - FermionPat(name, False, 1, 1)
                    else:
                        norm = simp
                else:
                    norm = simp
            else:
                norm = None

        target = orig if action == 'original' else (simp if action == 'patform' else norm)
        target_str = _fermion_target_to_expr_str(target, name)
        has_ff = target.has(FallingFactorial)
        note = "Subscripts in (x)_k indicate the falling factorial (Pochhammer symbol): (x)_k = x(x-1)...(x-k+1), rather than operator powers." if has_ff else None
        return latex(target), is_hermitian, target_str, note


