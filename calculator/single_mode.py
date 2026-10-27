from __future__ import annotations
import re
from typing import Tuple

from op_patterns import BosonNum, BosonPat, FermionPat
from sympy.core.singleton import S
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex

from .common import expr_to_code_str


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
    return expr_to_code_str(target)


def eval_single_mode(mode_type: str, name: str, first_op: str, exp_str: str, action: str) -> Tuple[str, bool, str]:
    """
    Evaluates single-mode quantum operator product (BosonPat or FermionPat).
    Returns: (latex_str, is_hermitian, expr_str)
    """
    name = (name or '').strip()
    if not name:
        name = 'a' if mode_type == 'boson' else 'c'
    is_ann = (first_op == 'annihilation')
    exps = [int(x) for x in re.findall(r'\d+', exp_str or '')]

    if mode_type == 'boson':
        if not exps or all(e == 0 for e in exps):
            orig = patform = norm = S.One
            is_hermitian = True
        else:
            orig = BosonPat(name, is_ann, *exps, eval_exp=False)
            if isinstance(orig, BosonOp):
                patform = norm = orig
                is_hermitian = False
            elif isinstance(orig, BosonNum):
                patform = norm = orig
                is_hermitian = True
            else:
                patform = orig.rewrite(BosonNum)
                norm = orig.to_normal_order(via_dict=True).rewrite(BosonOp)
                is_hermitian = (getattr(orig, 'parity_diff', None) == 0)

        target = orig if action == 'original' else (patform if action == 'patform' else norm)
        return latex(target), is_hermitian, expr_to_code_str(target)

    else:  # fermion
        if not exps or all(e == 0 for e in exps):
            orig = simp = norm = S.One
            is_hermitian = True
        else:
            orig = FermionPat(name, is_ann, *exps, eval_exp=False)
            simp = orig.doit()
            if simp == 0:
                norm = S.Zero
                is_hermitian = True
            elif simp == 1:
                norm = S.One
                is_hermitian = True
            elif isinstance(simp, FermionOp):
                norm = simp
                is_hermitian = False
            elif isinstance(simp, FermionPat) and simp.exps == (1, 1):
                if simp.left_base.is_annihilation:
                    # c c† -> 1 - c† c
                    norm = 1 - FermionPat(name, False, 1, 1)
                    is_hermitian = True
                else:
                    norm = simp
                    is_hermitian = True
            else:
                norm = simp
                is_hermitian = (getattr(simp, 'is_hermitian', None) is True)

        target = orig if action == 'original' else (simp if action == 'patform' else norm)
        target_str = _fermion_target_to_expr_str(target, name)
        return latex(target), is_hermitian, target_str

