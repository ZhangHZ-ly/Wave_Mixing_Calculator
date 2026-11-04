from __future__ import annotations
from typing import Optional

from op_patterns import BosonNum
from parametric import (
    f_dict, d_apply, t_, _g, _ps, nlc, pm, a_p, a_i, nd_simp
)
from perturb_eval import PF
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.complexes import Abs

from .common import g_hat
from .conditions import normalize_process


def _resolve_nd_and_gc(
    process: Optional[str] = None,
    mismatch: bool = False,
    dc: Optional[bool] = None,
):
    proc = normalize_process(process, dc)
    if proc == 'fwm':
        nd = BosonNum(a_p.name)
        if mismatch:
            gc = S.Half * (pm * (-pm + 4 * nlc * nd)) ** S.Half
        else:
            gc = nlc * nd
        return nd, gc

    is_dc = (proc != 'twm_fc')
    nd = BosonNum(a_p.name if is_dc else a_i.name)
    if mismatch:
        gc = S.Half * (4 * nlc**2 * nd - pm**2)**S.Half if is_dc else S.Half * (4 * nlc**2 * nd + pm**2)**S.Half
    else:
        gc = nlc * nd ** S.Half
    return nd, gc


def t_to_g(
    obj,
    process: Optional[str] = None,
    mismatch: bool = False,
    dc: Optional[bool] = None,
    coherent: bool = False,
    gain_sym=None,
    alpha=None,
    solver=None,
):
    r"""
    Convert time variable t_ to parametric gain (operator \hat{g} / gc or scalar g / gc)
    and simplify expressions involving the driving field.
    Supports Three-Wave Mixing ('twm_dc', 'twm_fc') and Four-Wave Mixing ('fwm').
    """
    if hasattr(obj, 'val'):
        obj = obj.val

    if process is not None and not isinstance(process, str):
        solver = process
        process = None

    if process is not None:
        proc = normalize_process(process, dc)
    elif isinstance(solver, str):
        proc = normalize_process(solver, dc)
    elif solver is not None and getattr(solver, '__class__', None).__name__ == 'FWMSolver':
        proc = 'fwm'
    elif solver is not None and hasattr(solver, 'dc'):
        proc = 'twm_dc' if solver.dc else 'twm_fc'
    else:
        proc = normalize_process(None, dc)

    is_dc = (proc != 'twm_fc')
    nd, gc = _resolve_nd_and_gc(process=proc, mismatch=mismatch, dc=is_dc)

    if coherent:
        gain = Symbol('g', real=True) if gain_sym is None else gain_sym
        if alpha is None:
            alpha = Symbol('alpha_0') if is_dc else Symbol('alpha_2')
        nd_drive = Abs(alpha) ** 2
        gc = gc.subs({nd: nd_drive})
        if proc != 'fwm' and not mismatch:
            gc = _ps(gc)
        inv_gc = bottom_up(1 / gc, _ps)
        sub_dict = {t_: gain * inv_gc, _g: gain, nd: nd_drive}
    else:
        gain = g_hat if gain_sym is None else gain_sym
        nd_drive = nd
        inv_gc = bottom_up(1 / gc, _ps)
        sub_dict = {t_: gain * inv_gc, _g: gain}

    def _simp_single(expr):
        if not hasattr(expr, 'subs'):
            return expr
        expr = bottom_up(expr.subs(sub_dict), _ps)
        expr = nd_simp(expr, nd_drive)
        return _ps(expr)

    if isinstance(obj, (PF, dict, f_dict)):
        return d_apply(obj, _simp_single)
    return _simp_single(obj)


cohr_t_to_g = lambda obj, *args, **kwargs: t_to_g(obj, *args, coherent=True, **kwargs)
