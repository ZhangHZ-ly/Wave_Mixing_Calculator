from __future__ import annotations
from typing import Optional

from wave_mixing.pattern_form.op_patterns import BosonNum
from wave_mixing.multimode import d_apply, _ps, nd_simp
from wave_mixing.wm_processor import t_, _g, nlc, pm, a_p, a_i
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.complexes import Abs

from .common import g_hat
from .conditions import normalize_process

__all__ = ['t_to_g']

alphas = (Symbol('alpha_2'), Symbol('alpha_0'))
gain = Symbol('g', real=True)

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
    elif solver is not None and solver.__class__.__name__ == 'FWMSolver':
        proc = 'fwm'
    elif solver is not None:
        proc = 'twm_dc' if solver.dc else 'twm_fc'
    else:
        proc = normalize_process(None, dc)

    is_dc = (proc != 'twm_fc')
    nd, gc = _resolve_nd_and_gc(process=proc, mismatch=mismatch, dc=is_dc)

    if coherent:
        nd_drive = Abs(alphas[is_dc]) ** 2
        gc = gc.subs({nd: nd_drive})
        if proc != 'fwm' and not mismatch:
            gc = _ps(gc)
        inv_gc = bottom_up(1 / gc, _ps)
        sub_dict = {t_: gain * inv_gc, _g: gain, nd: nd_drive}
    else:
        nd_drive = nd
        inv_gc = bottom_up(1 / gc, _ps)
        sub_dict = {t_: g_hat * inv_gc, _g: g_hat}

    return d_apply(obj, lambda expr: nd_simp(expr.subs(sub_dict), nd_drive))
