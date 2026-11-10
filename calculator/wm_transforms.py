from __future__ import annotations
from typing import Optional

from cas_qwm.op_pattern import BosonNum
from cas_qwm.parametric_multimode.multimode import _ps, nd_simp
from cas_qwm.parametric_multimode.wm_processor import t_, _g, nlc, a_p, a_i
from cas_qwm.parametric_multimode.wm_solver import pm
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.complexes import Abs

from .common import g_hat
from .conditions import normalize_process

__all__ = ['t_to_g']

alphas = (Symbol('alpha_2'), Symbol('alpha_0'))
gain = Symbol('g', real=True)


def t_to_g(
    obj,
    process: Optional[str] = None,
    mismatch: bool = False,
    dc: Optional[bool] = None,
    coherent: bool = False,
):
    r"""
    Convert time variable t_ to parametric gain (operator \hat{g} / gc or scalar g / gc)
    and simplify expressions involving the driving field.
    Supports Three-Wave Mixing ('twm_dc', 'twm_fc') and Four-Wave Mixing ('fwm').
    """
    proc = normalize_process(process, dc)
    is_dc = (proc != 'twm_fc')
    nd = BosonNum(a_p.name if is_dc else a_i.name)

    if proc == 'fwm':
        gc = S.Half * (pm * (-pm + 4 * nlc * nd)) ** S.Half if mismatch else nlc * nd
    elif mismatch:
        gc = S.Half * (4 * nlc**2 * nd - pm**2)**S.Half if is_dc else S.Half * (4 * nlc**2 * nd + pm**2)**S.Half
    else:
        gc = nlc * nd ** S.Half

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

    f = lambda expr: nd_simp(expr.subs(sub_dict), nd_drive)
    return obj.apply(f) if hasattr(obj, 'apply') else f(obj)
