from __future__ import annotations
from typing import Optional, Tuple

from cas_qwm.op_pattern import BosonNum
from cas_qwm.pattern_perturb import PF
from cas_qwm.pattern_perturb.pattern_form import _ps, bn_simp
from cas_qwm.parametric_multimode.wm_processor import t_, _g, nlc, a_p, a_i
from cas_qwm.parametric_multimode.wm_solver import pm
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.complexes import Abs

from .common import g_hat
from .conditions import normalize_process

__all__ = ['t_to_g', 'apply_assertions']

alphas = (Symbol('alpha_2'), Symbol('alpha_0'))
gain = Symbol('g', real=True)


def t_to_g(
    obj,
    process: Optional[str] = None,
    mismatch: bool = False,
    dc: Optional[bool] = None,
    coherent: bool = False,
    alpha=None,
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
        nd_drive = Abs(alphas[is_dc] if alpha is None else alpha) ** 2
        gc = gc.subs({nd: nd_drive})
        if proc != 'fwm' and not mismatch:
            gc = _ps(gc)
        g_sym, extra_subs = gain, {nd: nd_drive}
    else:
        g_sym, extra_subs = g_hat, {}

    inv_gc = bottom_up(1 / gc, _ps)
    sub_dict = {t_: g_sym * inv_gc, _g: g_sym, **extra_subs}

    f = lambda expr: bn_simp(expr.subs(sub_dict))
    return obj.apply(f, on_npf=True) if hasattr(obj, 'apply') else f(obj)


def apply_assertions(
    val,
    s,
    order: int,
    process: Optional[str] = None,
    mismatch: bool = False,
    single_mode: bool = False,
    coherent_drive: bool = False,
    vacuum_signal: bool = False,
    vacuum_other_weak: bool = False,
    times: Tuple = (),
    alpha=None,
    degenerate: bool = False,
) -> Tuple:
    """
    Evaluates a parsed Tab 3 WMWrapper under the input-state assertions and returns
    its time and gain representations (val_t, val_g):
    - coherent_drive: strong driving field input is in coherent state |alpha>, with the
      amplitude symbol `alpha` (default: alpha_0 for dc, alpha_2 for fc)
    - vacuum_signal: signal field (a_1) input is in vacuum |0>
    - vacuum_other_weak: remaining weak field (a_2 if dc else a_0) input is in vacuum |0>
    Without any assertion, the result is expanded to the given order and simplified
    by product-to-sum grouping of the functions of t and, separately, of each further
    time variable in `times` (cascaded systems).
    With `degenerate` (single-mode down-conversion, signal = idler = a_1), the value is
    converted to the degenerate single mode first, and either vacuum assertion puts
    the degenerate mode a_1 in vacuum.
    """
    proc = normalize_process(process, s.dc)
    mismatch = bool(mismatch)
    p = s.processor
    degenerate = bool(single_mode) and bool(degenerate) and proc == 'twm_dc'
    if degenerate:
        val = val.to_single_mode(degenerate=True)

    vac_modes = []
    if degenerate:
        if vacuum_signal or vacuum_other_weak:
            vac_modes.append('a_1')
    else:
        if vacuum_signal:
            vac_modes.append('a_1')
        if vacuum_other_weak:
            vac_modes.append('a_2' if s.dc else 'a_0')
    if vac_modes:
        val = val.eval_vacuum(vac_modes)

    if coherent_drive:
        if single_mode:
            val = val.to_single_mode(degenerate=degenerate)
        if alpha is None:
            alpha = alphas[s.dc]
        val_t = val.eval_coherent(order, alpha)
        if proc == 'fwm' and not mismatch:
            pm_cohr = 2 * nlc * (Abs(alpha) ** 2)
            val_t = val_t.apply(lambda x: bottom_up(x.subs({s.pm: pm_cohr}), _ps), on_npf=True)
        return val_t, t_to_g(val_t, process=proc, mismatch=mismatch, coherent=True, alpha=alpha)

    ts = tuple(dict.fromkeys((t_, *times)))

    def _postprocess_pf(pf_item: PF) -> PF:
        if not pf_item:
            return pf_item
        if order >= 0:
            return pf_item.apply(lambda t: p.p2s(t, ts, simp=p.n_simp))
        return pf_item.apply(p.n_simp)

    val = val.expand(order)
    if proc == 'fwm' and not mismatch:
        val = s._zq(val)
    val = val.apply(_postprocess_pf)
    if single_mode:
        val = val.to_single_mode(degenerate=degenerate)
    return val, t_to_g(val, process=proc, mismatch=mismatch)
