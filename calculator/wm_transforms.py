from __future__ import annotations
from typing import Optional, Tuple

from cas_qwm.op_pattern import BosonNum
from cas_qwm.pattern_perturb import PF
from cas_qwm.pattern_perturb.pattern_form import _ps, bn_simp
from cas_qwm.wm_variables import a_p, a_i, t_, nlc, pm, alpha_0, alpha_2, gain, g_hat
from sympy.core.function import _mexpand
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp

from .conditions import normalize_process

__all__ = ['t_to_g', 'apply_assertions']



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
        nd_drive = Abs((alpha_0 if is_dc else alpha_2) if alpha is None else alpha) ** 2
        gc = gc.subs({nd: nd_drive})
        if proc != 'fwm' and not mismatch:
            gc = _ps(gc)
        g_sym, extra_subs = gain, {nd: nd_drive}
    else:
        g_sym, extra_subs = g_hat, {}

    inv_gc = bottom_up(1 / gc, _ps)
    sub_dict = {t_: g_sym * inv_gc, **extra_subs}

    if proc == 'fwm' and mismatch:
        # the radicand of gc appears both factored and expanded (product-to-sum); all its powers
        # are brought to the expanded form so that gc t -> g cancels
        rad = _mexpand((2 * gc) ** 2)
        sub_dict[t_] = g_sym * 2 * rad ** (-S.Half)
        is_rad = lambda x: x.is_Pow and x.base.has(pm) and x.base != rad and _mexpand(x.base) == rad

        # the self-phase-modulation phases exp(±i nlc N t) keep t (in g they would carry 1/gc)
        def f(expr):
            spm = {a: Dummy() for a in expr.atoms(exp) if a.has(t_) and not a.has(pm)}
            back = {d: a.subs(extra_subs) for a, d in spm.items()}
            e = expr.xreplace(spm).subs(extra_subs)
            e = e.replace(is_rad, lambda x: rad ** x.exp).subs(sub_dict)
            return bn_simp(e).xreplace(back)
    else:
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
    seed_mode: Optional[int] = None,
    seed_vac_c: bool = True,
    seed_observed: bool = False,
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
    - seed_mode (None: no seeding), seed_vac_c, seed_observed: coherent seeding of a separate
      weak mode |beta_j>, coupled with its partner in vacuum (seed_vac_c) or in |beta_k>; see
      WMWrapper.seeded_separate for the modes (down-conversion: 1 signal, 0 degenerate;
      state transfer: the index of the seeded mode; four-wave mixing: signal). In multimode
      processing, the seeded modes are split from the multimode sums (whose rest keeps n_m)
      and averaged exactly; `seed_observed` makes the single-mode weak-field operators the
      seeded modes themselves, and the vacuum assertions then apply to the remaining modes.
      In single-mode processing, the weak modes themselves are in these states (the vacuum
      assertions of the weak fields are void); a degenerate seed then requires degenerate
      single-mode processing.
    """
    proc = normalize_process(process, s.dc)
    mismatch = bool(mismatch)
    p = s.processor
    degenerate = bool(single_mode) and bool(degenerate) and proc == 'twm_dc'
    if degenerate:
        val = val.to_single_mode(degenerate=True)

    seeding = seed_mode is not None
    if seeding:
        if not single_mode:
            val = val.seeded_separate(seed_mode, seed_vac_c, observed=seed_observed)
        elif degenerate:  # the degenerate single mode a_1 itself is in |beta_1>
            val = val.seeded_separate(1, True, observed=True)
        else:
            if proc == 'twm_dc' and int(seed_mode) == 0:
                raise ValueError('A degenerate seed requires multimode or degenerate '
                                 'single-mode processing')
            val = val.to_single_mode().seeded_separate(seed_mode, seed_vac_c, observed=True)

    # with seeding in single-mode processing, the weak modes are in the seeded states
    vac_modes = []
    if seeding and single_mode:
        vacuum_signal = vacuum_other_weak = False
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
            alpha = alpha_0 if s.dc else alpha_2
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
