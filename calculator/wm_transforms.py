from __future__ import annotations
from typing import Optional

from op_patterns import BosonNum
from parametric import (
    f_dict, d_apply, t_, _g, _nd, _ps, nlc, pm, a_p, a_i
)
from perturb_eval import PF
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.traversal import bottom_up

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


def _is_pure_nd_power(a, nd_sym):
    if a == nd_sym:
        return True, S.One
    if a.is_Pow and a.base == nd_sym:
        return True, a.exp
    return False, S.Zero


def t_to_g(
    obj,
    process: Optional[str] = None,
    mismatch: bool = False,
    dc: Optional[bool] = None,
):
    r"""
    Convert time variable t_ to parametric gain operator \hat{g} / gc
    and simplify commutative number-operator expressions.
    Supports Three-Wave Mixing ('twm_dc', 'twm_fc') and Four-Wave Mixing ('fwm').
    """
    nd, gc = _resolve_nd_and_gc(process=process, mismatch=mismatch, dc=dc)

    def _simp_n(expr):
        if not isinstance(expr, Expr):
            return expr
        expr = expr.subs({t_: g_hat / gc, _g: g_hat})

        def _combine(e):
            if e.is_Mul:
                args = []
                for a in Mul.make_args(e.subs({nd: _nd, g_hat: _g})):
                    if a.is_Add and a.has(_nd):
                        args.extend(Mul.make_args(factor_terms(a, clear=True)))
                    elif a.is_Pow and a.base.is_Add and a.base.has(_nd):
                        args.extend(arg**a.exp for arg in Mul.make_args(factor_terms(a.base, clear=True)))
                    else:
                        args.append(a)

                nd_powers = []
                add_factors = []
                other_nd = []
                other = []
                for arg in args:
                    is_p, p_exp = _is_pure_nd_power(arg, _nd)
                    if is_p:
                        nd_powers.append(p_exp)
                    elif arg.is_Add and arg.has(_nd):
                        add_factors.append(arg)
                    elif arg.has(_nd, _g):
                        other_nd.append(arg)
                    else:
                        other.append(arg)

                total_nd_exp = Add(*nd_powers) if nd_powers else S.Zero
                if total_nd_exp != S.Zero and add_factors:
                    target_idx = 0
                    if len(add_factors) > 1:
                        for i, af in enumerate(add_factors):
                            if total_nd_exp > 0 and any(t.has(_nd) and any(_is_pure_nd_power(f, _nd)[0] and _is_pure_nd_power(f, _nd)[1] < 0 for f in Mul.make_args(t)) for t in af.args):
                                target_idx = i
                                break
                            elif total_nd_exp < 0 and any(t.has(_nd) and any(_is_pure_nd_power(f, _nd)[0] and _is_pure_nd_power(f, _nd)[1] > 0 for f in Mul.make_args(t)) for t in af.args):
                                target_idx = i
                                break
                    nd_factor = _nd**total_nd_exp
                    first_add = add_factors[target_idx]
                    new_terms = [Mul(term, nd_factor) for term in first_add.args]
                    add_factors[target_idx] = Add(*new_terms)
                    total_nd_exp = S.Zero

                n_factors = list(other_nd)
                if total_nd_exp != S.Zero:
                    n_factors.append(_nd**total_nd_exp)
                n_factors.extend(add_factors)

                n_part = _ps(Mul(*n_factors)).subs({_nd: nd, _g: g_hat}) if n_factors else S.One
                e = Mul(*other, n_part)
            return _ps(e)

        return bottom_up(expr, _combine)

    if isinstance(obj, (PF, dict, f_dict)):
        return d_apply(obj, _simp_n)
    return _simp_n(obj)


