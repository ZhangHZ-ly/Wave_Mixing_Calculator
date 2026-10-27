from __future__ import annotations

from op_patterns import BosonNum
from parametric import (
    f_dict, d_apply, t_, _g, _nd, _ps, nlc, pm, a_p, a_i
)
from perturb_eval import PF
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.traversal import bottom_up

from .common import g_hat


def t_to_g(obj, dc: bool = True, mismatch: bool = False):
    r"""
    Convert time variable t_ to parametric gain operator \hat{g} / gc
    and simplify commutative number-operator expressions.
    Independent of TWMSolver; only requires dc and mismatch to identify the driving mode and effective coupling.
    """
    nd = BosonNum(a_p.name if dc else a_i.name)
    if mismatch:
        gc = (nlc**2 * nd - pm**2 / 4)**S.Half if dc else (nlc**2 * nd + pm**2 / 4)**S.Half
    else:
        gc = nlc * nd ** S.Half

    def _simp_n(expr):
        if not isinstance(expr, Expr):
            return expr
        expr = expr.subs({t_: g_hat / gc, _g: g_hat})

        def _combine(e):
            if e.is_Mul:
                n_part = []
                other = []
                for arg in Mul.make_args(e.subs({nd: _nd, g_hat: _g})):
                    if arg.has(_nd, _g):
                        n_part.append(arg)
                    else:
                        other.append(arg)
                if n_part:
                    simp_n = _ps(Mul(*n_part)).subs({_nd: nd, _g: g_hat})
                    e = Mul(*other, simp_n)
            return _ps(e)

        return bottom_up(expr, _combine)

    if isinstance(obj, (PF, dict, f_dict)):
        return d_apply(obj, _simp_n)
    return _simp_n(obj)


def g_to_t(obj, dc: bool = True, mismatch: bool = False):
    r"""
    Inverse of t_to_g: convert parametric gain operator \hat{g} back to gc * t_
    and simplify commutative number-operator expressions.
    Independent of TWMSolver; only requires dc and mismatch to identify the driving mode and effective coupling.
    """
    nd = BosonNum(a_p.name if dc else a_i.name)
    if mismatch:
        gc = (nlc**2 * nd - pm**2 / 4)**S.Half if dc else (nlc**2 * nd + pm**2 / 4)**S.Half
    else:
        gc = nlc * nd ** S.Half

    def _simp_n(expr):
        if not isinstance(expr, Expr):
            return expr
        expr = expr.subs({g_hat: gc * t_, _g: gc * t_})

        def _combine(e):
            if e.is_Mul:
                n_part = []
                other = []
                for arg in Mul.make_args(e.subs({nd: _nd})):
                    if arg.has(_nd):
                        n_part.append(arg)
                    else:
                        other.append(arg)
                if n_part:
                    simp_n = _ps(Mul(*n_part)).subs({_nd: nd})
                    e = Mul(*other, simp_n)
            return _ps(e)

        return bottom_up(expr, _combine)

    if isinstance(obj, (PF, dict, f_dict)):
        return d_apply(obj, _simp_n)
    return _simp_n(obj)

