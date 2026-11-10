from collections.abc import Sequence
from functools import cache
from sympy.core.singleton import S
from sympy.core.numbers import I
from sympy.core.symbol import Symbol
from sympy.core.mul import Mul
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum.dagger import Dagger
from ..pattern_perturb.pattern_form import PF, pf_add, _vpk
from .dsolve_pm import dsd1, dsd2
from .multimode import MultimodeNum2
from .wm_processor import wm_processors, a_p, a_s, a_i, t_, nlc, _g
from .wm_wrapper import WMWrapper

__all__ = ['pm', 'PMSolver', 'TWMSolver', 'FWMSolver']

pm = Symbol('Δk', real=True)


class PMSolver(Sequence):
    """
    Protected list-like container storing perturbative wave-mixing solutions
    as 3-tuples of WMWrapper objects per order.
    """

    def __init__(self, dc=True, mismatch=False):
        self.processor = wm_processors[bool(dc)]
        self.pm = pm if mismatch else S.Zero
        self._sol = []

    @property
    def dc(self) -> bool:
        return self.processor.dc

    @property
    def nd(self):
        return self.processor.nd

    @property
    def pe(self):
        return self.processor.pe

    def __len__(self) -> int:
        return len(self._sol)

    def __getitem__(self, index):
        return self._sol[index]

    def __setitem__(self, index, value):
        raise TypeError(f"'{self.__class__.__name__}' object does not support item assignment")

    def __delitem__(self, index):
        raise TypeError(f"'{self.__class__.__name__}' object does not support item deletion")

    def get_sol(self, order: int):
        raise NotImplementedError


class TWMSolver(PMSolver):
    """
    Three-wave mixing quantum multimode parametric process perturbation solver.

    Under the wave-mixing ordering principle, the time differential equations are:
        d/dt a0 = -i χ a1 a2 exp(-i Δk t),
        d/dt a1 = -i χ a0 a2† exp(i Δk t),
        d/dt a2 = -i χ a1† a0 exp(i Δk t)
    """

    def __init__(self, dc=True, mismatch=False):
        super().__init__(dc=dc, mismatch=mismatch)
        self.half_p_factor = exp(I * self.pm * t_ * S.Half)
        p = self.processor
        if dc:
            self._sol = [(
                WMWrapper(p, {(None,): PF({((a_p,), ()): S.One})}),
                WMWrapper(p),
                WMWrapper(p),
            )]
            self.gc = S.Half * (4 * nlc**2 * self.nd - self.pm**2)**S.Half
        else:
            self._sol = [(
                WMWrapper(p),
                WMWrapper(p),
                WMWrapper(p, {(None,): PF({((a_i,), ()): S.One})}),
            )]
            self.gc = S.Half * (4 * nlc**2 * self.nd + self.pm**2)**S.Half

        # d/dt a_drive / (-i χ) for the last solved order
        self.dd_last = WMWrapper(p)

        @cache
        def _fdf0(n0, mode):
            if n0:
                if mode == 0:
                    return PF({_vpk: dsd2(self.gc * t_, dict(), False,
                                              S.One, I * self.pm * S.Half / self.gc)})
                return PF({_vpk: dsd2(self.gc * t_, dict(), self.dc,
                                          S.One, -I * self.pm * S.Half / self.gc)})
            elif self.dc:
                return PF({((a_p,), ()): dsd2(self.gc * t_, dict(), True,
                                              S.Zero, -I * nlc / self.gc)})
            elif mode == 0:
                return PF({((a_i,), ()): dsd2(self.gc * t_, dict(), False,
                                              S.Zero, -I * nlc / self.gc)})
            else:
                return PF({((Dagger(a_i),), ()): dsd2(self.gc * t_, dict(), False,
                                                      S.Zero, -I * nlc / self.gc)})
        self.fdf0 = _fdf0

    def get_sol(self, order: int):
        if order < -1:
            raise ValueError('The order should be at least -1')
        i = order + 1
        if len(self._sol) > i:
            return self._sol[i]
        last_sol = self.get_sol(order - 1)
        p = self.processor
        dc = self.dc
        gc_m2 = self.gc**(-2)
        if i & 1:
            def _d2ts(mode, n0, b):
                s1 = S.NegativeOne if mode == 0 else S.One
                s3 = S.NegativeOne if dc else S.One
                t1_b = term1.dict_repr.get(b, {})
                t2_b = term2.dict_repr.get(b, {})
                last_b = last_sol[mode].dict_repr.get(b, {})
                raw_b = {}
                for k in t1_b.keys() | t2_b.keys() | last_b.keys():
                    pfs = []
                    if k in t1_b:
                        pfs.append(s1 * t1_b[k])
                    if k in t2_b:
                        pfs.append(-t2_b[k])
                    if k in last_b:
                        pfs.append(last_b[k] * (s3 * self.nd))
                    raw_b[k] = pf_add(*pfs) * Mul(nlc**2, gc_m2)
                out_b = WMWrapper(p, raw_b).expand(order, True, self.gc).apply(
                    lambda t: p.n_simp(dsd2(_g, p.p2s_group(t), dc, subs=self.gc * t_))
                ).dict_repr
                out_b[(None,)] = (pf_add(out_b[(None,)], self.fdf0(n0, mode))
                                  if (None,) in out_b else self.fdf0(n0, mode))
                return out_b

            if dc:
                pd = last_sol[0].adjoint()
                term1 = last_sol[0] * pd * last_sol[1]
                term2 = self.dd_last * last_sol[2].adjoint()
                sol1 = WMWrapper(p, {a_s: _d2ts(1, True, a_s), Dagger(a_i): _d2ts(1, False, Dagger(a_i))})
                term1 = last_sol[2] * (pd * last_sol[0])
                term2 = last_sol[1].adjoint() * self.dd_last
                sol2 = WMWrapper(p, {Dagger(a_s): _d2ts(2, False, Dagger(a_s)), a_i: _d2ts(2, True, a_i)})
                self._sol.append((last_sol[0], sol1, sol2))
            else:
                id_ = last_sol[2].adjoint()
                term1 = last_sol[0] * (id_ * last_sol[2])
                term2 = last_sol[1] * self.dd_last
                sol0 = WMWrapper(p, {a_p: _d2ts(0, True, a_p), a_s: _d2ts(0, False, a_s)})
                term1 = last_sol[0] * self.dd_last.adjoint()
                term2 = last_sol[1] * (last_sol[2] * id_)
                sol1 = WMWrapper(p, {a_s: _d2ts(1, True, a_s), a_p: _d2ts(1, False, a_p)})
                self._sol.append((sol0, sol1, last_sol[2]))
        else:
            def _d1ts():
                b = self._sol[0][0] if dc else self._sol[0][2]
                w_rhs = (self.dd_last * (-I * nlc / self.gc)).expand(order, True, self.gc)
                s0 = w_rhs.apply(lambda t: p.n_simp(dsd1(_g, p.p2s_group(t), subs=self.gc * t_)))
                return s0 + b

            if dc:
                self.dd_last = last_sol[1] * last_sol[2]
                self._sol.append((_d1ts(), *last_sol[1:]))
            else:
                self.dd_last = last_sol[1].adjoint() * last_sol[0]
                self._sol.append((*last_sol[:2], _d1ts()))
        return self._sol[i]

    def sol_expr(self, mode: int, order: int):
        """
        Get the expression of the solution for the given mode and order.
        """
        sol = self.get_sol(order)[mode]
        if sol.b_left is None:
            p = S.One
        elif mode == 0:
            p = self.half_p_factor**S.NegativeOne
        else:
            p = self.half_p_factor
        return p * sol.expr


class FWMSolver(PMSolver):
    """
    Four-wave mixing quantum multimode parametric process perturbation solver.
    """

    def __init__(self):
        super().__init__(dc=True, mismatch=True)
        self.hpm = -I * self.pm * S.Half + I * nlc * self.nd
        self.gc = S.Half * (self.pm * (-self.pm + 4 * nlc * self.nd))**S.Half
        self._solve()

    def _solve(self):
        p = self.processor
        sol_m1 = (
            WMWrapper(p, {(None,): PF({((a_p,), ()): S.One})}),
            WMWrapper(p),
            WMWrapper(p),
        )

        fwm0_0 = WMWrapper(p, {(None,): PF({((a_p,), ()): S.One})})
        fwm0_1 = WMWrapper(p, {
            a_s: {(None,): PF({_vpk: dsd2(self.gc * t_, dict(),
                                              True, S.One, self.hpm / self.gc)})},
            Dagger(a_i): {(None,): PF({((a_p**2,), ()): dsd2(self.gc * t_, dict(),
                                                             True, S.Zero, -I * nlc / self.gc)})}
        })
        fwm0_2 = WMWrapper(p, {
            Dagger(a_s): {(None,): PF({((a_p**2,), ()): dsd2(self.gc * t_, dict(),
                                                             True, S.Zero, -I * nlc / self.gc)})},
            a_i: {(None,): PF({_vpk: dsd2(self.gc * t_, dict(),
                                              True, S.One, self.hpm / self.gc)})}
        })
        sol_0 = (fwm0_0, fwm0_1, fwm0_2)

        xpm_c = WMWrapper(p, {(None,): PF({((a_p,), ()): S.Half * MultimodeNum2(a_s.name, a_i.name)})})
        coeff = WMWrapper(p, {(None,): PF({((a_p,), ()): S.Half})})
        xpm_t = sol_0[1] * sol_0[1].adjoint() + sol_0[2].adjoint() * sol_0[2]
        dd2 = sol_0[1] * sol_0[0].adjoint() * sol_0[2]

        p.pe.update({a_p.name: -S.Half})
        w_rhs = ((dd2 + xpm_c + coeff * xpm_t) * (-2 * I * nlc / self.gc)).expand(1, True, self.gc)
        s0 = w_rhs.apply(lambda t: p.n_simp(dsd1(_g, p.p2s_group(t), subs=self.gc * t_)))
        p.pe.update({a_p.name: S.NegativeOne})

        sol_1 = (sol_0[0] + s0, sol_0[1], sol_0[2])
        self._sol = [sol_m1, sol_0, sol_1]

    def get_sol(self, order: int):
        if order < -1 or order > 1:
            raise ValueError(f"Order {order} out of range for FWMSolver (supports orders -1, 0, 1)")
        return self._sol[order + 1]

    def sol_expr(self, mode: int, order: int):
        expr = self.get_sol(order)[mode].expr
        if mode == 0 and order >= 0:
            return expr * exp(-I * nlc * self.nd * t_)
        return expr

    def _zq(self, obj):
        f = lambda x: x.subs({self.pm: 2 * nlc * self.nd})
        return obj.apply(f) if isinstance(obj, WMWrapper) else f(obj)
