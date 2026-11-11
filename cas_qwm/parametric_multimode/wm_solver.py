from collections.abc import Sequence
from functools import cache
from sympy.core.singleton import S
from sympy.core.numbers import I
from sympy.core.symbol import Symbol
from sympy.core.mul import Mul
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum.dagger import Dagger
from ..pattern_perturb.pattern_form import PF, _vpk
from ..dsolve_pm import dsd1, dsd2
from .multimode import MultimodeNum2
from .wm_processor import wm_processors, a_p, a_s, a_i, t_, nlc, _g
from .wm_wrapper import WMWrapper

__all__ = ['pm', 'WMSolver', 'TWMSolver', 'FWMSolver']

pm = Symbol('Δk', real=True)


class WMSolver(Sequence):
    """
    Protected list-like container storing perturbative wave-mixing solutions
    as 3-tuples of WMWrapper objects per order.
    """
    __slots__ = ('processor', 'pm', '_sol')

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


class TWMSolver(WMSolver):
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
        self.reset_pe()
        gc_m2 = self.gc**(-2)
        if i & 1:
            s3 = S.NegativeOne if dc else S.One
            gc2 = Mul(nlc**2, gc_m2)

            def _d2ts(mode, b_t, b_f, term1, term2):
                init = (self._sol[1][mode] if order > 0 else
                        WMWrapper(p, {b_t: {(None,): self.fdf0(True, mode)},
                                      b_f: {(None,): self.fdf0(False, mode)}}))
                t1 = -term1 if mode == 0 else term1
                rhs = p.d_add(t1, -term2, last_sol[mode] @ (s3 * self.nd))
                w_rhs = (rhs @ gc2).expand(order, True, self.gc)
                sol = w_rhs.apply(
                    lambda t: p.n_simp(dsd2(_g, p.p2s_group(t), dc, subs=self.gc * t_))
                )
                return sol + init

            if dc:
                pd = last_sol[0].adjoint()
                sol1 = _d2ts(1, a_s, Dagger(a_i),
                             last_sol[0] @ pd @ last_sol[1],
                             self.dd_last @ last_sol[2].adjoint())
                sol2 = _d2ts(2, a_i, Dagger(a_s),
                             last_sol[2] @ (pd @ last_sol[0]),
                             last_sol[1].adjoint() @ self.dd_last)
                self._sol.append((last_sol[0], sol1, sol2))
            else:
                id_ = last_sol[2].adjoint()
                sol0 = _d2ts(0, a_p, a_s,
                             last_sol[0] @ (id_ @ last_sol[2]),
                             last_sol[1] @ self.dd_last)
                sol1 = _d2ts(1, a_s, a_p,
                             last_sol[0] @ self.dd_last.adjoint(),
                             last_sol[1] @ (last_sol[2] @ id_))
                self._sol.append((sol0, sol1, last_sol[2]))
        else:
            def _d1ts():
                b = self._sol[0][0] if dc else self._sol[0][2]
                w_rhs = (self.dd_last * (-I * nlc / self.gc)).expand(order, True, self.gc)
                s0 = w_rhs.apply(lambda t: p.n_simp(dsd1(_g, p.p2s_group(t), subs=self.gc * t_)))
                return s0 + b

            if dc:
                self.dd_last = last_sol[1] @ last_sol[2]
                self._sol.append((_d1ts(), *last_sol[1:]))
            else:
                self.dd_last = last_sol[1].adjoint() @ last_sol[0]
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

    def reset_pe(self):
        p = self.processor
        p.pe.update({name: S.Zero for name in p.a[0].names})
        p.pe.update({p.nd.name: S.NegativeOne})


class FWMSolver(WMSolver):
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
        xpm_t = sol_0[1] @ sol_0[1].adjoint() + sol_0[2].adjoint() @ sol_0[2]
        dd2 = sol_0[1] @ sol_0[0].adjoint() @ sol_0[2]

        self.reset_pe()
        w_rhs = ((dd2 + xpm_c + coeff @ xpm_t) @ (-2 * I * nlc / self.gc)).expand(1, True, self.gc)
        s0 = w_rhs.apply(lambda t: p.n_simp(dsd1(_g, p.p2s_group(t), subs=self.gc * t_)))

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

    def reset_pe(self):
        self.processor.pe.update({a_p.name: -S.Half, a_s.name: S.Zero, a_i.name: S.Zero})
