from collections.abc import Sequence
from functools import cache
from sympy.core.singleton import S
from sympy.core.numbers import I
from sympy.core.mul import Mul
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum.dagger import Dagger
from ..pattern_perturb.pattern_form import PF, _vpk
from ..dsolve_pm import dsd1, dsd2
from .multimode import MultimodeNum2
from ..wm_variables import a_p, a_s, a_i, t_, nlc, pm, gain
from .wm_processor import WMProcessor
from .wm_wrapper import WMWrapper

__all__ = ['twm_processors', 'fwm_processor', 'WMSolver', 'TWMSolver', 'FWMSolver']

# Three- and four-wave mixing are maintained by separate processors, so that each keeps
# its own perturbative mode orders (drive -1 for TWM, -1/2 for FWM; weak fields 0)
# and no order setting has to be switched between calculations.
twm_processors = (WMProcessor(False), WMProcessor(True))  # indexed by dc
fwm_processor = WMProcessor(True, nlc_order=2)


class WMSolver(Sequence):
    """
    Protected list-like container storing perturbative wave-mixing solutions
    as 3-tuples of WMWrapper objects per order.
    """
    __slots__ = ('processor', 'pm', '_sol', '_labeled')

    def __init__(self, processor: WMProcessor, mismatch=False):
        self.processor = processor
        self.pm = pm if mismatch else S.Zero
        # TWM: exact (unlabeled) solutions used by the recursion, get_sol returns copies labeled
        # with their accuracy order (cached in _labeled); FWM: the labeled solutions themselves
        self._sol = []
        self._labeled = {}

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
        """
        3-tuple of the solutions (a_0(t), a_1(t), a_2(t)) accurate up to `order`, as wrappers
        whose `order` attribute is set to their actual accuracy (see sol_orders).
        """
        if order not in self._labeled:
            self._labeled[order] = tuple(w.with_order(o) for w, o in
                                         zip(self._get_sol(order), self.sol_orders(order)))
        return self._labeled[order]

    def sol_orders(self, order: int) -> tuple:
        """Accuracy orders of the three solutions of get_sol(order)."""
        return (order,) * 3

    def _get_sol(self, order: int):
        """Solutions accurate up to `order`, without accuracy label (exact arithmetic)."""
        raise NotImplementedError

    def _expand_g(self, w: WMWrapper, order) -> WMWrapper:
        """
        Expands w to the given order and rewrites t_ as gain / gc for the g-domain dsolve.
        The result is kept unlabeled (exact arithmetic): the recursion needs the products of
        the lower-order solutions without truncation.
        """
        sub = {t_: gain / self.gc}
        return w.expand(order).apply(lambda t: t.subs(sub), on_npf=True).with_order(S.Infinity)


class TWMSolver(WMSolver):
    """
    Three-wave mixing quantum multimode parametric process perturbation solver.

    Under the wave-mixing ordering principle, the time differential equations are:
        d/dt a0 = -i χ a1 a2 exp(-i Δk t),
        d/dt a1 = -i χ a0 a2† exp(i Δk t),
        d/dt a2 = -i χ a1† a0 exp(i Δk t)
    """

    def sol_orders(self, order: int) -> tuple:
        """
        The weak fields are corrected at even orders only and the strong driving field at odd
        orders only, so a solution is accurate up to the order before its next correction:
        order + 1 for the weak fields at even orders and for the drive at odd orders.
        """
        drive = 0 if self.dc else 2
        return tuple(order + 1 if bool(order & 1) == (m == drive) else order for m in range(3))

    def __init__(self, dc=True, mismatch=False):
        super().__init__(twm_processors[bool(dc)], mismatch=mismatch)
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

    def _get_sol(self, order: int):
        if order < -1:
            raise ValueError('The order should be at least -1')
        i = order + 1
        if len(self._sol) > i:
            return self._sol[i]
        last_sol = self._get_sol(order - 1)
        p = self.processor
        dc = self.dc
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
                w_rhs = self._expand_g(rhs @ gc2, order)
                sol = w_rhs.apply(
                    lambda t: p.n_simp(dsd2(gain, p.p2s_group(t), dc, subs=self.gc * t_))
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
                w_rhs = self._expand_g(self.dd_last * (-I * nlc / self.gc), order)
                s0 = w_rhs.apply(lambda t: p.n_simp(dsd1(gain, p.p2s_group(t), subs=self.gc * t_)))
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


class FWMSolver(WMSolver):
    """
    Four-wave mixing quantum multimode parametric process perturbation solver.
    """

    def __init__(self):
        super().__init__(fwm_processor, mismatch=True)
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

        w_rhs = self._expand_g((dd2 + xpm_c + coeff @ xpm_t) @ (-2 * I * nlc / self.gc), 2)
        s0 = w_rhs.apply(lambda t: p.n_simp(dsd1(gain, p.p2s_group(t), subs=self.gc * t_)))

        sol_1 = (sol_0[0] + s0, sol_0[1], sol_0[2])
        # Each stored solution carries the highest order it applies to (with χ of order 2: the
        # pump has its leading order -1 and its first depletion correction s0 at 1, the next
        # one, not computed, at 3; the weak fields have their leading order 0 and their first
        # correction, not computed, at 2)
        self._sol = [tuple(w.with_order(o) for w, o in zip(sols, orders))
                     for sols, orders in zip((sol_m1, sol_0, sol_1), self._SOL_ORDERS)]

    # highest applicable orders of (a_0, a_1, a_2) in the stored solutions (undepleted pump
    # without and with the weak fields, first pump-depletion correction); the solution orders
    # (lowest of each row) are -1, 0, 1
    _SOL_ORDERS = ((S.Zero, S.NegativeOne, S.NegativeOne),
                   (S.Zero, S.One, S.One),
                   (S(2), S.One, S.One))

    @property
    def orders(self) -> tuple:
        """
        Orders of the stored solutions, (-1, 0, 1): the order of a solution is the lowest
        accuracy among its modes.
        """
        return tuple(min(w.order for w in sol) for sol in self._sol)

    def get_sol(self, order):
        """
        The stored solution of the highest order not exceeding `order` (see `orders`), labeled
        with the highest applicable orders of its modes. Above order 1, the solution of order 1
        (pump accurate up to order 2, weak fields up to order 1) is returned.
        """
        for sol, o in zip(reversed(self._sol), reversed(self.orders)):
            if o <= order:
                return sol
        raise ValueError(f"Order {order} out of range for FWMSolver (the lowest solution order is "
                         f"{self.orders[0]})")

    def sol_orders(self, order) -> tuple:
        return tuple(w.order for w in self.get_sol(order))

    def sol_expr(self, mode: int, order):
        """
        Get the expression of the solution for the given mode and order; from order 0 on, the
        pump carries the self-phase modulation factor exp(-i χ N_0 t).
        """
        expr = self.get_sol(order)[mode].expr
        if mode == 0 and order >= 0:
            return expr * exp(-I * nlc * self.nd * t_)
        return expr

    def _zq(self, obj):
        f = lambda x: x.subs({self.pm: 2 * nlc * self.nd})
        return obj.apply(f) if isinstance(obj, WMWrapper) else f(obj)
