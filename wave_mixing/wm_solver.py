from sympy.core.singleton import S
from sympy.core.numbers import I
from sympy.core.symbol import Symbol
from sympy.core.mul import Mul
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum.dagger import Dagger
from .pattern_form.pattern_form import PF, PF0, pf_add
from .dsolve_pm import dsd1, dsd2
from .multimode import *
from .wm_processor import WMProcessor, a_p, a_s, a_i, t_, nlc, _g
from functools import cache

__all__ = ['pm', 'TWMSolver', 'FWMSolver']

pm = Symbol('Δk', real=True)

class TWMSolver(WMProcessor):
    """
    Three-wave mixing quantum multimode parametric process perturbation solver

    Ordering principle: The strong field and commutative parameters
    are extracted to the end, ordered in pat-form;
    the multimode patterns are left-multiplied to multimode number operators;
    the annihilation operator of single weak field with larger index number
    is left-multipied to the rest of the expression (multi-mode part),
    and the other operators are ordered as such that the commutation
    between singlemode and multimode operators is avoided.

    The above means that: a0, a1, a2† are right/left-multiplied to the expression
    in the down-conversion/phase transfer case,
    while their conjugates are left/right-multiplied.

    Under this ordering principle, we take the time differential equation
    in the following form:

    d/dt a0 = -i χ a1 a2 exp(-i Δk t),
    d/dt a1 = -i χ a0 a2† exp(i Δk t),
    d/dt a2 = -i χ a1† a0 exp(i Δk t)

    :param dc: whether the process is down-conversion (True)
    or state transfer (False); a2 is strong in the state transfer case
    :param mismatch: Phase mismatch of the process, assumed same for all modes,
    k0 - k1 - k2
    """

    def __init__(self, dc=True, mismatch=False):
        super().__init__(dc=dc, mismatch=mismatch)
        self.half_p_factor = exp(I * self.pm * t_ * S.Half)
        if dc:
            # Mark multiplication directions:
            # a_s/a_i right-multiplied (b_left=False),
            # a_s†/a_i† left-multiplied (b_left=True)
            self.sol = [({(None,): PF({((a_p,), ()): S.One})},
                         f_dict({a_s: {(None,): PF0}, Dagger(a_i): {(None,): PF0}}, b_left=False),
                         f_dict({Dagger(a_s): {(None,): PF0}, a_i: {(None,): PF0}}))]
            self.gc = S.Half*(4*nlc**2*self.nd-self.pm**2)**S.Half
            self.pe.update({a_p.name: S.NegativeOne})
        else:
            self.sol = [(f_dict({a_p: {(None,): PF0}, a_s: {(None,): PF0}}),
                         f_dict({a_p: {(None,): PF0}, a_s: {(None,): PF0}}),
                         {(None,): PF({((a_i,), ()): S.One})})]
            self.gc = S.Half*(4*nlc**2*self.nd+self.pm**2)**S.Half
            self.pe.update({a_i.name: S.NegativeOne})
        self.pm = pm if mismatch else S.Zero
        # d/dt a_drive / (-i χ) for the last solved order
        self.dd_last = {(None,): PF0}
        # Homogeneous solutions to the time differential equations
        @cache
        def _fdf0(n0, mode):
            if n0:
                if mode == 0:
                    return PF({((), ()): dsd2(self.gc*t_, dict(), False,
                                              S.One, I*self.pm*S.Half/self.gc)})
                return PF({((), ()): dsd2(self.gc*t_, dict(), self.dc,
                                          S.One, -I*self.pm*S.Half/self.gc)})
            elif self.dc:
                return PF({((a_p,), ()): dsd2(self.gc*t_, dict(), True,
                                                 S.Zero, -I*nlc/self.gc)})
            elif mode == 0:
                return PF({((a_i,), ()): dsd2(self.gc*t_, dict(), False,
                                                 S.Zero, -I*nlc/self.gc)})
            else:
                return PF({((Dagger(a_i),),()): dsd2(self.gc*t_, dict(), False,
                                                        S.Zero, -I*nlc/self.gc)})
        self.fdf0 = _fdf0

    def get_sol(self, order: int):
        if order < -1:
            raise ValueError('The order should be at least -1')
        i = order + 1
        if len(self.sol) > i:
            return self.sol[i]
        last_sol = self.get_sol(order-1)
        dc = self.dc
        gc_m2 = self.gc**(-2)
        if i & 1:
            def _d2ts(mode, n0, b):
                s1 = S.NegativeOne if mode == 0 else S.One
                s3 = S.NegativeOne if dc else S.One
                for k in term1[b].keys() | term2[b].keys() | last_sol[mode][b].keys():
                    pfs = []
                    if k in term1[b]:
                        pfs.append(s1 * term1[b][k])
                    if k in term2[b]:
                        pfs.append(-term2[b][k])
                    if k in last_sol[mode][b]:
                        pfs.append(last_sol[mode][b][k] * (s3 * self.nd))
                    rhs = self.expand(pf_add(*pfs) * Mul(nlc**2, gc_m2), order, True)
                    s0 = rhs.n_apply(lambda t: self.n_simp(dsd2(_g, self.p2s_group(t), self.dc,
                                                                subs=self.gc*t_)))
                    self.sol[i][mode][b][k] = (pf_add(s0, self.fdf0(n0, mode))
                                               if len(k) == 1 else s0)
            if dc:
                self.sol.append((last_sol[0],
                                 f_dict({a_s: dict(), Dagger(a_i): dict()}, b_left=False),
                                 f_dict({Dagger(a_s): dict(), a_i: dict()})))
                pd = self.d_adjoint(last_sol[0])
                term1 = self.e_mul(self.e_mul(last_sol[0], pd), last_sol[1])
                term2 = self.e_mul(self.dd_last, self.d_adjoint(last_sol[2]))
                _d2ts(1, True, a_s)
                _d2ts(1, False, Dagger(a_i))
                term1 = self.e_mul(last_sol[2], self.e_mul(pd, last_sol[0]))
                term2 = self.e_mul(self.d_adjoint(last_sol[1]), self.dd_last)
                _d2ts(2, False, Dagger(a_s))
                _d2ts(2, True, a_i)
            else:
                self.sol.append((f_dict({a_p: dict(), a_s: dict()}),
                                 f_dict({a_p: dict(), a_s: dict()}),
                                 last_sol[2]))
                id = self.d_adjoint(last_sol[2])
                term1 = self.e_mul(last_sol[0], self.e_mul(id, last_sol[2]))
                term2 = self.e_mul(last_sol[1], self.dd_last)
                _d2ts(0, True, a_p)
                _d2ts(0, False, a_s)
                term1 = self.e_mul(last_sol[0], self.d_adjoint(self.dd_last))
                term2 = self.e_mul(last_sol[1], self.e_mul(last_sol[2], id))
                _d2ts(1, True, a_s)
                _d2ts(1, False, a_p)
        else:
            def _d1ts():
                b = self.sol[0][0][(None,)] if dc else self.sol[0][2][(None,)]
                result = dict()
                for k, v in self.dd_last.items():
                    rhs = self.expand(v * (-I * nlc / self.gc), order, True)
                    s0 = rhs.n_apply(lambda t: self.n_simp(dsd1(_g, self.p2s_group(t),
                                                                subs=self.gc*t_))) 
                    result[k] = pf_add(s0, b) if len(k) == 1 else s0
                return result
            if dc:
                self.dd_last = self.e_mul(last_sol[1], last_sol[2])
                self.sol.append((_d1ts(), *last_sol[1:]))
            else:
                self.dd_last = self.e_mul(self.d_adjoint(last_sol[1]), last_sol[0])
                self.sol.append((*last_sol[:2], _d1ts()))
        return self.sol[i]

    def sol_expr(self, mode: int, order: int):
        """
        Get the expression of the solution for the given mode and order.

        The order is consistent with the order of the perturbative expansion,
        so the classical solution with zero input is considered as order -1,
        and the parametirc amplifier/oscillator solution with undepleted drive
        is considered as order 0.
        """
        s0 = self.sol[0]
        if len(s0[mode]) == 1:
            p = S.One
        elif mode == 0:
            p = self.half_p_factor**S.NegativeOne
        else:
            p = self.half_p_factor
        if len(s0[mode]) == 1 or mode != 1 or len(s0[0]) == 2:  # driving field
            return p*self.to_expr(self.get_sol(order)[mode])
        return p*self.to_expr(self.get_sol(order)[mode])


class FWMSolver(WMProcessor):
    """
    Four-wave mixing quantum multimode parametric process perturbation solver.

    The driving pump field a_0 is strong and treated parametrically.
    Solutions are obtained up to the first correction to the pump field (order 1),
    including self-phase modulation and cross-phase modulation pump depletion.

    Because the solution is capped at the first pump correction, results are
    computed and stored directly upon initialization rather than solved recursively.
    """

    def __init__(self):
        super().__init__(dc=True, mismatch=True)
        self.a = (MultimodeCOP(a_s.name, a_i.name, annih1=0),
                  MultimodeCOP(a_s.name, a_i.name))
        self.hpm = -I*self.pm*S.Half + I*nlc*self.nd
        self.gc = S.Half*(self.pm * (-self.pm + 4*nlc*self.nd))**S.Half
        self.pe.update({a_p.name: -S.Half})
        self._solve()

    def _solve(self):
        # Order -1: Initial operators at t=0
        sol_m1 = (
            {(None,): PF({((a_p,), ()): S.One})},
            f_dict({a_s: {(None,): PF0}, Dagger(a_i): {(None,): PF0}}, b_left=False),
            f_dict({Dagger(a_s): {(None,): PF0}, a_i: {(None,): PF0}})
        )

        # Order 0: Undepleted pump and parametric weak field evolution
        fwm0_0 = {(None,): PF({((a_p,), ()): S.One})}
        fwm0_1 = f_dict({
            a_s: {(None,): PF({((), ()): dsd2(self.gc*t_, dict(),
                                              True, S.One, self.hpm/self.gc)})},
            Dagger(a_i): {(None,): PF({((a_p**2,), ()): dsd2(self.gc*t_, dict(),
                                                             True, S.Zero, -I*nlc/self.gc)})}
        }, b_left=False)
        fwm0_2 = f_dict({
            Dagger(a_s): {(None,): PF({((a_p**2,), ()): dsd2(self.gc*t_, dict(),
                                                             True, S.Zero, -I*nlc/self.gc)})},
            a_i: {(None,): PF({((), ()): dsd2(self.gc*t_, dict(),
                                              True, S.One, self.hpm/self.gc)})}
        }, b_left=True)
        sol_0 = (fwm0_0, fwm0_1, fwm0_2)

        # Order 1: First correction to pump (pump depletion)
        xpm_c = {(None,): PF({((a_p,), ()): S.Half*MultimodeNum2(a_s.name, a_i.name)})}
        coeff = {(None,): PF({((a_p,), ()): S.Half})}
        xpm_t1 = self.e_mul(sol_0[1], self.d_adjoint(sol_0[1]))
        xpm_t2 = self.e_mul(self.d_adjoint(sol_0[2]), sol_0[2])
        xpm_t = self.d_add(xpm_t1, xpm_t2)
        dd2 = self.e_mul(self.e_mul(sol_0[1], self.d_adjoint(sol_0[0])), sol_0[2])
        fwm1p = dict()
        for k, v in self.d_add(dd2, xpm_c, self.e_mul(coeff, xpm_t)).items():
            rhs = self.pe.expand_pf(v.n_apply(lambda t:
                                              -2*I*nlc*t.subs({t_: _g/self.gc})/self.gc), 1)
            s0 = rhs.n_apply(lambda t: self.n_simp(dsd1(_g, self.p2s_group(t), subs=self.gc*t_)))
            fwm1p[k] = pf_add(sol_0[0][(None,)], s0) if len(k) == 1 else s0

        sol_1 = (fwm1p, sol_0[1], sol_0[2])
        self.sol = [sol_m1, sol_0, sol_1]

    def get_sol(self, order: int):
        """
        Get the solution tuple for the given order (-1, 0, or 1).
        """
        if order < -1 or order > 1:
            raise ValueError(f"Order {order} out of range for"
                             " FWMSolver (supports orders -1, 0, 1)")
        return self.sol[order + 1]

    def sol_expr(self, mode: int, order: int):
        """
        Get the expression of the solution for the given mode and order (up to order 1).
        For the pump mode (mode 0), includes the SPM phase factor
        exp(-i * χ * N_0 * t) for order >= 0.
        """
        sol_dict = self.get_sol(order)[mode]
        expr = self.to_expr(sol_dict)
        if mode == 0 and order >= 0:
            return expr * exp(-I * nlc * self.nd * t_)
        return expr
    
    def _zq(self, obj):
        """
        Assumes that the phase mismatch exactly compensates
        the self-phase modulation of the strong field,
        and evaluates the expression accordingly.

        :param obj: Expr, PF, dict, or f_dict.
        """
        return d_apply(obj, lambda x: x.subs({self.pm: 2*nlc*self.nd}))
