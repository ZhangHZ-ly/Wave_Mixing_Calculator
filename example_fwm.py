# Use the methods for TWM to calculate the FWM parametric solution and pump depletion
from op_patterns import BosonNum
from perturb_eval import PE, PF
from parametric import TWMSolver
from sympy.core.singleton import S
from parametric import (I, Dagger, exp, t_, nlc, a_i, a_p, a_s, pm, _g,
                        MultimodeNum as MMN, MultimodeNum2 as MMN2, MultimodeCOP as MMC)
from dsolve_pm import dsd1, dsd2

solver_pm = TWMSolver()

# The constrution of these lines is similar to TWMSolver.__init__
n_p = BosonNum(a_p.name)
hpm = -I*pm*S.Half + I*nlc*n_p  # Half phase mismatch
gc = S.Half*(pm * (pm + 4*nlc*n_p))**S.Half
pe = PE({a_p.name: -S.Half, nlc: S.One})
pe.register(MMN)
pe.register(MMN2)
pe.register(MMC)
fwm0 = ({(None,): PF({((a_p,), ()): S.One})},
        {a_s: {(None,): PF({((), ()): dsd2(gc*t_, dict(), True, S.One, hpm/gc)})},
         Dagger(a_i): {(None,): PF({((a_p**2,), ()): dsd2(gc*t_, dict(), True, S.Zero,
                                                          -I*nlc/gc)})}},
        {Dagger(a_s): {(None,): PF({((a_p**2,), ()): dsd2(gc*t_, dict(), True, S.Zero,
                                                          -I*nlc/gc)})},
         a_i: {(None,): PF({((), ()): dsd2(gc*t_, dict(), True, S.One, hpm/gc)})}})

# The following lines are similar to the "_d1ts" branch of TWMSolver.get_sol
prod_10 = {a_s: dict(), Dagger(a_i): dict()}
d0 = solver_pm.d_adjoint(fwm0[0])
for b, e in fwm0[1].items():
    for a, d in solver_pm.bdswap(b, d0, False).items():
        solver_pm.d_mul(prod_10[a], e, d)
xpm_c = {(None,): PF({((a_p,), ()): S.Half*MMN2(a_s.name, a_i.name)})}
coeff = {(None,): PF({((a_p,), ()): S.Half})}
xpm_t1 = solver_pm.e_mul(fwm0[1], solver_pm.d_adjoint(fwm0[1]))
xpm_t2 = solver_pm.e_mul(solver_pm.d_adjoint(fwm0[2]), fwm0[2])
xpm_t = solver_pm.d_add(xpm_t1, xpm_t2)
dd2 = solver_pm.e_mul(prod_10, fwm0[2])
fwm1p = dict()
for k, v in solver_pm.d_add(dd2, xpm_c, solver_pm.e_mul(coeff, xpm_t)).items():
    rhs = pe.expand_pf(v.n_apply(lambda t: -2*I*nlc*t.subs({t_: _g/gc})/gc), 1)
    s0 = rhs.n_apply(lambda t: solver_pm.n_simp(dsd1(_g, solver_pm.p2s_group(t), subs=gc*t_)))
    fwm1p[k] = PF.add(fwm0[0][(None,)], s0) if len(k) == 1 else s0

# FWM pump output with depletion
fwm_1_0 = solver_pm.to_expr(fwm1p)*exp(-I * nlc * n_p * t_)
