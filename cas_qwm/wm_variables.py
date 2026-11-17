from sympy.core.symbol import Symbol, Dummy
from sympy.physics.quantum.boson import BosonOp

__all__ = ['a_p', 'a_s', 'a_i', 't_', 'nlc', 'pm', 'nm',
           'alpha_0', 'alpha_2', 'beta_0', 'beta_1', 'beta_2', 'betas', 'gain', 'g_hat']

# Field names: a0 - pump (high frequency), a1, a2 - signal and idler (low frequency)
a_p = BosonOp('a_0')
a_s = BosonOp('a_1')
a_i = BosonOp('a_2')
# Time / nonlinear coefficient / phase mismatch symbols: t, χ, Δk
t_ = Dummy('t', real=True)
nlc = Symbol('χ', real=True, nonnegative=True)
pm = Symbol('Δk', real=True)
# Number of mode pairs of a multimode process
nm = Symbol('n_m', nonnegative=True)

# Coherent amplitudes of the input states: alpha_j of the strong driving field (a_0 in
# down-conversion and four-wave mixing, a_2 in state transfer), beta_j of a coherently seeded
# weak mode j (indexed by the mode: betas[j] = beta_j)
alpha_0 = Symbol('alpha_0')
alpha_2 = Symbol('alpha_2')
beta_0 = Symbol('beta_0')
beta_1 = Symbol('beta_1')
beta_2 = Symbol('beta_2')
betas = (beta_0, beta_1, beta_2)

# Parametric gain: the scalar g (coherent driving field) and the operator \hat{g}
# (non-commutative to preserve operator ordering)
gain = Symbol('g', real=True)
g_hat = Symbol(r'\hat{g}', commutative=False)
