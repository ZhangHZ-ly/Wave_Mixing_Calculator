from sympy.core.symbol import Symbol, Dummy
from sympy.physics.quantum.boson import BosonOp

__all__ = ['a_p', 'a_s', 'a_i', 't_', 'nlc']

# Field names: a0 - pump (high frequency), a1, a2 - signal and idler (low frequency)
a_p = BosonOp('a_0')
a_s = BosonOp('a_1')
a_i = BosonOp('a_2')
# Time / nonlinear coefficient symbols: t, χ
t_ = Dummy('t', real=True)
nlc = Symbol('χ', real=True, nonnegative=True)
