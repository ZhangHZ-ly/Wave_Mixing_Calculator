"""
Jacobi elliptic functions and the elliptic-polynomial algebra built on them.

- jacobi: SymPy functions jacobi_sn, jacobi_cn, jacobi_dn, jacobi_epsilon with closed-form
  derivatives in both arguments and mpmath evaluation.
- dsolve_lame: the ring Q(m)[n, u, E, sn, cn, dn, j], its derivatives, closed-form integration in u
  and the Lamé solver.
- elliptic_algebra: λ-series of that ring as functions of number operators, the PF coefficient
  EllipticTail and the PE coefficient algebra EllipticAlgebra.
twm_dual_stimulated applies them to three-wave mixing with two strong inputs.
"""
from .jacobi import jacobi_sn, jacobi_cn, jacobi_dn, jacobi_epsilon
from . import dsolve_lame, elliptic_algebra
from .elliptic_algebra import EllipticTail, EllipticAlgebra

__all__ = ['jacobi_sn', 'jacobi_cn', 'jacobi_dn', 'jacobi_epsilon', 'dsolve_lame', 'elliptic_algebra',
           'EllipticTail', 'EllipticAlgebra']
