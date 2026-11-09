from sympy.core.singleton import S
from sympy.core.function import DefinedFunction
from sympy.functions.elementary.trigonometric import sin, cos
from sympy.functions.elementary.hyperbolic import tanh, cosh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.elliptic_integrals import elliptic_k
from sympy.core.symbol import Dummy
from sympy.integrals.integrals import Integral
from sympy.simplify.simplify import simplify
import mpmath

K = lambda m: elliptic_k(m)  # Complete elliptic integral of the first kind

# ============ sn(u|m) ============
class jacobi_sn(DefinedFunction):
    nargs = 2

    @classmethod
    def eval(cls, u, m):
        # Special values
        if u == 0:
            return S.Zero
        if m == 0:
            return sin(u)
        if m == 1:
            return tanh(u)

        n = simplify(u / K(m))
        if n.is_integer:
            n_int = simplify(n)
            if n_int % 2 == 0:
                return S.Zero
            else:
                return S.One if (n_int % 4 == 1) else -S.One
        return None

    def fdiff(self, argindex=1):
        u, m = self.args
        if argindex == 1:
            return jacobi_cn(u, m) * jacobi_dn(u, m)
        elif argindex == 2:
            t = Dummy('t')
            integrand = jacobi_sn(t, m) * jacobi_cn(t, m) / jacobi_dn(t, m)
            return -Integral(integrand, (t, 0, u)) / (2*m)
        raise ValueError("jacobi_sn(u,m) has two arguments")

    def evalf(self, prec=None, **kwargs):
        u, m = self.args
        u_val = complex(u.evalf()) if not u.is_number else complex(u)
        m_val = complex(m.evalf()) if not m.is_number else complex(m)
        sn, _, _ = mpmath.ellipfun(m_val)
        val = sn(u_val)
        return S(val).evalf(prec)

# ============ cn(u|m) ============
class jacobi_cn(DefinedFunction):
    nargs = 2

    @classmethod
    def eval(cls, u, m):
        if u == 0:
            return S.One
        if m == 0:
            return cos(u)
        if m == 1:
            return 1 / cosh(u)

        n = simplify(u / K(m))
        if n.is_integer:
            n_int = simplify(n)
            mod4 = n_int % 4
            if mod4 == 0:
                return S.One
            elif mod4 == 1 or mod4 == 3:
                return S.Zero
            elif mod4 == 2:
                return -S.One
        return None

    def fdiff(self, argindex=1):
        u, m = self.args
        if argindex == 1:
            return -jacobi_sn(u, m) * jacobi_dn(u, m)
        elif argindex == 2:
            t = Dummy('t')
            integrand = jacobi_sn(t, m)**2 / jacobi_cn(t, m)
            return Integral(integrand, (t, 0, u)) / (2*m)
        raise ValueError("jacobi_cn(u,m) has two arguments")

    def evalf(self, prec=None, **kwargs):
        u, m = self.args
        u_val = complex(u.evalf()) if not u.is_number else complex(u)
        m_val = complex(m.evalf()) if not m.is_number else complex(m)
        _, cn, _ = mpmath.ellipfun(m_val)
        val = cn(u_val)
        return S(val).evalf(prec)

# ============ dn(u|m) ============
class jacobi_dn(DefinedFunction):
    nargs = 2

    @classmethod
    def eval(cls, u, m):
        if u == 0:
            return S.One
        if m == 0:
            return S.One
        if m == 1:
            return 1 / cosh(u)

        n = simplify(u / K(m))
        if n.is_integer:
            n_int = simplify(n)
            return S.One if n_int % 2 == 0 else sqrt(1 - m)
        return None

    def fdiff(self, argindex=1):
        u, m = self.args
        if argindex == 1:
            return -m * jacobi_sn(u, m) * jacobi_cn(u, m)
        elif argindex == 2:
            t = Dummy('t')
            integrand = jacobi_sn(t, m)**2 / jacobi_dn(t, m)
            return -Integral(integrand, (t, 0, u)) / 2
        raise ValueError("jacobi_dn(u,m) has two arguments")

    def evalf(self, prec=None, **kwargs):
        u, m = self.args
        u_val = complex(u.evalf()) if not u.is_number else complex(u)
        m_val = complex(m.evalf()) if not m.is_number else complex(m)
        _, _, dn = mpmath.ellipfun(m_val)
        val = dn(u_val)
        return S(val).evalf(prec)
