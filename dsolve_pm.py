from sympy.core.singleton import S
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.core.sympify import sympify
from sympy.core.function import _mexpand, expand_mul
from sympy.core.traversal import bottom_up
from sympy.core.exprtools import gcd_terms
from collections import defaultdict

__all__ = ['dsolve_pm1', 'dsolve_pm2']

def TR8h_selective(rv, x, first=True):
    """
    Product-to-sum for hyperbolic functions that include x.
    """

    def f(rv):
        if not (rv.is_Mul or
            rv.is_Pow and rv.base.func in (cosh, sinh) and
            (rv.exp.is_integer or rv.base.is_positive)):
            return rv
        if first:
            n, d = [expand_mul(i) for i in rv.as_numer_denom()]
            newn = TR8h_selective(n, x, first=False)
            newd = TR8h_selective(d, x, first=False)
            if newn != n or newd != d:
                rv = gcd_terms(newn/newd)
                if (rv.is_Mul and rv.args[0].is_Rational and
                    len(rv.args) == 2 and rv.args[1].is_Add):
                    rv = Mul(*rv.as_coeff_Mul())
            return rv
        
        args = {cosh: [], sinh: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (cosh, sinh) and a.args[0].has(x):
                args[type(a)].append(a.args[0])
            elif (a.is_Pow and a.exp.is_Integer and a.exp > 0 and \
                    a.base.func in (cosh, sinh) and a.base.args[0].has(x)):
                args[type(a.base)].extend([a.base.args[0]] * a.exp)
            else:
                args[None].append(a)
        c = args[cosh]
        s = args[sinh]
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv
        args = args[None]
        n = min(len(c), len(s))
        for _ in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sinh(a1 + a2) + sinh(a1 - a2))/2)
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cosh(a1 + a2) + cosh(a1 - a2))/2)
        if c:
            args.append(cosh(c.pop()))
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((cosh(a1 + a2) - cosh(a1 - a2))/2)
        if s:
            args.append(sinh(s.pop()))
        return TR8h_selective(expand_mul(Mul(*args)), x)

    return bottom_up(rv, f)

def TR8_selective(rv, x, first=True):
    """Converting products of ``cos`` and/or ``sin`` that include x
    to a sum or difference of ``cos`` and or ``sin`` terms.
    """

    def f(rv):
        if not (
            rv.is_Mul or
            rv.is_Pow and
            rv.base.func in (cos, sin) and
            (rv.exp.is_integer or rv.base.is_positive)):
            return rv

        if first:
            n, d = [expand_mul(i) for i in rv.as_numer_denom()]
            newn = TR8_selective(n, x, first=False)
            newd = TR8_selective(d, x, first=False)
            if newn != n or newd != d:
                rv = gcd_terms(newn/newd)
                if rv.is_Mul and rv.args[0].is_Rational and \
                        len(rv.args) == 2 and rv.args[1].is_Add:
                    rv = Mul(*rv.as_coeff_Mul())
            return rv

        args = {cos: [], sin: [], None: []}
        for a in Mul.make_args(rv):
            if a.func in (cos, sin) and a.args[0].has(x):
                args[type(a)].append(a.args[0])
            elif (a.is_Pow and a.exp.is_Integer and a.exp > 0 and \
                    a.base.func in (cos, sin) and a.base.args[0].has(x)):
                # XXX this is ok but pathological expression could be handled
                # more efficiently as in TRmorrie
                args[type(a.base)].extend([a.base.args[0]]*a.exp)
            else:
                args[None].append(a)
        c = args[cos]
        s = args[sin]
        if not (c and s or len(c) > 1 or len(s) > 1):
            return rv

        args = args[None]
        n = min(len(c), len(s))
        for _ in range(n):
            a1 = s.pop()
            a2 = c.pop()
            args.append((sin(a1 + a2) + sin(a1 - a2))/2)
        while len(c) > 1:
            a1 = c.pop()
            a2 = c.pop()
            args.append((cos(a1 + a2) + cos(a1 - a2))/2)
        if c:
            args.append(cos(c.pop()))
        while len(s) > 1:
            a1 = s.pop()
            a2 = s.pop()
            args.append((-cos(a1 + a2) + cos(a1 - a2))/2)
        if s:
            args.append(sin(s.pop()))
        return TR8_selective(expand_mul(Mul(*args)), x)

    return bottom_up(rv, f)

# A light function that collapses multiple exponents
def _ps(expr):
    if expr.is_Pow and expr.base.is_Pow:
        return _ps(expr.base.base**(expr.exp*expr.base.exp))
    return expr

def group_terms(expr, x, h=True):
    expr = (_mexpand(TR8h_selective(expr, x)) if h
            else _mexpand(TR8_selective(expr, x)))
    basis_funcs = (cosh, sinh) if h else (cos, sin)
    grouped = defaultdict(list)
    for term in Add.make_args(expr):
        func = None
        nk = 0
        deg = 0
        const_factors = []
        for t in Mul.make_args(term):
            if x == t:
                deg += 1
            elif t.is_Pow and t.base == x and t.exp.is_Integer:
                deg += t.exp
            elif t.func in basis_funcs:
                func = t.func
                factors = Mul.make_args(t.args[0])
                if x in factors:
                    nk = Mul(*(_ps(f) for f in factors if f != x))
                else:
                    const_factors.append(t)
            else:
                const_factors.append(t)
        if func is None:
            nk = h
        const_part = Mul(*const_factors)
        grouped[(func, nk, deg)].append(const_part)
    return grouped

def dsd1(x, d, f0=0, subs=None):
    """
    Solve the first-order LDE from dict 
    """
    terms = []
    f0s = []
    if subs is None:
        for (func, nk, deg), coeff in d.items():
            t = Add(*coeff).simplify()
            terms.append(pm_expr(x, func, nk, deg, True) * t)
            if func is not None and (func in (sinh, sin)) ^ (deg & 1):
                f0s.append(-pm_coeffs1(func, nk, deg)[0] * t)
    else:
        for (func, nk, deg), coeff in d.items():
            t = Add(*coeff).simplify()
            terms.append(t * pm_expr(x, func, nk, deg, True).subs({x: subs}))
            if func is not None and (func in (sinh, sin)) ^ (deg & 1):
                f0s.append(-pm_coeffs1(func, nk, deg)[0] * t)
    return Add(Add(f0, *f0s).simplify(), *terms)

def dsolve_pm1(rhs, x, h=True, f0=0, subs=None):
    """
    integral of rhs with the initial value of f0 and the form of

    Σ_(nk, deg) x^deg * func(nk * x)

    where the func can be cos, sin, cosh or sinh.

    h (hyperbolic: True/False) parameter is still included
    for distinguishment of the product-to-sum principles.
    """
    return dsd1(x, group_terms(rhs, x, h), f0, subs)

def dsd2(x, d, h=True, f0=0, df0=0, subs=None):
    """
    Solve the second-order LDE from dict 
    """
    terms = []
    f0s = []
    df0s = []
    if subs is None:
        for (func, nk, deg), coeff in d.items():
            t = Add(*coeff).simplify()
            ff0, fdf0 = v02(func, nk, deg)
            terms.append(pm_expr(x, func, nk, deg) * t)
            if ff0 is not S.Zero:
                f0s.append(-ff0 * t)
            if fdf0 is not S.Zero:
                df0s.append(-fdf0 * t)
    else:
        for (func, nk, deg), coeff in d.items():
            t = Add(*coeff).simplify()
            ff0, fdf0 = v02(func, nk, deg)
            terms.append(t * pm_expr(x, func, nk, deg).subs({x: subs}))
            if ff0 is not S.Zero:
                f0s.append(-ff0 * t)
            if fdf0 is not S.Zero:
                df0s.append(-fdf0 * t)
    if subs is None:
        subs = x
    if h:
        return Add(Add(f0, *f0s).simplify()*cosh(subs),
                   Add(df0, *df0s).simplify()*sinh(subs), *terms)
    return Add(Add(f0, *f0s).simplify()*cos(subs),
               Add(df0, *df0s).simplify()*sin(subs), *terms)

def dsolve_pm2(rhs, x, h=True, f0=0, df0=0, subs=None):
    """
    Special solution  to

    f''(x) ± f(x) = Σ_(nk, deg) x^deg * func(nk * x)

    with initial conditions f(0) = f0, f'(0) = df0

    where func can be cos, sin when the left-hand sign is '+'
    or cosh, sinh when the left-hand sign is '-'
    
    The sign is identified by parameter 'h':
    True for '-' and False for '+'
    """
    return dsd2(x, group_terms(rhs, x, h), h, f0, df0, subs)

_pm1_cache = dict()
def pm_coeffs1(func, nk, deg):
    """
    coeffs of the integral of rhs with the form of

    Σ_(nk, deg) x^deg * func(nk * x)

    where nk != 0 and the func can be cos, sin, cosh or sinh.
    """
    if nk <= 0 or deg < 0:
        raise ValueError('Expect positive nk and nonnegative deg input')
    if func not in (cosh, sinh, cos, sin):
        raise ValueError('Expect function types only from (cosh, sinh, cos, sin)')
    key = (func, nk, deg) if func in (sin, cos) else ('h', nk, deg)
    if key in _pm1_cache:
        return _pm1_cache[key]
    if deg == 0:
        return (-1/nk,) if func is sin else (1/nk,)
    elif func is sin:
        coeffs = list(c*deg/nk for c in pm_coeffs1(cos, nk, deg-1))
        coeffs.append(-1/nk)
    else:
        other = sin if func is cos else func
        coeffs = list(-c*deg/nk for c in pm_coeffs1(other, nk, deg-1))
        coeffs.append(1/nk)
    coeffs = tuple(coeffs)
    _pm1_cache[key] = coeffs
    return coeffs

_pm2_cache = dict()
def pm_coeffs2(func, nk, deg):
    """
    Coefficients of each term of the soluiton to

    f''(x) ± f(x) = x^deg * func(nk * x)

    where func can be cos, sin, cosh, sinh or None

    The sign in the left-hand side is determined by the function:
    sin/cos -> "+", sinh/cosh -> "-"
    If func is None, then nk denotes whether the sign is negative
    """
    nk = sympify(nk)
    deg = sympify(deg)
    if func is not None and nk < 0 or deg < 0:
        raise ValueError('Expect nonnegative nk/deg input')
    key = (func, nk, deg) if func in (sin, cos) else ('h', nk, deg)
    if key in _pm2_cache:
        return _pm2_cache[key]
    if deg == 0:
        if func is None:
            coeffs = (S.NegativeOne,) if nk else (S.One,)
        elif nk == 1:
            coeffs = (-S.Half,) if func is sin else (S.Half,)
        else:
            coeffs = (1/(nk**2-1),) if func in (sinh, cosh) else (1/(1-nk**2),)
    elif func is cos:
        if nk == 1:
            coeffs = list(-deg*c/2 for c in pm_coeffs2(sin, 1, deg-1))
            coeffs.append(S.Half/(deg+1))
        else:
            ci = 1-nk**2
            coeffs = list(2*nk*deg*c/ci for c in pm_coeffs2(sin, nk, deg-1))
            coeffs.append(1/ci)
            if deg > 1:
                for i, c in enumerate(pm_coeffs2(cos, nk, deg-2)):
                    coeffs[i] -= deg*(deg-1)*c/ci
        coeffs = tuple(coeffs)
    elif func is sin:
        if nk == 1:
            coeffs = list(deg*c/2 for c in pm_coeffs2(cos, 1, deg-1))
            coeffs.append(-S.Half/(deg+1))
        else:
            ci = 1-nk**2
            coeffs = list(-2*nk*deg*c/ci for c in pm_coeffs2(cos, nk, deg-1))
            coeffs.append(1/ci)
            if deg > 1:
                for i, c in enumerate(pm_coeffs2(sin, nk, deg-2)):
                    coeffs[i] -= deg*(deg-1)*c/ci
        coeffs = tuple(coeffs)
    elif func is None:
        if deg == 1:
            coeffs = (S.Zero, S.NegativeOne) if nk else (S.Zero, S.One)
        else:
            coeffs = []
            s = S.One if nk else S.NegativeOne
            for c in pm_coeffs2(None, nk, deg-2):
                if c is S.Zero:
                    coeffs.append(S.Zero)
                else:
                    coeffs.append(s*c*deg*(deg-1))
            coeffs.extend((S.Zero, -s))
            coeffs = tuple(coeffs)
    else:
        if nk == 1:
            coeffs = list(-deg*c/2 for c in pm_coeffs2(func, 1, deg-1))
            coeffs.append(S.Half/(deg+1))
        else:
            ci = nk**2-1
            coeffs = list(-2*nk*deg*c/ci for c in pm_coeffs2(func, nk, deg-1))
            coeffs.append(1/ci)
            if deg > 1:
                for i, c in enumerate(pm_coeffs2(func, nk, deg-2)):
                    coeffs[i] -= deg*(deg-1)*c/ci
        coeffs = tuple(coeffs)
    _pm2_cache[key] = coeffs
    return coeffs

def pm_expr(x, func, nk, deg, o1=False):
    if func is cos:
        f = (cos(nk*x), sin(nk*x))
    elif func is sin:
        f = (sin(nk*x), cos(nk*x))
    elif func is cosh:
        f = (cosh(nk*x), sinh(nk*x))
    elif func is sinh:
        f = (sinh(nk*x), cosh(nk*x))
    elif o1:
        return x**(deg+1)/(deg+1)
    else:
        terms = (x**i for i in range(deg+1))
        return Add(*(c*t for c, t in zip(pm_coeffs2(func, nk, deg), terms)))
    if o1:
        terms = (x**i * f[(i&1)^(not(deg&1))] for i in range(deg+1))
    else:
        r = range(1, deg+2) if nk == 1 else range(deg+1)
        terms = (x**i * f[(i&1)^(deg&1)] for i in r)
    coeffs = pm_coeffs1(func, nk, deg) if o1 else pm_coeffs2(func, nk, deg)
    return Add(*(c*t for c, t in zip(coeffs, terms)))

def v02(func, nk, deg):
    coeffs = pm_coeffs2(func, nk, deg)
    if func is None:
        c1 = coeffs[1] if deg > 0 else S.Zero
        return coeffs[0], c1
    if nk == 1:
        if (func in (sinh, sin)) ^ (deg & 1):
            return S.Zero, coeffs[0]
        return S.Zero, S.Zero
    if (func in (sinh, sin)) ^ (deg & 1):
        c1 = nk * coeffs[0]
        if deg > 0:
            c1 += coeffs[1]
        return S.Zero, c1
    return coeffs[0], S.Zero
