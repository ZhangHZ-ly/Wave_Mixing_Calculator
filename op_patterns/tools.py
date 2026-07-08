from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.core.symbol import Symbol
from sympy.core.expr import Expr
from sympy.core.function import Function, Subs, Derivative
from sympy.core.singleton import S
from sympy.core.mul import Mul
from .fermion_pattern import FermionPat, fp_separable
from .boson_pattern import BosonNum, BosonPat, NumShift

___all___=['extract_modes', 'check_num', 'in_boson_num',
           'mul_pattern', 'redo_mul']

def extract_modes(expr, _visited=None):
    """
    Extracts basic modes and variables included in the expression
    Returns a tuple with 5 elements, respectively for:
    boson modes, fermion modes, non_commutative symbols, commutative symbols
    and integral variables / summation indices / replaced symbols
    """
    from sympy.core.symbol import Symbol
    from sympy.core.containers import Tuple
    if _visited is None:
        _visited = set()
    if id(expr) in _visited:
        return None
    _visited.add(id(expr))

    modes = tuple(set() for _ in range(5))

    if isinstance(expr, (BosonOp, BosonNum)):
        modes[0].add(BosonOp(expr.name))
    elif isinstance(expr, FermionPat):
        modes[1].add(FermionOp(expr.name))
    elif isinstance(expr, Symbol):
        if expr.is_commutative:
            modes[3].add(expr)
        else:
            modes[2].add(expr)
    elif isinstance(expr, Subs):
        m = extract_modes(expr.expr, _visited)
        if m is not None:
            for s1, s2 in zip(modes, m):
                s1.update(s2)
        modes[4].update(expr.variables)
        for arg in expr.point:
            m = extract_modes(arg, _visited)
            if m is not None:
                for s1, s2 in zip(modes, m):
                    s1.update(s2)
    elif isinstance(expr, Derivative):
        m = extract_modes(expr.expr, _visited)
        if m is not None:
            for s1, s2 in zip(modes, m):
                s1.update(s2)
    elif isinstance(expr, Tuple):
        modes[4].add(expr[0])
        for arg in expr.args[1:]:
            m = extract_modes(arg, _visited)
            if m is not None:
                for s1, s2 in zip(modes, m):
                    s1.update(s2)
    else:
        for arg in expr.args:
            m = extract_modes(arg, _visited)
            if m is not None:
                for s1, s2 in zip(modes, m):
                    s1.update(s2)

    return modes

def check_num(expr: Expr, cache=None) -> bool:
    """
    Checks whether the expression is represented only in boson number operators
    """
    if expr.is_commutative or getattr(expr, 'boson_num_only', False):
        return True
    if isinstance(expr, (BosonPat, FermionPat, Symbol)):
        return False
    if isinstance(expr, NumShift):
        return check_num(expr.function, cache)
    if cache and expr in cache:
        return cache[expr]
    if hasattr(expr, 'args'):
        result= all(check_num(arg, cache) for arg in expr.args)
        if cache is not None:
            cache[expr] = result
        return result
    return False

def in_boson_num(expr: Expr, mode, prune=False):
    """
    The first output is the same as check_num, while the second
    represents whether the expression commutes with the BosonNum
    in the given mode (name) assuming non-independent modes;
    the exception is when 'prune' is set True, which can provide None
    output for check-num result if the second output is determined to be False

    If multiple modes of BosonNums appears in the same Function
    or Pow with non-integer exponent, any of the modes in the included
    expression is considered as commute with each other

    This design is to avoid scanning the same expression for more than once.
    """
    if expr.is_commutative:
        return True, True
    if isinstance(expr, BosonNum):
        return True, expr.name == mode
    if isinstance(expr, (BosonPat, FermionPat, Symbol)):
        return False, False
    if isinstance(expr, NumShift):
        return in_boson_num(expr.function, mode)
    if expr.is_Pow and not expr.exp.is_integer or isinstance(expr, Function):
        ibn = False
        for arg in expr.args:
            n, ib = in_boson_num(arg, mode)
            if not n:
                return False, False
            if ib:
                ibn = True
        return True, ibn
    if hasattr(expr, 'args'):
        ibn = False
        for arg in expr.args:
            n, ib = in_boson_num(arg, mode)
            if not n:
                return False, False
            if not ib:
                if prune:
                    return None, False
                ibn = False
        return True, ibn
    return False, False

def mul_pattern(e1, e2):
    if isinstance(e1, FermionPat) and isinstance(e2, FermionPat) and e1.name == e2.name:
        if fp_separable(e1, e2):
            if (e1.commutation_symmetry is not None
                or all(n.is_integer and n.is_positive for n in e1.exps)):
                if e1.is_hermitian or not (len(e1.exps) & 1):
                    return e2,
                else:
                    return S.Zero,
            else:
                exp_cut = list(e2.exps[(1-len(e1.exps)):])
                exp_cut[0] -= e1.exps[-1]
                if exp_cut[0] == 0:
                    del exp_cut[0]
                if fp_separable(e1, FermionPat(e2.name,
                                e2.left_base.is_annihilation ^
                                (len(e2.exps) - len(exp_cut)),
                                exp_cut, eval_exp=False)):
                    return e2,
        return FermionPat(e1, e2.exps, eval_exp=False,
                          merge_ends=(e1.right_base==e2.left_base)),
    
    elif isinstance(e1, BosonPat) and isinstance(e2, BosonPat) and e1.name == e2.name:
        return BosonPat(e1, e2.exps, eval_exp=False,
                          merge_ends=(e1.right_base==e2.left_base)),

    elif isinstance(e2, BosonPat) and in_boson_num(e1, e2.name)[1]:
        return e2, NumShift(e1, 1, e2).doit()

    return None

def redo_mul(expr, prune_zero=True, num_repr=False):
    """
    Redo the multimplcation in the expression to apply self-defined __mul__ rules

    :param prune_zero: If set True, the branch will be pruned once "0" appears
    :param combine_numop: If set True, the algorithm will rewrite a† a as N_(a),
    or a a† as 1±N_(a) for any quantum operator a
    """
    def _redo_mul_node(node: Mul):
        new_args = []
        a = node.args[-1]
        rep = False
        def mp(arg):
            nonlocal a, rep
            match mul_pattern(arg, a):
                case None:
                    new_args.append(a)
                    a = arg
                case(x,):
                    rep = True
                    a = x
                case (x, y):
                    rep = True
                    new_args.append(y)
                    a = x
        if num_repr:
            def update(arg):
                mp(arg)
                nonlocal a, rep
                if isinstance(a, BosonPat):
                    match a.num_repr():
                        case (_, S.One):
                            pass
                        case (S.One, x):
                            rep = True
                            a = x
                        case (x, y):
                            rep = True
                            new_args.append(y)
                            a = x
        else:
            update = mp
        for arg in reversed(node.args[:-1]):
            update(arg)

            if prune_zero and a == 0:
                return S.Zero
        new_args.append(a)
        if rep:
            new_args.reverse()
            return Mul(*new_args)
        else:
            return node

    from sympy.core.traversal import bottom_up

    return bottom_up(expr, lambda node: _redo_mul_node(node)
                        if isinstance(node, Mul) else node)
