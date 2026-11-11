from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from cas_qwm.op_pattern import BosonNum, BosonPat, FermionPat
from cas_qwm.pattern_perturb import PF, pf
from sympy.core.function import diff
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex

from .common import TRANSFORMS, is_valid_variable_name


PF_VARIABLES: Dict[str, PF] = {}
LAST_EVAL_PF: Dict[str, Any] = {}


def _to_pf(val: Any, use_rem: bool = False) -> PF:
    if isinstance(val, PF):
        return val
    if isinstance(val, tuple) and len(val) == 2 and isinstance(val[0], PF):
        return PF.from_val(val)
    if not use_rem:
        return PF.from_val(val)
    expr = sympify(val)
    if getattr(expr, 'is_commutative', True) or not expr.has(BosonOp, BosonPat, FermionOp, FermionPat):
        return PF({((), ()): expr}, rem=S.Zero)
    return PF.from_val(pf(expr, as_dict=None, rem=True))


def save_pf_variable(name: str, pf_val: Any, rem: Any = None, use_rem: bool = False, **kwargs) -> PF:
    val_rem = rem if rem is not None else S.Zero
    p = _to_pf(pf_val, use_rem=use_rem)
    if rem is not None or not hasattr(p, 'rem'):
        p.rem = sympify(val_rem)
    PF_VARIABLES[name] = p
    return p


def save_latest_pf_variable(name: str, **kwargs) -> Optional[PF]:
    if not LAST_EVAL_PF:
        return None
    return save_pf_variable(
        name,
        LAST_EVAL_PF['pf'],
        rem=LAST_EVAL_PF.get('rem', S.Zero),
        **kwargs
    )


def get_pf_variable(name: str) -> Optional[PF]:
    return PF_VARIABLES.get(name)


def clear_pf_variables() -> None:
    PF_VARIABLES.clear()
    LAST_EVAL_PF.clear()


def eval_pf_form(
    text: str,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    use_rem: bool = False
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Evaluates quantum expressions using PF add/mul/power arithmetic.
    Returns: (latex_result, expr_str, remainder_latex, remainder_str)
    """
    text = (text or '').strip()
    if not text:
        raise ValueError("Expression input cannot be empty.")

    def _make_boson_num(item):
        if isinstance(item, PF):
            name = str(item.expr())
        elif hasattr(item, 'name'):
            name = str(item.name)
        else:
            name = str(item)
        bn = BosonNum(name)
        return bn if use_rem else PF.from_val(bn)

    mode_kind_map = {op.get('name', '').strip(): op.get('kind', 'boson') for op in operators if op.get('name')}
    op_mode_map = {
        op.get('name', '').strip(): (op.get('display_name') or op.get('displayName') or '').strip()
        for op in operators if op.get('name')
    }

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, PF):
                if item.npf == S.Zero and set(item.table.keys()).issubset({((), ())}):
                    arg = item.table.get(((), ()), S.Zero)
                    return PF({((), ()): sympify(fn(arg))}, rem=item.rem if use_rem else S.Zero)
                return _to_pf(fn(item.expr()), use_rem)
            res = fn(sympify(item))
            return res if use_rem else PF.from_val(res)
        return _inner

    def _unwrap_args(args):
        raw_args = []
        for a in args:
            if isinstance(a, (tuple, list)):
                raw_args.extend(x if isinstance(x, bool) else (x.expr() if isinstance(x, PF) else sympify(x)) for x in a)
            else:
                raw_args.append(a if isinstance(a, bool) else (a.expr() if isinstance(a, PF) else sympify(a)))
        return raw_args

    def _make_op_pat(pat_cls, item, *args):
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        pat = pat_cls(name, *_unwrap_args(args))
        return pat if use_rem else PF.from_val(pat)

    def _make_pat(item, *args):
        if isinstance(item, PF) and item.table:
            bs, fs = next(iter(item.table.keys()))
            op = bs[0] if bs else fs[0]
        else:
            op = item

        from sympy.physics.quantum.dagger import Dagger
        if isinstance(op, Dagger):
            inner = op.args[0]
            is_ann = False
        else:
            inner = op
            is_ann = op.is_annihilation

        name = inner.name
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])

        exps = _unwrap_args(args)
        if exps and isinstance(exps[0], bool):
            is_ann = exps.pop(0)
        if not exps:
            exps = [1]

        is_fermion = (mode_kind_map.get(name) == 'fermion') if mode_kind_map else isinstance(inner, (FermionOp, FermionPat))
        if is_fermion:
            fp = FermionPat(name, is_ann, *exps)
            return fp if use_rem else PF.from_val(fp)

        bp = BosonPat(name, is_ann, *exps)
        if use_rem:
            return bp
        a_op, tail = bp.num_repr()
        b_key = (a_op,) if a_op is not S.One else ()
        return PF({(b_key, ()): tail}, rem=S.Zero)

    from sympy.physics.quantum.dagger import Dagger as _Dagger
    _dag_fn = ((lambda item: _Dagger(item)) if use_rem
               else (lambda item: PF.from_val(item).adjoint()))
    _pow_fn = ((lambda b, e: b ** e) if use_rem
               else (lambda b, e: (b ** e) if isinstance(b, PF) else ((PF.from_val(b)) ** e)))

    local_dict: Dict[str, Any] = {
        'i': I,
        'I': I,
        'pi': pi,
        'e': exp(1),
        'exp': _wrap_fn(exp),
        'sin': _wrap_fn(sin),
        'cos': _wrap_fn(cos),
        'sinh': _wrap_fn(sinh),
        'cosh': _wrap_fn(cosh),
        'tanh': _wrap_fn(tanh),
        'sqrt': _wrap_fn(sqrt),
        'dag': _dag_fn,
        'Dagger': _dag_fn,
        'dagger': _dag_fn,
        'N_': _make_boson_num,
        'BosonNum': _make_boson_num,
        'Pat': _make_pat,
        'pattern': _make_pat,
        'BP': lambda item, *args: _make_op_pat(BosonPat, item, *args),
        'BosonPat': lambda item, *args: _make_op_pat(BosonPat, item, *args),
        'FP': lambda item, *args: _make_op_pat(FermionPat, item, *args),
        'FermionPat': lambda item, *args: _make_op_pat(FermionPat, item, *args),
        'Pow': _pow_fn,
        'pow': _pow_fn,
        'diff': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'Derivative': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
    }

    disp_subs = {}

    # Register symbols as scalar PF (or Expr when use_rem=True)
    for s in symbols:
        s_name = s.get('name', '').strip()
        valid, err_msg = is_valid_variable_name(s_name)
        if not valid:
            raise ValueError(f"Invalid symbol identifier: {err_msg}")
        kwargs = {}
        if s.get('is_positive'): kwargs['positive'] = True
        elif s.get('is_nonnegative'): kwargs['nonnegative'] = True
        elif s.get('is_integer'): kwargs['integer'] = True
        elif s.get('is_real'): kwargs['real'] = True
        sym_obj = Symbol(s_name, **kwargs)
        local_dict[s_name] = sym_obj if use_rem else PF.from_val(sym_obj)

        d_name = (s.get('display_name') or '').strip()
        if d_name and d_name != s_name:
            disp_subs[sym_obj] = Symbol(d_name)

    # Register field operators
    for op in operators:
        op_name = op.get('name', '').strip()
        valid, err_msg = is_valid_variable_name(op_name)
        if not valid:
            raise ValueError(f"Invalid operator identifier: {err_msg}")
        kind = op.get('kind', 'boson')
        op_obj = BosonOp(op_name) if kind == 'boson' else FermionOp(op_name)
        local_dict[op_name] = op_obj if use_rem else PF.from_val(op_obj)

        d_name = (op.get('display_name') or '').strip()
        if d_name and d_name != op_name:
            disp_subs[Symbol(op_name)] = Symbol(d_name)

    # Register saved named expressions directly from PF
    for ne in named_expressions:
        ne_name = ne.get('name', '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")

        if 'val' in ne and ne['val'] is not None:
            val_obj = ne['val']
            p_ne = _to_pf(val_obj, use_rem)
            if 'rem' in ne and ne['rem'] is not None:
                p_ne.rem = sympify(ne['rem'])
            local_dict[ne_name] = (p_ne.expr() + p_ne.rem) if use_rem else p_ne
            save_pf_variable(ne_name, p_ne, rem=p_ne.rem)
        elif ne_name in PF_VARIABLES:
            p_ne = PF_VARIABLES[ne_name]
            local_dict[ne_name] = (p_ne.expr() + getattr(p_ne, 'rem', S.Zero)) if use_rem else p_ne
        else:
            ne_str = (ne.get('expr_str') or ne.get('source_input') or ne.get('sourceInput') or '').strip()
            if not ne_str:
                continue
            try:
                parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
                p_ne = _to_pf(parsed_ne, use_rem)
                local_dict[ne_name] = (p_ne.expr() + getattr(p_ne, 'rem', S.Zero)) if use_rem else p_ne
                save_pf_variable(ne_name, p_ne, rem=p_ne.rem)
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    try:
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS)
        pf_result = _to_pf(parsed_target, use_rem)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    expr_result = pf_result.expr()

    # Apply display name substitutions to LaTeX output
    expr_disp = expr_result
    if disp_subs:
        try:
            expr_disp = expr_result.subs(disp_subs)
        except Exception:
            expr_disp = expr_result

    latex_str = latex(expr_disp)
    expr_str = str(expr_disp)

    remainder_expr = pf_result.rem if use_rem else S.Zero
    rem_latex = None
    rem_str = None
    if use_rem:
        rem_disp = remainder_expr
        if disp_subs:
            try:
                rem_disp = remainder_expr.subs(disp_subs)
            except Exception:
                rem_disp = remainder_expr
        rem_latex = latex(rem_disp)
        rem_str = str(rem_disp)

    global LAST_EVAL_PF
    LAST_EVAL_PF = {
        'pf': pf_result,
        'rem': remainder_expr,
        'latex_str': latex_str,
        'remainder_latex': rem_latex,
    }

    return latex_str, expr_str, rem_latex, rem_str
