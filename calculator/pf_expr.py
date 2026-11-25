from __future__ import annotations
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from cas_qwm.op_pattern import BosonNum, BosonPat, FermionPat
from cas_qwm.pattern_perturb import PF, pf
from sympy.core.function import _mexpand, diff
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex

from .common import (
    DAGGER_ALIASES,
    ELEMENTARY_FUNCS,
    TRANSFORMS,
    safe_subs,
    order_by_refs,
    symbol_assumptions,
    validated_name,
)


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


def save_pf_variable(def_id: str, pf_val: Any, rem: Any = None) -> PF:
    p = _to_pf(pf_val)
    if rem is not None:
        p = p.with_rem(rem)
    PF_VARIABLES[def_id] = p
    return p


def save_latest_pf_variable(def_id: str) -> Optional[PF]:
    if not LAST_EVAL_PF:
        return None
    return save_pf_variable(def_id, LAST_EVAL_PF['pf'], rem=LAST_EVAL_PF['rem'])


def eval_pattern_form(
    text: str,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    use_rem: bool = False,
    expand: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    Evaluates quantum expressions using PF add/mul/power arithmetic.
    If expand is True, the coefficients (functions of number operators), the non-PF part
    and the remainder are expanded with _mexpand (multiplication and power expansion only).
    Returns: (latex_result, remainder_latex); remainder_latex is None without use_rem
    """
    text = (text or '').strip()
    if not text:
        raise ValueError("Expression input cannot be empty.")

    def _out(val):
        """Values are kept as raw sympy objects in _rem mode, and as PF otherwise."""
        return val if use_rem else PF.from_val(val)

    def _bind(p: PF):
        """Value bound to a named expression in the parser namespace."""
        return (p.expr() + getattr(p, 'rem', S.Zero)) if use_rem else p

    mode_kind_map = {op.get('name', '').strip(): op.get('kind', 'boson') for op in operators if op.get('name')}

    def _base_op(item):
        """The single canonical operator (or its adjoint) represented by item."""
        if isinstance(item, PF):
            if item.table:
                bs, fs = next(iter(item.table.keys()))
                if bs or fs:
                    return bs[0] if bs else fs[0]
            return item.expr()
        return item

    def _mode_name(op) -> str:
        """Identifier (mode name) of an operator; display names are applied only at output."""
        inner = op.args[0] if isinstance(op, Dagger) else op
        return str(getattr(inner, 'name', inner))

    def _is_fermion(op, name: str) -> bool:
        inner = op.args[0] if isinstance(op, Dagger) else op
        return isinstance(inner, (FermionOp, FermionPat)) or mode_kind_map.get(name) == 'fermion'

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, PF):
                if item.npf == S.Zero and set(item.table.keys()).issubset({((), ())}):
                    arg = item.table.get(((), ()), S.Zero)
                    return PF({((), ()): sympify(fn(arg))}, rem=item.rem if use_rem else S.Zero)
                return _to_pf(fn(item.expr()), use_rem)
            return _out(fn(sympify(item)))
        return _inner

    def _raw_arg(x):
        return x if isinstance(x, bool) else (x.expr() if isinstance(x, PF) else sympify(x))

    def _unwrap_args(args):
        raw_args = []
        for a in args:
            raw_args.extend(map(_raw_arg, a) if isinstance(a, (tuple, list)) else (_raw_arg(a),))
        return raw_args

    def _make_boson_num(item):
        return _out(BosonNum(_mode_name(_base_op(item))))

    def _make_op_pat(pat_cls, item, *args):
        return _out(pat_cls(_mode_name(_base_op(item)), *_unwrap_args(args)))

    def _make_pat(item, *args):
        op = _base_op(item)
        is_ann = False if isinstance(op, Dagger) else bool(getattr(op, 'is_annihilation', True))
        name = _mode_name(op)

        exps = _unwrap_args(args)
        if exps and isinstance(exps[0], bool):
            is_ann = exps.pop(0)
        if not exps:
            exps = [1]

        if _is_fermion(op, name):
            return _out(FermionPat(name, is_ann, *exps))

        bp = BosonPat(name, is_ann, *exps)
        if use_rem:
            return bp
        a_op, tail = bp.num_repr()
        b_key = (a_op,) if a_op is not S.One else ()
        return PF({(b_key, ()): tail}, rem=S.Zero)

    if use_rem:
        _dag_fn = lambda item: Dagger(item)
        _pow_fn = lambda b, e: b ** e
    else:
        _dag_fn = lambda item: PF.from_val(item).adjoint()
        _pow_fn = lambda b, e: (b if isinstance(b, PF) else PF.from_val(b)) ** e

    _diff_fn = lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs)

    local_dict: Dict[str, Any] = {
        'i': I,
        'I': I,
        'pi': pi,
        'e': exp(1),
        **{k: _wrap_fn(fn) for k, fn in ELEMENTARY_FUNCS.items()},
        **dict.fromkeys(DAGGER_ALIASES, _dag_fn),
        **dict.fromkeys(('N_', 'BosonNum'), _make_boson_num),
        **dict.fromkeys(('Pat', 'pattern'), _make_pat),
        **dict.fromkeys(('BP', 'BosonPat'), partial(_make_op_pat, BosonPat)),
        **dict.fromkeys(('FP', 'FermionPat'), partial(_make_op_pat, FermionPat)),
        **dict.fromkeys(('Pow', 'pow'), _pow_fn),
        **dict.fromkeys(('diff', 'Derivative'), _diff_fn),
    }

    disp_subs = {}

    # Register symbols as scalar PF (or Expr when use_rem=True)
    for s in symbols:
        s_name = validated_name(s, 'symbol')
        sym_obj = Symbol(s_name, **symbol_assumptions(s))
        local_dict[s_name] = _out(sym_obj)

        d_name = (s.get('display_name') or '').strip()
        if d_name and d_name != s_name:
            disp_subs[sym_obj] = Symbol(d_name)

    # Register field operators
    for op in operators:
        op_name = validated_name(op, 'operator')
        op_obj = BosonOp(op_name) if op.get('kind', 'boson') == 'boson' else FermionOp(op_name)
        local_dict[op_name] = _out(op_obj)

        d_name = (op.get('display_name') or '').strip()
        if d_name and d_name != op_name:
            disp_subs[Symbol(op_name)] = Symbol(d_name)

    # Saved expressions: the stored object of each definition, or its source input re-evaluated
    # with the definitions it was saved with; the table names are bound to their current definitions
    base_dict = dict(local_dict)
    built: Dict[str, PF] = {}
    for ne in order_by_refs(named_expressions):
        ne_name = validated_name(ne, 'expression') if ne.get('name') else None
        p_ne = PF_VARIABLES.get(ne['id'])
        if p_ne is None:
            ne_str = (ne.get('source_input') or '').strip()
            if not ne_str:
                continue
            scope = {**base_dict, **{n: _bind(built[r]) for n, r in ne.get('refs', {}).items() if r in built}}
            try:
                p_ne = _to_pf(parse_expr(ne_str, local_dict=scope, transformations=TRANSFORMS), use_rem)
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name or ne_str}': {e}")
            save_pf_variable(ne['id'], p_ne, rem=p_ne.rem)
        built[ne['id']] = p_ne
        if ne_name:
            local_dict[ne_name] = _bind(p_ne)

    try:
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS)
        pf_result = _to_pf(parsed_target, use_rem)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    remainder_expr = pf_result.rem if use_rem else S.Zero
    if expand:
        pf_result = pf_result.apply(_mexpand, on_npf=True)
        remainder_expr = _mexpand(remainder_expr)
        pf_result = pf_result.with_rem(remainder_expr)

    # Apply display name substitutions to LaTeX output
    latex_str = latex(safe_subs(pf_result.expr(), disp_subs))

    rem_latex = None
    if use_rem:
        rem_latex = latex(safe_subs(remainder_expr, disp_subs))

    LAST_EVAL_PF.clear()
    LAST_EVAL_PF.update({
        'pf': pf_result,
        'rem': remainder_expr,
    })

    return latex_str, rem_latex
