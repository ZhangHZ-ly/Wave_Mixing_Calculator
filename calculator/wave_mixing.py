from __future__ import annotations
import ast
import io
import math
import re
import tokenize
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from cas_qwm.op_pattern import BosonNum, BosonPat, FermionPat
from cas_qwm.pattern_perturb.pattern_form import _vpk
from cas_qwm.parametric_multimode import *
from cas_qwm.parametric_multimode.wm_processor import t_, _g, nm, nlc, _tdz
from cas_qwm.parametric_multimode.wm_solver import pm
from cas_qwm.pattern_perturb import PF, PF1
from sympy.core.expr import Expr
from sympy.core.function import diff
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
    RESERVED_NAMES,
    WM_TRANSFORMS,
    safe_subs,
    symbol_assumptions,
    validated_name,
)
from .conditions import normalize_process, process_is_dc
from .wm_transforms import t_to_g, apply_assertions
from .pf_expr import PF_VARIABLES

# Cached TWMSolver instances indexed by [dc][mismatch]
TWM_SOLVERS = (
    (TWMSolver(False, False), TWMSolver(False, True)),
    (TWMSolver(True, False), TWMSolver(True, True)),
)
FWM_SOLVER = FWMSolver()


def get_wm_solver(process: Optional[str] = None, mismatch: bool = False, dc: Optional[bool] = None) -> WMSolver:
    proc = normalize_process(process, dc)
    if proc == 'fwm':
        return FWM_SOLVER
    return TWM_SOLVERS[1 if proc == 'twm_dc' else 0][1 if mismatch else 0]


def _solver_process(s: WMSolver) -> str:
    if isinstance(s, FWMSolver):
        return 'fwm'
    return 'twm_dc' if s.dc else 'twm_fc'


def eval_wm_solve(
    dc: bool = True,
    mismatch: bool = False,
    var_index: int = 0,
    order: int = 1,
    process: Optional[str] = None,
    **kwargs,
) -> Tuple[str, str]:
    """
    Evaluates Wave Mixing Heisenberg evolution for Tab 3.
    Returns: (latex_t, latex_g)
    """
    proc = normalize_process(process, dc)
    s = get_wm_solver(proc, mismatch=bool(mismatch))

    if proc == 'fwm':
        expr_t = s.sol_expr(var_index, min(1, max(-1, order)))
        if not mismatch:
            expr_t = s._zq(expr_t)
    else:
        expr_t = s.sol_expr(var_index, max(-1, order - 1))
    expr_g = t_to_g(expr_t, process=proc, mismatch=bool(mismatch))

    return latex(expr_t), latex(expr_g)


WM_VARIABLES: Dict[str, WMWrapper] = {}
LAST_EVAL_WM: Dict[str, Any] = {}


def save_wm_variable(
    name: str,
    val_t: Any,
    val_g: Any = None,
    solver: Any = None,
    latex_t: Optional[str] = None,
    latex_g: Optional[str] = None,
    **kwargs
) -> WMWrapper:
    proc = normalize_process(kwargs.get('process'), kwargs.get('dc'))
    s = solver if isinstance(solver, WMSolver) else get_wm_solver(proc, mismatch=bool(kwargs.get('mismatch', False)))
    w = WMWrapper.from_val(val_t, s.processor)
    w = WMWrapper(w.processor, dict(w.dict_repr), w.b_left)
    w.name = name
    w.val_t = w
    w.val_g = val_g if val_g is not None else t_to_g(
        w,
        process=kwargs.get('process', _solver_process(s)),
        mismatch=bool(kwargs.get('mismatch', False)),
    )
    w.latex_t = latex_t
    w.latex_g = latex_g
    for k, v in kwargs.items():
        if k != 'dc':
            setattr(w, k, v)
    WM_VARIABLES[name] = w
    return w


def save_latest_wm_variable(name: str, **kwargs) -> Optional[WMWrapper]:
    if not LAST_EVAL_WM:
        return None
    proc = normalize_process(LAST_EVAL_WM.get('process'), LAST_EVAL_WM.get('dc', True))
    s = LAST_EVAL_WM.get('solver') or get_wm_solver(proc, mismatch=bool(LAST_EVAL_WM.get('mismatch', False)))
    return save_wm_variable(
        name,
        LAST_EVAL_WM['val_t'],
        val_g=LAST_EVAL_WM.get('val_g'),
        solver=s,
        latex_t=LAST_EVAL_WM.get('latex_t'),
        latex_g=LAST_EVAL_WM.get('latex_g'),
        order=LAST_EVAL_WM.get('order'),
        leading_order=LAST_EVAL_WM.get('leading_order'),
        dc=LAST_EVAL_WM.get('dc'),
        process=proc,
        mismatch=LAST_EVAL_WM.get('mismatch', False),
        single_mode=LAST_EVAL_WM.get('single_mode'),
        **kwargs
    )


def get_wm_variable(name: str) -> Optional[WMWrapper]:
    return WM_VARIABLES.get(name)


def clear_wm_variables() -> None:
    WM_VARIABLES.clear()
    LAST_EVAL_WM.clear()


def _format_order_num(val) -> Union[int, float]:
    f = float(val)
    return int(f) if f.is_integer() else f


def compute_val_leading_order(
    val: Any,
    s: WMSolver,
    dc: Optional[bool] = None,
    process: Optional[str] = None,
) -> Union[int, float]:
    """
    Computes the leading perturbation order of a calculated WMWrapper result `val`.
    """
    proc = normalize_process(process if process is not None else _solver_process(s), dc)
    pe = s.pe

    def _sub_t_for_order(expr_t: Expr) -> Expr:
        if not expr_t.has(t_):
            return expr_t
        gc_eff = (nlc * s.nd) if (proc == 'fwm' and not expr_t.has(pm)) else s.gc
        return expr_t.subs({t_: _g / gc_eff})

    w_sub = WMWrapper.from_val(val, s.processor).apply(_sub_t_for_order, on_npf=True)
    if w_sub.b_left is None:
        groups = [(0, w_sub.dict_repr.values())]
    else:
        groups = [(pe.leading_order(b)[1], d.values()) for b, d in w_sub.dict_repr.items()]

    orders: List[Any] = []
    for b_ord, pfs in groups:
        for p in pfs:
            try:
                o = pe.lo_pf(p)[1]
                if o is not S.Infinity:
                    orders.append(o + b_ord)
            except Exception:
                pass

    if not orders:
        return 0
    try:
        return _format_order_num(min(orders))
    except Exception:
        return 0


_NLC_NAMES = ('nlc', 'chi', r'\chi', 'χ')


def _ref_attr(ref: Any, key: str, default: Any = None) -> Any:
    return ref.get(key, default) if isinstance(ref, dict) else getattr(ref, key, default)


def analyze_expression_terms(
    expr_str: str,
    target_order: int,
    dc: bool = True,
    symbols: Optional[List[Dict[str, Any]]] = None,
    operators: Optional[List[Dict[str, Any]]] = None,
    named_expressions: Optional[List[Dict[str, Any]]] = None,
    process: Optional[str] = None,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Analyzes an expression across each additive term to determine required
    evolved operator order and identify any saved named expressions with
    insufficient perturbation order.
    """
    proc = normalize_process(process, dc)
    is_dc = process_is_dc(proc)
    strong_lo: float = -0.5 if proc == 'fwm' else -1.0
    strong_nd_lo: float = 2.0 * strong_lo

    strong_mode_idx = 0 if is_dc else 2
    strong_names = {'a_p', 'a_0'} if is_dc else {'a_i', 'a_2'}
    evolved_names = {'a_p', 'a_s', 'a_i'}

    if operators:
        for op in operators:
            name = (op.get('name') or '').strip()
            mode_idx = int(op.get('mode_index', 0))
            if mode_idx == strong_mode_idx and name:
                strong_names.add(name)
            if op.get('type') == 'evolved_op' and name:
                evolved_names.add(name)

    named_map = {
        (ne.get('name') or '').strip(): ne
        for ne in (named_expressions or ())
        if (ne.get('name') or '').strip()
    }

    sym_orders: Dict[str, float] = {}
    if symbols:
        for s_def in symbols:
            name = (s_def.get('name') or '').strip()
            if name:
                if name in _NLC_NAMES or s_def.get('builtin_key') == 'nlc':
                    sym_orders[name] = 1.0
                else:
                    try:
                        sym_orders[name] = float(s_def.get('order', 0))
                    except (ValueError, TypeError):
                        sym_orders[name] = 0.0

    def _terms_lo(expr_node, mult: float) -> Tuple[bool, float, str, Any]:
        inner_terms = expand_additive_products(expr_node)
        min_lo = 0.0
        has_evolved = False
        evolved_ref = None
        for t in inner_terms:
            t_lo = sum(get_factor_info(fn)[1] * sm for fn, sm in t)
            min_lo = min(min_lo, t_lo)
            for fn, _ in t:
                _, _, cat, ref = get_factor_info(fn)
                if cat == 'evolved':
                    has_evolved = True
                    evolved_ref = ref
        lo = min_lo * mult
        return (False, lo, 'evolved', evolved_ref) if has_evolved else (True, lo, 'exact', None)

    def get_factor_info(node) -> Tuple[bool, float, str, Any]:
        if isinstance(node, ast.Constant):
            return True, 0.0, 'exact', None
        if isinstance(node, ast.Name):
            name = node.id
            if name in named_map:
                ne = named_map[name]
                raw_lo = ne.get('leading_order', ne.get('leadingOrder', 0))
                return False, float(raw_lo or 0), 'named', ne
            if name in WM_VARIABLES:
                cvar = WM_VARIABLES[name]
                lo = float(getattr(cvar, 'leading_order', None) or 0)
                return False, lo, 'named', cvar
            if name in evolved_names:
                lo = strong_lo if name in strong_names else 0.0
                return False, lo, 'evolved', name
            if name in sym_orders:
                return True, sym_orders[name], 'exact', name
            if name in _NLC_NAMES:
                return True, 1.0, 'exact', name
            lo = strong_lo if name in strong_names else 0.0
            return True, lo, 'exact', name
        if isinstance(node, ast.UnaryOp):
            return get_factor_info(node.operand)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ''
            if func_name in (*DAGGER_ALIASES, 'diff', 'Derivative') and node.args:
                return get_factor_info(node.args[0])
            if func_name in ('N_', 'BosonNum') and node.args:
                arg = node.args[0]
                arg_name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else (arg.id if isinstance(arg, ast.Name) else '')
                lo = strong_nd_lo if arg_name in strong_names else 0.0
                return True, lo, 'exact', None
            if func_name == 'sqrt' and node.args:
                return _terms_lo(node.args[0], 0.5)
            return True, 0.0, 'exact', None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp_val = 1.0
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float)):
                exp_val = float(node.right.value)
            elif isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub) and isinstance(node.right.operand, ast.Constant):
                exp_val = -float(node.right.operand.value)
            elif isinstance(node.right, ast.BinOp) and isinstance(node.right.op, ast.Div):
                if isinstance(node.right.left, ast.Constant) and isinstance(node.right.right, ast.Constant):
                    exp_val = float(node.right.left.value) / float(node.right.right.value)

            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, (ast.Mult, ast.MatMult)):
                return _terms_lo(node.left, exp_val)
            base_exact, base_lo, cat, ref = get_factor_info(node.left)
            return base_exact, base_lo * exp_val, cat, ref
        return True, 0.0, 'exact', None

    def expand_additive_products(node) -> List[List[Tuple[Any, int]]]:
        if isinstance(node, ast.Expression):
            return expand_additive_products(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return expand_additive_products(node.left) + expand_additive_products(node.right)
        if isinstance(node, ast.UnaryOp):
            return expand_additive_products(node.operand)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ''
            if func_name in DAGGER_ALIASES and len(node.args) == 1:
                inner_terms = expand_additive_products(node.args[0])
                if len(inner_terms) > 1 or (len(inner_terms) == 1 and len(inner_terms[0]) > 1):
                    return inner_terms
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.MatMult)):
            left_terms = expand_additive_products(node.left)
            right_terms = expand_additive_products(node.right)
            if len(left_terms) * len(right_terms) <= 64:
                return [lt + rt for lt in left_terms for rt in right_terms]
            return [[(node.left, 1), (node.right, 1)]]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            left_terms = expand_additive_products(node.left)
            right_terms = expand_additive_products(node.right)
            if len(right_terms) == 1 and len(left_terms) <= 64:
                neg_right = [(n, -s) for n, s in right_terms[0]]
                return [lt + neg_right for lt in left_terms]
            return [lt + [(node.right, -1)] for lt in left_terms]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and 1 < node.right.value <= 16:
                k = int(node.right.value)
                base_terms = expand_additive_products(node.left)
                if len(base_terms) == 1:
                    if len(base_terms[0]) > 1:
                        return [base_terms[0] * k]
                    is_ex, _, _, _ = get_factor_info(node.left)
                    if not is_ex:
                        return [[(node.left, 1)] * k]
                elif len(base_terms) ** k <= 64:
                    acc: List[List[Tuple[Any, int]]] = [[]]
                    for _ in range(k):
                        acc = [a + b for a in acc for b in base_terms]
                    return acc
        return [[(node, 1)]]

    try:
        s = expr_str.replace('^', '**')
        s = re.sub(r'\)\s*(?=[A-Za-z0-9_(])', ')*', s)
        s = re.sub(r'(\d)\s*(?=[A-Za-z_(])', r'\1*', s)
        s = re.sub(r'([A-Za-z0-9_])\s+(?=[A-Za-z0-9_])', r'\1*', s)
        tree = ast.parse(s, mode='eval')
        expanded_terms = expand_additive_products(tree)
    except Exception:
        return -1, []

    max_evolved_needed: int = -1
    insufficient_items: List[Dict[str, Any]] = []
    seen_warnings = set()

    for raw_factors in expanded_terms:
        parsed_factors = []
        total_lo_sum = 0.0
        has_estimated = False

        for f_node, sign_mult in raw_factors:
            is_exact, lo, cat, ref = get_factor_info(f_node)
            eff_lo = lo * sign_mult
            total_lo_sum += eff_lo
            if not is_exact:
                has_estimated = True
            parsed_factors.append((is_exact, eff_lo, cat, ref))

        if not has_estimated:
            continue

        for is_exact, eff_lo, cat, ref in parsed_factors:
            if is_exact:
                continue
            other_lo_sum = total_lo_sum - eff_lo
            req_order_raw = float(target_order) - other_lo_sum
            req_order_int = int(math.ceil(req_order_raw - 1e-12))

            warning = None
            if cat == 'evolved':
                if proc == 'fwm':
                    # FWM evolved operators are capped at order 1
                    needed_for_op = max(-1, min(1, req_order_int))
                    if req_order_int > 1:
                        warning = (str(ref), 'evolved_op', 1)
                elif ref in strong_names:
                    needed_for_op = -1 if req_order_int <= 0 else (req_order_int if req_order_int % 2 == 1 else req_order_int - 1)
                else:
                    needed_for_op = -1 if req_order_int < 0 else (req_order_int if req_order_int % 2 == 0 else req_order_int - 1)
                max_evolved_needed = max(max_evolved_needed, needed_for_op)
            elif cat == 'named':
                rec_order = _ref_attr(ref, 'order')
                if rec_order is not None and float(rec_order) < req_order_raw - 1e-12:
                    warning = (_ref_attr(ref, 'name', ''), 'named_expr', rec_order)

            if warning is not None:
                name, kind, rec_order = warning
                key = (kind, name, rec_order, req_order_int)
                if key not in seen_warnings:
                    seen_warnings.add(key)
                    insufficient_items.append({
                        'name': name,
                        'kind': kind,
                        'order': rec_order,
                        'required_order': req_order_int,
                    })

    return max_evolved_needed, insufficient_items


def validate_input_identifiers(expr_str: str, local_dict: dict):
    """
    Validates identifiers in expr_str. Variable names concatenated without spaces (e.g. a_sa_i)
    are recognized as a single overall long variable name (containing two or more underscores).
    """
    if not expr_str or not expr_str.strip():
        return
    try:
        toks = list(tokenize.tokenize(io.BytesIO(expr_str.encode('utf-8')).readline))
    except Exception:
        return
    for tok in toks:
        if tok.type == tokenize.NAME:
            name = tok.string
            if name.count('_') >= 2 and name not in local_dict and name not in RESERVED_NAMES:
                raise ValueError(
                    f"Invalid input: undefined compound identifier '{name}'. To multiply, separate the "
                    f"variables with a space / '@' (multimode product) or '*' (single-mode product)."
                )


def get_evolved_sol(s: WMSolver, evolved_order: int) -> tuple:
    """
    Evolved (a_p, a_s, a_i) solution wrappers used for expression evaluation,
    including the self-phase-modulation factor of the FWM pump.
    """
    sol = s.get_sol(evolved_order)
    if isinstance(s, FWMSolver) and evolved_order >= 0:
        spm_w = WMWrapper(s.processor, {(None,): PF({((), ()): exp(-I * nlc * s.nd * t_)})})
        sol = (sol[0] @ spm_w, sol[1], sol[2])
    return sol


@dataclass
class WMEvalContext:
    """Parsed Tab 3 expression together with the evaluation settings it was parsed under."""
    val: WMWrapper
    solver: WMSolver
    sol: tuple
    evolved_order: int
    disp_subs: Dict[Any, Any]
    local_dict: Dict[str, Any]
    proc: str
    is_dc: bool
    mismatch: bool
    single_mode: bool
    symbols: List[Dict[str, Any]] = field(default_factory=list)


def _prepare_wm_evaluation(
    text: str,
    dc: bool,
    order: int,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    mm_operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    mismatch: bool = False,
    single_mode: bool = False,
    process: Optional[str] = None,
):
    """
    Prepares and algebraically evaluates wave-mixing operator combinations in Tab 3.
    Returns a WMEvalContext holding the parsed wrapper and the evaluation settings.
    """
    proc = normalize_process(process, dc)
    is_dc = process_is_dc(proc)
    mismatch = bool(mismatch)
    s = get_wm_solver(proc, mismatch=mismatch)
    p = s.processor

    needed_evolved_order, _ = analyze_expression_terms(
        text,
        target_order=order,
        dc=is_dc,
        symbols=symbols,
        operators=operators,
        named_expressions=named_expressions,
        process=proc,
    )
    evolved_order = max(-1, order, needed_evolved_order)
    if proc == 'fwm':
        evolved_order = min(1, evolved_order)
    p._evolved_order = evolved_order
    sol = get_evolved_sol(s, evolved_order)

    init_ops = p.init_ops

    op_mode_map = {
        'a_0': 'a_0', 'a_p': 'a_0',
        'a_1': 'a_1', 'a_s': 'a_1',
        'a_2': 'a_2', 'a_i': 'a_2',
    }
    mode_kind_map = {k: 'boson' for k in op_mode_map}

    custom_orders = {}
    for item_def in (*symbols, *operators):
        name = (item_def.get('name') or '').strip()
        if name and item_def.get('order') is not None:
            try:
                ord_v = sympify(item_def['order'])
                custom_orders[name] = ord_v
                custom_orders[Symbol(name)] = ord_v
            except Exception:
                pass
    if custom_orders:
        p.pe.update(custom_orders)

    def _resolve_mode_name(item) -> str:
        if isinstance(item, WMWrapper):
            for idx, ew in enumerate(sol):
                if item is ew:
                    return f'a_{idx}'
            for k in ('a_0', 'a_1', 'a_2'):
                if item is init_ops[k]:
                    return k
            if item.b_left is not None and len(item.dict_repr) == 1:
                name = str(next(iter(item.dict_repr)).name)
            else:
                name = str(item.expr)
        else:
            name = str(getattr(item, 'name', item))
        return op_mode_map.get(name, name)

    def _unwrap_scalar(x):
        if isinstance(x, WMWrapper):
            sc = x.as_scalar
            return sc if sc is not None else x.expr
        return sympify(x)

    def _raw_args(args) -> list:
        return [a if isinstance(a, bool) else _unwrap_scalar(a) for a in args]

    def _split_pat_args(args):
        """Split pattern arguments into (explicit leading annihilation flag or None, exponents)."""
        exps = _raw_args(args)
        explicit_ann = exps.pop(0) if (exps and isinstance(exps[0], bool)) else None
        return explicit_ann, (exps or [1])

    def _make_boson_num(item):
        return WMWrapper.from_val(BosonNum(_resolve_mode_name(item)), p)

    def _make_mmn(name, disp=0):
        return WMWrapper.from_val(MultimodeNum(_resolve_mode_name(name), _unwrap_scalar(disp)), p)

    def _make_mmn2(disp=0):
        d_val = _unwrap_scalar(disp)
        val = p.mmn2 + d_val * nm if d_val != 0 else p.mmn2
        return WMWrapper.from_val(val, p)

    def _make_pat(item, *args):
        explicit_ann, exps = _split_pat_args(args)

        # Multimode canonical operator A (or its adjoint)
        is_A = isinstance(item, WMWrapper) and (item is init_ops['A'] or item == init_ops['A'])
        is_dagA = (isinstance(item, WMWrapper) and item.b_left is None
                   and list(item.dict_repr.keys()) == [(False, 1)]
                   and item.dict_repr[(False, 1)].table == {_vpk: S.One})
        if is_A or is_dagA:
            is_ann = is_A if explicit_ann is None else explicit_ann
            int_exps = [int(sympify(e)) for e in exps]
            if any(e < 0 for e in int_exps):
                raise ValueError("Pattern exponents must be non-negative integers.")
            return WMWrapper(p, {_tdz(is_ann, int_exps): PF1})

        # Evolved / multimode wrapped operators: alternate the operator and its adjoint
        if isinstance(item, WMWrapper) and (item.b_left is not None or set(item.dict_repr.keys()) != {(None,)}):
            if item.b_left is not None and len(item.dict_repr) == 1:
                b_first = next(iter(item.dict_repr))
                op_ann = item if b_first.is_annihilation else item.adjoint()
                is_a = bool(b_first.is_annihilation) if explicit_ann is None else bool(explicit_ann)
            else:
                op_ann = item
                is_a = True if explicit_ann is None else bool(explicit_ann)
            res = WMWrapper(p, PF1)
            for e in exps:
                cur = op_ann if is_a else op_ann.adjoint()
                res = res @ (cur ** int(sympify(e)))
                is_a = not is_a
            return res

        # Plain single-mode field operators
        if isinstance(item, WMWrapper):
            pf_item = item.dict_repr.get((None,), PF1)
            if pf_item.table:
                bs, fs = next(iter(pf_item.table.keys()))
                op = bs[0] if bs else fs[0]
            else:
                op = item.expr
        else:
            op = item

        if isinstance(op, Dagger):
            inner, is_ann = op.args[0], False
        else:
            inner, is_ann = op, op.is_annihilation
        if explicit_ann is not None:
            is_ann = explicit_ann

        name = str(inner.name)
        name = op_mode_map.get(name, name)
        is_fermion = (mode_kind_map.get(name) == 'fermion') or isinstance(inner, (FermionOp, FermionPat))
        if is_fermion:
            return WMWrapper.from_val(FermionPat(name, is_ann, *exps), p)

        bp = BosonPat(name, is_ann, *exps)
        a_op, tail = bp.num_repr()
        b_key = (a_op,) if a_op is not S.One else ()
        return WMWrapper(p, PF({(b_key, ()): tail}))

    def _make_op_pat(pat_cls, item, *args):
        return WMWrapper.from_val(pat_cls(_resolve_mode_name(item), *_raw_args(args)), p)

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, WMWrapper):
                sc = item.as_scalar
                if sc is not None:
                    return WMWrapper(p, PF({_vpk: sympify(fn(sc))}))
                if item.b_left is None:
                    return p.expand_func_d(fn, evolved_order, item)
            raw_item = item.expr if isinstance(item, WMWrapper) else item
            return WMWrapper.from_val(fn(sympify(raw_item)), p)
        return _inner

    def _diff_fn(expr, *args, **kwargs):
        if isinstance(expr, WMWrapper):
            return expr.diff(*args, gc=s.gc, **kwargs)
        return diff(expr, *args, **kwargs)

    def wrap(v):
        return WMWrapper.from_val(v, p)

    w_g_internal = WMWrapper(p, PF({_vpk: s.gc * t_}))
    w_g_internal._is_g = True
    w_g_internal._gc = s.gc

    w_t = wrap(t_)
    w_t._is_t = True

    pm_internal = s.pm if proc == 'fwm' else (pm if mismatch else S.Zero)

    local_dict: Dict[str, Any] = {
        'i': wrap(I),
        'I': wrap(I),
        'pi': wrap(pi),
        'e': wrap(exp(1)),
        **{k: _wrap_fn(fn) for k, fn in ELEMENTARY_FUNCS.items()},
        **dict.fromkeys(DAGGER_ALIASES, lambda item: wrap(item).adjoint()),
        **dict.fromkeys(('N_', 'BosonNum'), _make_boson_num),
        **dict.fromkeys(('Pat', 'pattern'), _make_pat),
        **dict.fromkeys(('BP', 'BosonPat'), partial(_make_op_pat, BosonPat)),
        **dict.fromkeys(('FP', 'FermionPat'), partial(_make_op_pat, FermionPat)),
        'MMN': _make_mmn,
        'MMN2': _make_mmn2,
        't': w_t,
        'g': w_g_internal,
        **dict.fromkeys(('diff', 'Derivative'), _diff_fn),
        'nlc': wrap(nlc),
        'pm': wrap(pm_internal),
        'n_m': wrap(S.One if single_mode else nm),
        **{k: init_ops[k] for k in ('a_0', 'a_1', 'a_2', 'A', 'N_t')},
        'a_p': sol[0],
        'a_s': sol[1],
        'a_i': sol[2],
    }

    disp_subs: Dict[Any, Any] = {}

    for sym_def in symbols:
        s_name = validated_name(sym_def, 'symbol')
        d_name = (sym_def.get('display_name') or '').strip()
        builtin_key = sym_def.get('builtin_key')

        if s_name == 'nlc' or builtin_key == 'nlc':
            local_dict[s_name] = wrap(nlc)
            if d_name and d_name not in ('nlc', r'\chi', 'χ'):
                disp_subs[nlc] = Symbol(d_name)
        elif s_name == 'pm' or builtin_key == 'pm':
            if not mismatch:
                local_dict[s_name] = wrap(pm_internal)
            else:
                local_dict[s_name] = wrap(pm)
                if d_name and d_name not in ('pm', 'Δk', r'\Delta k'):
                    disp_subs[pm] = Symbol(d_name)
        else:
            sym_obj = Symbol(s_name, **symbol_assumptions(sym_def))
            local_dict[s_name] = wrap(sym_obj)
            if d_name and d_name != s_name:
                disp_subs[sym_obj] = Symbol(d_name)

    for op_def in operators:
        op_name = validated_name(op_def, 'operator')
        d_name = (op_def.get('display_name') or '').strip()

        if op_def.get('type') == 'operator' or ('kind' in op_def and 'mode_index' not in op_def):
            kind = op_def.get('kind', 'boson')
            mode_kind_map[op_name] = kind
            op_mode_map[op_name] = op_name
            local_dict[op_name] = wrap(BosonOp(op_name) if kind == 'boson' else FermionOp(op_name))
            if d_name and d_name != op_name:
                disp_subs[Symbol(op_name)] = Symbol(d_name)
        else:
            mode_idx = int(op_def.get('mode_index', 0))
            if 0 <= mode_idx <= 2:
                default_mode_name = f'a_{mode_idx}'
                local_dict[op_name] = sol[mode_idx]
                op_mode_map[op_name] = default_mode_name
                mode_kind_map[op_name] = 'boson'
                mode_kind_map[default_mode_name] = 'boson'
                if d_name and d_name != default_mode_name:
                    disp_subs[Symbol(default_mode_name)] = Symbol(d_name)

    for mm_def in mm_operators:
        mm_name = validated_name(mm_def, 'multimode operator')
        local_dict[mm_name] = init_ops['A'] if mm_def.get('mm_kind', 'COP') == 'COP' else init_ops['N_t']

    def _bind_named(ne_name: str, ne: Dict[str, Any], w_ne: WMWrapper, save: bool) -> None:
        if single_mode:
            w_ne = w_ne.to_single_mode()
        local_dict[ne_name] = w_ne
        if save:
            save_wm_variable(
                ne_name,
                w_ne,
                solver=s,
                latex_t=ne.get('latex_t'),
                latex_g=ne.get('latex_g'),
                order=ne.get('order'),
                leading_order=ne.get('leading_order'),
                dc=is_dc,
                process=proc,
                mismatch=mismatch,
                single_mode=single_mode
            )

    for ne in named_expressions:
        ne_name = validated_name(ne, 'expression')

        if ne.get('val') is not None:
            _bind_named(ne_name, ne, wrap(ne['val']), save=True)
        elif ne_name in WM_VARIABLES:
            _bind_named(ne_name, ne, WM_VARIABLES[ne_name], save=False)
        elif ne_name in PF_VARIABLES:
            _bind_named(ne_name, ne, wrap(PF_VARIABLES[ne_name]), save=False)
        else:
            ne_str = (ne.get('source_input') or ne.get('sourceInput') or '').strip()
            if not ne_str:
                continue
            try:
                validate_input_identifiers(ne_str, local_dict)
                parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=WM_TRANSFORMS)
                _bind_named(ne_name, ne, wrap(parsed_ne), save=True)
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    try:
        validate_input_identifiers(text, local_dict)
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=WM_TRANSFORMS)
        wrapper_target = wrap(parsed_target)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    return WMEvalContext(
        val=wrapper_target, solver=s, sol=sol, evolved_order=evolved_order,
        disp_subs=disp_subs, local_dict=local_dict, proc=proc, is_dc=is_dc,
        mismatch=mismatch, single_mode=single_mode, symbols=list(symbols),
    )


def finalize_wm_result(
    ctx: WMEvalContext,
    val_t: WMWrapper,
    val_g: WMWrapper,
    order: int,
    use_gain: bool,
) -> Tuple[str, str, str, Union[int, float], int]:
    """
    Renders the time / gain results, computes the leading order and records the
    result as the latest Tab 3 evaluation (for saving to the table).
    Returns: (latex_curr, latex_t, latex_g, leading_order, solver_length)
    """
    s, proc = ctx.solver, ctx.proc
    latex_t = latex(safe_subs(val_t.expr, ctx.disp_subs))
    latex_g = latex(safe_subs(val_g.expr, ctx.disp_subs))
    latex_curr = latex_g if use_gain else latex_t

    try:
        leading_order = compute_val_leading_order(val_t, s, dc=ctx.is_dc, process=proc)
    except Exception:
        leading_order = 0

    LAST_EVAL_WM.clear()
    LAST_EVAL_WM.update({
        'val_t': val_t,
        'val_g': val_g,
        'solver': s,
        'latex_t': latex_t,
        'latex_g': latex_g,
        'order': order,
        'leading_order': leading_order,
        'dc': ctx.is_dc,
        'process': proc,
        'mismatch': ctx.mismatch,
        'single_mode': ctx.single_mode,
    })

    return latex_curr, latex_t, latex_g, leading_order, len(s)


def eval_combination_assertions(
    text: str,
    dc: bool,
    order: int,
    use_gain: bool,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    mm_operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    mismatch: bool = False,
    single_mode: bool = False,
    coherent_drive: bool = False,
    vacuum_signal: bool = False,
    vacuum_other_weak: bool = False,
    process: Optional[str] = None,
) -> Tuple[str, str, str, Union[int, float], int]:
    """
    Evaluates operator combination (optionally under input-state assertions):
    - coherent_drive: strong driving field input is in coherent state |alpha>
    - vacuum_signal: signal field (a_1) input is in vacuum |0>
    - vacuum_other_weak: remaining weak field (a_2 if is_dc else a_0) input is in vacuum |0>
    """
    ctx = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode, process=process
    )
    val_t, val_g = apply_assertions(
        ctx.val, ctx.solver, order, process=ctx.proc, mismatch=ctx.mismatch, single_mode=single_mode,
        coherent_drive=coherent_drive, vacuum_signal=vacuum_signal, vacuum_other_weak=vacuum_other_weak,
    )
    return finalize_wm_result(ctx, val_t, val_g, order, use_gain)


def eval_operator_combination(
    text: str,
    dc: bool,
    order: int,
    use_gain: bool,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    mm_operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    mismatch: bool = False,
    single_mode: bool = False,
    process: Optional[str] = None,
) -> Tuple[str, str, str, Union[int, float], int]:
    """Evaluates an operator combination without any input-state assertions."""
    return eval_combination_assertions(
        text, dc, order, use_gain, symbols, operators, mm_operators, named_expressions,
        mismatch=mismatch, single_mode=single_mode, process=process,
    )
