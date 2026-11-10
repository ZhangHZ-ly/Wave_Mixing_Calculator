from __future__ import annotations
import ast
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from cas_qwm.parametric_multimode.dsolve_pm import d_p2s
from cas_qwm.op_pattern import BosonNum, BosonPat, FermionPat
from cas_qwm.pattern_perturb.pattern_form import _vpk
from cas_qwm.parametric_multimode import *
from cas_qwm.parametric_multimode.wm_processor import t_, _g, nm, _ps, nlc, _tdz
from cas_qwm.parametric_multimode.wm_solver import pm
from cas_qwm.pattern_perturb import PF, PF1
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex

from .common import (
    WM_TRANSFORMS,
    is_valid_variable_name,
    RESERVED_NAMES,
    g_hat,
)
from .conditions import normalize_process, process_is_dc
from .wm_transforms import t_to_g
from .pf_expr import PF_VARIABLES

# Cached TWMSolver instances indexed by [dc][mismatch]
TWM_SOLVERS = (
    (TWMSolver(False, False), TWMSolver(False, True)),
    (TWMSolver(True, False), TWMSolver(True, True)),
)
FWM_SOLVER = FWMSolver()


def get_wm_solver(process: Optional[str] = None, mismatch: bool = False, dc: Optional[bool] = None) -> PMSolver:
    proc = normalize_process(process, dc)
    if proc == 'fwm':
        return FWM_SOLVER
    return TWM_SOLVERS[1 if proc == 'twm_dc' else 0][1 if mismatch else 0]


def _solver_process(s: PMSolver) -> str:
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
        sol_order = min(1, max(-1, order))
        expr_t = s.sol_expr(var_index, sol_order)
        if not mismatch:
            expr_t = s._zq(expr_t)
        expr_g = t_to_g(expr_t, process='fwm', mismatch=bool(mismatch))
    else:
        sol_order = max(-1, order - 1)
        expr_t = s.sol_expr(var_index, sol_order)
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
    s = solver if isinstance(solver, PMSolver) else get_wm_solver(proc, mismatch=bool(kwargs.get('mismatch', False)))
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
    s: PMSolver,
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

    w_sub = WMWrapper.from_val(val, s.processor).apply(_sub_t_for_order)
    orders: List[Any] = []
    if w_sub.b_left is None:
        for p in w_sub.dict_repr.values():
            try:
                o = pe.lo_pf(p)[1]
                if o is not S.Infinity:
                    orders.append(o)
            except Exception:
                pass
    else:
        for b, d in w_sub.dict_repr.items():
            b_ord = pe.leading_order(b)[1]
            for p in d.values():
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
                if name in ('nlc', 'chi', r'\chi', 'χ') or s_def.get('builtin_key') == 'nlc':
                    sym_orders[name] = 1.0
                else:
                    try:
                        sym_orders[name] = float(s_def.get('order', 0))
                    except (ValueError, TypeError):
                        sym_orders[name] = 0.0

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
            if name in ('nlc', 'chi', r'\chi', 'χ'):
                return True, 1.0, 'exact', name
            lo = strong_lo if name in strong_names else 0.0
            return True, lo, 'exact', name
        if isinstance(node, ast.UnaryOp):
            return get_factor_info(node.operand)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ''
            if func_name in ('dag', 'Dagger', 'dagger', 'diff', 'Derivative') and node.args:
                return get_factor_info(node.args[0])
            if func_name in ('N_', 'BosonNum') and node.args:
                arg = node.args[0]
                arg_name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else (arg.id if isinstance(arg, ast.Name) else '')
                lo = strong_nd_lo if arg_name in strong_names else 0.0
                return True, lo, 'exact', None
            if func_name == 'sqrt' and node.args:
                inner_terms = expand_additive_products(node.args[0])
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
                lo = min_lo * 0.5
                return (False, lo, 'evolved', evolved_ref) if has_evolved else (True, lo, 'exact', None)
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

            if isinstance(node.left, ast.BinOp) and isinstance(node.left.op, ast.Mult):
                inner_terms = expand_additive_products(node.left)
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
                lo = min_lo * exp_val
                return (False, lo, 'evolved', evolved_ref) if has_evolved else (True, lo, 'exact', None)
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
            if func_name in ('dag', 'Dagger', 'dagger') and len(node.args) == 1:
                inner_terms = expand_additive_products(node.args[0])
                if len(inner_terms) > 1 or (len(inner_terms) == 1 and len(inner_terms[0]) > 1):
                    return inner_terms
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
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
        tree = ast.parse(expr_str, mode='eval')
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

            if cat == 'evolved':
                is_strong = (ref in strong_names)
                if proc == 'fwm':
                    needed_for_op = 1 if req_order_int >= 1 else (0 if req_order_int >= 0 else -1)
                    if req_order_int > 1:
                        name = str(ref)
                        key = ('evolved', name, 1, req_order_int)
                        if key not in seen_warnings:
                            seen_warnings.add(key)
                            insufficient_items.append({
                                'name': name,
                                'kind': 'evolved_op',
                                'order': 1,
                                'required_order': req_order_int,
                            })
                elif is_strong:
                    needed_for_op = -1 if req_order_int <= 0 else (req_order_int if req_order_int % 2 == 1 else req_order_int - 1)
                else:
                    needed_for_op = -1 if req_order_int < 0 else (req_order_int if req_order_int % 2 == 0 else req_order_int - 1)

                if needed_for_op > max_evolved_needed:
                    max_evolved_needed = needed_for_op
            elif cat == 'named':
                rec_order = ref.get('order') if isinstance(ref, dict) else getattr(ref, 'order', None)
                if rec_order is not None and float(rec_order) < req_order_raw - 1e-12:
                    name = ref.get('name', '') if isinstance(ref, dict) else getattr(ref, 'name', '')
                    key = ('named', name, rec_order, req_order_int)
                    if key not in seen_warnings:
                        seen_warnings.add(key)
                        insufficient_items.append({
                            'name': name,
                            'kind': 'named_expr',
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
    import io, tokenize
    try:
        toks = list(tokenize.tokenize(io.BytesIO(expr_str.encode('utf-8')).readline))
    except Exception:
        return
    for tok in toks:
        if tok.type == tokenize.NAME:
            name = tok.string
            if name.count('_') >= 2 and name not in local_dict and name not in RESERVED_NAMES:
                raise ValueError(
                    f"不规范输入：未定义的复合变量名 '{name}'。若要执行乘法运算，请在变量之间使用空格（单模乘法）或'*'（多模乘法）分隔。"
                )


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
    Returns: (wrapper_target, solver, disp_subs, proc, is_dc)
    """
    proc = normalize_process(process, dc)
    is_dc = process_is_dc(proc)
    mismatch = bool(mismatch)
    s = get_wm_solver(proc, mismatch=mismatch)
    p = s.processor
    p.reset_pe(fwm=(proc == 'fwm'))

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
    sol = s.get_sol(evolved_order)

    if proc == 'fwm' and evolved_order >= 0:
        spm_w = WMWrapper(p, {(None,): PF({((), ()): exp(-I * nlc * s.nd * t_)})})
        sol = (sol[0] * spm_w, sol[1], sol[2])

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

    def _make_boson_num(item):
        return WMWrapper.from_val(BosonNum(_resolve_mode_name(item)), p)

    def _make_mmn(name, disp=0):
        return WMWrapper.from_val(MultimodeNum(_resolve_mode_name(name), _unwrap_scalar(disp)), p)

    def _make_mmn2(disp=0):
        d_val = _unwrap_scalar(disp)
        val = p.mmn2 + d_val * nm if d_val != 0 else p.mmn2
        return WMWrapper.from_val(val, p)

    def _make_pat(item, *args):
        raw_args = [a if isinstance(a, bool) else _unwrap_scalar(a) for a in args]

        is_A = isinstance(item, WMWrapper) and (item is init_ops['A'] or item == init_ops['A'])
        is_dagA = (isinstance(item, WMWrapper) and item.b_left is None
                   and list(item.dict_repr.keys()) == [(False, 1)]
                   and item.dict_repr[(False, 1)].table == {_vpk: S.One})
        if is_A or is_dagA:
            is_ann = is_A
            exps = list(raw_args)
            if exps and isinstance(exps[0], bool):
                is_ann = exps.pop(0)
            if not exps:
                exps = [1]
            int_exps = [int(sympify(e)) for e in exps]
            if any(e < 0 for e in int_exps):
                raise ValueError("Pattern exponents must be non-negative integers.")
            return WMWrapper(p, {_tdz(is_ann, int_exps): PF1})

        if isinstance(item, WMWrapper) and (item.b_left is not None or set(item.dict_repr.keys()) != {(None,)}):
            exps = list(raw_args)
            explicit_ann = exps.pop(0) if (exps and isinstance(exps[0], bool)) else None
            if not exps:
                exps = [1]
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
                res = res * (cur ** int(sympify(e)))
                is_a = not is_a
            return res

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
            inner = op.args[0]
            is_ann = False
        else:
            inner = op
            is_ann = op.is_annihilation

        name = str(op_mode_map.get(inner.name, inner.name))
        exps = list(raw_args)
        if exps and isinstance(exps[0], bool):
            is_ann = exps.pop(0)
        if not exps:
            exps = [1]

        is_fermion = (mode_kind_map.get(name) == 'fermion') or isinstance(inner, (FermionOp, FermionPat))
        if is_fermion:
            return WMWrapper.from_val(FermionPat(name, is_ann, *exps), p)

        bp = BosonPat(name, is_ann, *exps)
        a_op, tail = bp.num_repr()
        b_key = (a_op,) if a_op is not S.One else ()
        return WMWrapper(p, PF({(b_key, ()): tail}))

    def _make_op_pat(pat_cls, item, *args):
        raw_args = [a if isinstance(a, bool) else _unwrap_scalar(a) for a in args]
        return WMWrapper.from_val(pat_cls(_resolve_mode_name(item), *raw_args), p)

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

    w_g_internal = WMWrapper(p, PF({_vpk: s.gc * t_}))
    w_g_internal._is_g = True
    w_g_internal._gc = s.gc

    w_t = WMWrapper.from_val(t_, p)
    w_t._is_t = True

    pm_internal = s.pm if proc == 'fwm' else (pm if mismatch else S.Zero)

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
        'dag': lambda item: WMWrapper.from_val(item, p).adjoint(),
        'Dagger': lambda item: WMWrapper.from_val(item, p).adjoint(),
        'dagger': lambda item: WMWrapper.from_val(item, p).adjoint(),
        'N_': _make_boson_num,
        'BosonNum': _make_boson_num,
        'Pat': _make_pat,
        'pattern': _make_pat,
        'BP': lambda item, *args: _make_op_pat(BosonPat, item, *args),
        'BosonPat': lambda item, *args: _make_op_pat(BosonPat, item, *args),
        'FP': lambda item, *args: _make_op_pat(FermionPat, item, *args),
        'FermionPat': lambda item, *args: _make_op_pat(FermionPat, item, *args),
        'MMN': _make_mmn,
        'MMN2': _make_mmn2,
        't': w_t,
        'g': w_g_internal,
        'diff': lambda expr, *args, **kwargs: expr.diff(*args, gc=s.gc, **kwargs) if isinstance(expr, WMWrapper) else diff(expr, *args, **kwargs),
        'Derivative': lambda expr, *args, **kwargs: expr.diff(*args, gc=s.gc, **kwargs) if isinstance(expr, WMWrapper) else diff(expr, *args, **kwargs),
        'nlc': WMWrapper.from_val(nlc, p),
        'pm': WMWrapper.from_val(pm_internal, p),
        'n_m': WMWrapper.from_val(S.One if single_mode else nm, p),
        'a_0': init_ops['a_0'],
        'a_1': init_ops['a_1'],
        'a_2': init_ops['a_2'],
        'A': init_ops['A'],
        'N_t': init_ops['N_t'],
        'a_p': sol[0],
        'a_s': sol[1],
        'a_i': sol[2],
    }

    disp_subs: Dict[Any, Any] = {}

    for sym_def in symbols:
        s_name = (sym_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(s_name)
        if not valid:
            raise ValueError(f"Invalid symbol identifier: {err_msg}")
        d_name = (sym_def.get('display_name') or '').strip()

        if s_name == 'nlc' or sym_def.get('builtin_key') == 'nlc':
            local_dict[s_name] = WMWrapper.from_val(nlc, p)
            if d_name and d_name not in ('nlc', r'\chi', 'χ'):
                disp_subs[nlc] = Symbol(d_name)
            continue
        if s_name == 'pm' or sym_def.get('builtin_key') == 'pm':
            if not mismatch:
                local_dict[s_name] = WMWrapper.from_val(pm_internal, p)
            else:
                local_dict[s_name] = WMWrapper.from_val(pm, p)
                if d_name and d_name not in ('pm', 'Δk', r'\Delta k'):
                    disp_subs[pm] = Symbol(d_name)
            continue

        kwargs = {}
        if sym_def.get('is_positive'): kwargs['positive'] = True
        elif sym_def.get('is_nonnegative'): kwargs['nonnegative'] = True
        elif sym_def.get('is_integer'): kwargs['integer'] = True
        elif sym_def.get('is_real'): kwargs['real'] = True
        sym_obj = Symbol(s_name, **kwargs)
        local_dict[s_name] = WMWrapper.from_val(sym_obj, p)
        if d_name and d_name != s_name:
            disp_subs[sym_obj] = Symbol(d_name)

    for op_def in operators:
        op_name = (op_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(op_name)
        if not valid:
            raise ValueError(f"Invalid operator identifier: {err_msg}")

        op_type = op_def.get('type')
        if op_type == 'operator' or ('kind' in op_def and 'mode_index' not in op_def):
            kind = op_def.get('kind', 'boson')
            mode_kind_map[op_name] = kind
            op_mode_map[op_name] = op_name
            op_obj = BosonOp(op_name) if kind == 'boson' else FermionOp(op_name)
            local_dict[op_name] = WMWrapper.from_val(op_obj, p)

            d_name = (op_def.get('display_name') or '').strip()
            if d_name and d_name != op_name:
                disp_subs[Symbol(op_name)] = Symbol(d_name)
        else:
            mode_idx = int(op_def.get('mode_index', 0))
            if 0 <= mode_idx <= 2:
                local_dict[op_name] = sol[mode_idx]
                default_mode_name = f'a_{mode_idx}'
                op_mode_map[op_name] = default_mode_name
                mode_kind_map[op_name] = 'boson'
                mode_kind_map[default_mode_name] = 'boson'

                d_name = (op_def.get('display_name') or '').strip()
                if d_name and d_name != default_mode_name:
                    disp_subs[Symbol(default_mode_name)] = Symbol(d_name)

    for mm_def in mm_operators:
        mm_name = (mm_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(mm_name)
        if not valid:
            raise ValueError(f"Invalid multimode operator identifier: {err_msg}")
        mm_kind = mm_def.get('mm_kind', 'COP')
        local_dict[mm_name] = init_ops['A'] if mm_kind == 'COP' else init_ops['N_t']

    for ne in named_expressions:
        ne_name = (ne.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")

        val_obj = ne.get('val')
        if val_obj is not None:
            w_ne = WMWrapper.from_val(val_obj, p)
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
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
        elif ne_name in WM_VARIABLES:
            w_ne = WM_VARIABLES[ne_name]
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
        elif ne_name in PF_VARIABLES:
            w_ne = WMWrapper.from_val(PF_VARIABLES[ne_name], p)
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
        else:
            ne_str = (ne.get('source_input') or ne.get('sourceInput') or '').strip()
            if not ne_str:
                continue
            try:
                validate_input_identifiers(ne_str, local_dict)
                parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=WM_TRANSFORMS)
                w_ne = WMWrapper.from_val(parsed_ne, p)
                if single_mode:
                    w_ne = w_ne.to_single_mode()
                local_dict[ne_name] = w_ne
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
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    try:
        validate_input_identifiers(text, local_dict)
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=WM_TRANSFORMS)
        wrapper_target = WMWrapper.from_val(parsed_target, p)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    return wrapper_target, s, disp_subs, proc, is_dc


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
    val, s, disp_subs, proc, is_dc = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode, process=process
    )
    p = s.processor

    vac_modes = []
    if vacuum_signal:
        vac_modes.append('a_1')
    if vacuum_other_weak:
        vac_modes.append('a_2' if is_dc else 'a_0')

    if vac_modes:
        val = val.eval_vacuum(vac_modes)

    if coherent_drive:
        if single_mode:
            val = val.to_single_mode()
        val_t = val.eval_coherent(order)
        if proc == 'fwm' and not mismatch:
            alpha_sym = Symbol('alpha_0')
            pm_cohr = 2 * nlc * (Abs(alpha_sym) ** 2)
            val_t = val_t.apply(lambda x: bottom_up(x.subs({s.pm: pm_cohr}), _ps))
        val_g = t_to_g(val_t, process=proc, mismatch=mismatch, coherent=True)
    else:
        def _postprocess_pf(pf_item: PF) -> PF:
            if not pf_item:
                return pf_item
            if order >= 0 or pf_item.expr(False).has(t_):
                return pf_item.n_apply(
                    lambda t: p.n_simp(d_p2s(_g, p.p2s_group(t)).subs({_g: s.gc * t_}))
                )
            return pf_item.n_apply(p.n_simp)

        val = val.expand(order).apply(_postprocess_pf)
        if proc == 'fwm' and not mismatch:
            val = s._zq(val)
        if single_mode:
            val = val.to_single_mode()
        val_t = val
        val_g = t_to_g(val_t, process=proc, mismatch=mismatch)

    expr_disp_t = val_t.expr
    expr_disp_g = val_g.expr
    if disp_subs:
        try:
            expr_disp_t = expr_disp_t.subs(disp_subs)
        except Exception:
            pass
        try:
            expr_disp_g = expr_disp_g.subs(disp_subs)
        except Exception:
            pass

    latex_t = latex(expr_disp_t)
    latex_g = latex(expr_disp_g)
    latex_curr = latex_g if use_gain else latex_t

    try:
        leading_order = compute_val_leading_order(val_t, s, dc=is_dc, process=proc)
    except Exception:
        leading_order = 0

    sol_len = len(s)

    global LAST_EVAL_WM
    LAST_EVAL_WM = {
        'val_t': val_t,
        'val_g': val_g,
        'val': val_t,
        'solver': s,
        'latex_t': latex_t,
        'latex_g': latex_g,
        'order': order,
        'leading_order': leading_order,
        'dc': is_dc,
        'process': proc,
        'mismatch': mismatch,
        'single_mode': single_mode,
    }

    return latex_curr, latex_t, latex_g, leading_order, sol_len


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
    return eval_combination_assertions(
        text=text,
        dc=dc,
        order=order,
        use_gain=use_gain,
        symbols=symbols,
        operators=operators,
        mm_operators=mm_operators,
        named_expressions=named_expressions,
        mismatch=mismatch,
        single_mode=single_mode,
        coherent_drive=False,
        vacuum_signal=False,
        vacuum_other_weak=False,
        process=process,
    )
