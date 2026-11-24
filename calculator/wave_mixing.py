from __future__ import annotations
import ast
import io
import re
import tokenize

import sympy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

from cas_qwm.op_pattern import BosonNum, BosonPat, FermionPat
from cas_qwm.pattern_perturb.pattern_form import _vpk
from cas_qwm.parametric_multimode import *
from cas_qwm.parametric_multimode.wm_processor import _tdz
from cas_qwm.wm_variables import t_, nlc, pm, nm, gain
from cas_qwm.pattern_perturb import PF, PF1
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.numbers import I, pi
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import SympifyError, sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
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
    order_by_refs,
    symbol_assumptions,
    validated_name,
)
from .conditions import normalize_process, process_is_dc, is_degenerate, referenced_expressions
from .wm_transforms import apply_assertions

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


WM_VARIABLES: Dict[str, WMWrapper] = {}
LAST_EVAL_WM: Dict[str, Any] = {}
# latest evaluation under assertions (exponential form, saving)
LAST_ASSERT_WM: Dict[str, Any] = {}


def save_wm_variable(def_id: str, val: Any, solver: WMSolver, order: Any = None) -> WMWrapper:
    w = WMWrapper.from_val(val, solver.processor)
    # accuracy order: the recorded one, or the wrapper's own if lower
    if order is not None and not w.is_exact:
        order = min(as_order(order), w.order)
    elif order is None:
        order = w.order
    w = WMWrapper(w.processor, dict(w.dict_repr), w.b_left, order)
    WM_VARIABLES[def_id] = w
    return w


def save_latest_wm_variable(def_id: str) -> Optional[WMWrapper]:
    if not LAST_EVAL_WM:
        return None
    return save_wm_variable(def_id, LAST_EVAL_WM['val_t'], LAST_EVAL_WM['solver'], order=LAST_EVAL_WM['order'])


def save_latest_assert_variable(def_id: str) -> Optional[WMWrapper]:
    """Saves the latest evaluation under assertions (an asserted expression)."""
    if not LAST_ASSERT_WM:
        return None
    return save_wm_variable(def_id, LAST_ASSERT_WM['val_t'], LAST_ASSERT_WM['solver'], order=LAST_ASSERT_WM['order'])


def assertion_exp_latex(use_gain: bool = False) -> str:
    """
    Cached LaTeX of the latest result under assertions in exponential form (the gain form is
    rewritten itself, i.e. after t -> g).
    """
    if not LAST_ASSERT_WM:
        raise ValueError('No assertion result available: please calculate under assertions first.')
    key = 'val_g' if use_gain else 'val_t'
    cache = LAST_ASSERT_WM.setdefault('exp', {})
    if key not in cache:
        cache[key] = LAST_ASSERT_WM[key].expr.rewrite(exp).expand()
    return latex(safe_subs(cache[key], LAST_ASSERT_WM['disp_subs']))


def compute_val_leading_order(
    val: Any,
    s: WMSolver,
    dc: Optional[bool] = None,
    process: Optional[str] = None,
):
    """
    Computes the (exact) leading perturbation order of a calculated WMWrapper result `val`.
    """
    proc = normalize_process(process if process is not None else _solver_process(s), dc)
    pe = s.pe

    def _sub_t_for_order(expr_t: Expr) -> Expr:
        if not expr_t.has(t_):
            return expr_t
        gc_eff = (nlc * s.nd) if (proc == 'fwm' and not expr_t.has(pm)) else s.gc
        return expr_t.subs({t_: gain / gc_eff})

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
        return S.Zero
    try:
        return min(orders)
    except Exception:
        return S.Zero


_NLC_NAMES = ('nlc', 'chi', r'\chi', 'χ')
# functions of the Tab 3 input (other names before '(' multiply, except SymPy functions)
_INPUT_FUNCS = frozenset((*ELEMENTARY_FUNCS, *DAGGER_ALIASES, 'N_', 'BosonNum', 'Pat', 'pattern',
                          'BP', 'BosonPat', 'FP', 'FermionPat', 'MMN', 'MMN2', 'diff', 'Derivative'))


def _const_value(node):
    """Exact value of a numeric constant expression node (e.g. -1/2, (1+2)/3), or None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return as_order(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const_value(node.operand)
        return None if v is None else (-v if isinstance(node.op, ast.USub) else v)
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = _const_value(node.left), _const_value(node.right)
        if a is None or b is None:
            return None
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        return a / b if b != 0 else None
    return None


def _ref_attr(ref: Any, key: str, default: Any = None) -> Any:
    return ref.get(key, default) if isinstance(ref, dict) else getattr(ref, key, default)


def as_order(x):
    """
    A perturbative order parsed as an exact SymPy number: fractions and decimals both become
    Rationals ('1/2', '0.5' and 0.5 all give 1/2), so that no float enters the order algebra.
    """
    v = sympify(str(x), rational=True)
    if not v.is_Number:
        raise ValueError(f"Invalid perturbative order: {x!r}")
    return v


def order_text(x) -> str:
    """
    Text of an exact perturbative order, as exchanged with the frontend (e.g. '-2', '1/2'):
    orders are stored as text there, so that fractions stay exact.
    """
    return str(as_order(x))



def analyze_expression_terms(
    expr_str: str,
    target_order,
    dc: bool = True,
    symbols: Optional[List[Dict[str, Any]]] = None,
    operators: Optional[List[Dict[str, Any]]] = None,
    named_expressions: Optional[List[Dict[str, Any]]] = None,
    process: Optional[str] = None,
) -> Tuple[Any, List[Dict[str, Any]], Any]:
    """
    Analyzes an expression across each additive term to determine required
    evolved operator order and identify any saved named expressions with
    insufficient perturbation order.
    Returns (evolved order, insufficient items, function order); the function order is the
    truncation order needed by functions / powers / divisors of evolved operators (-1 if none).
    Both orders are None if the input is not fully analysable (e.g. diff).
    """
    proc = normalize_process(process, dc)
    is_dc = process_is_dc(proc)
    # base orders of the process (FWM: χ of order 2, TWM: 1; the drive of order -1 in both)
    wm_p = get_wm_solver(proc).processor
    strong_lo = wm_p.base_orders[wm_p.nd.name]
    strong_nd_lo = 2 * strong_lo
    nlc_lo = wm_p.base_orders[nlc]

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

    # orders are kept exact (SymPy numbers) throughout the analysis
    sym_orders: Dict[str, Any] = {}
    if symbols:
        for s_def in symbols:
            name = (s_def.get('name') or '').strip()
            if name:
                if name in _NLC_NAMES or s_def.get('builtin_key') == 'nlc':
                    sym_orders[name] = nlc_lo
                else:
                    try:
                        sym_orders[name] = as_order(s_def.get('order') or 0)
                    except (ValueError, TypeError, SympifyError):
                        sym_orders[name] = S.Zero

    def _terms_lo(expr_node, mult) -> Tuple[bool, Any, str, Any]:
        inner_terms = expand_additive_products(expr_node)
        min_lo = S.Zero
        has_evolved = False
        evolved_ref = None
        min_lo = None
        for t in inner_terms:
            t_lo = sum(get_factor_info(fn)[1] * sm for fn, sm in t)
            min_lo = t_lo if min_lo is None else min(min_lo, t_lo)
            for fn, _ in t:
                _, _, cat, ref = get_factor_info(fn)
                if cat == 'evolved':
                    has_evolved = True
                    evolved_ref = ref
        lo = (S.Zero if min_lo is None else min_lo) * mult
        return (False, lo, 'evolved', evolved_ref) if has_evolved else (True, lo, 'exact', None)

    def get_factor_info(node) -> Tuple[bool, Any, str, Any]:
        if isinstance(node, ast.Constant):
            return True, S.Zero, 'exact', None
        if isinstance(node, ast.Name):
            name = node.id
            if name in named_map:
                ne = named_map[name]
                return False, as_order(ne.get('leading_order') or 0), 'named', ne
            if name in evolved_names:
                lo = strong_lo if name in strong_names else S.Zero
                return False, lo, 'evolved', name
            if name in sym_orders:
                return True, sym_orders[name], 'exact', name
            if name in _NLC_NAMES:
                return True, nlc_lo, 'exact', name
            lo = strong_lo if name in strong_names else S.Zero
            return True, lo, 'exact', name
        if isinstance(node, ast.UnaryOp):
            return get_factor_info(node.operand)
        if isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else ''
            if func_name in ('diff', 'Derivative') and node.args and _has_evolved(node):
                unmodeled[0] = True  # the order of a derivative is not tracked
            if func_name in (*DAGGER_ALIASES, 'diff', 'Derivative') and node.args:
                return get_factor_info(node.args[0])
            if func_name in ('N_', 'BosonNum') and node.args:
                arg = node.args[0]
                arg_name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else (arg.id if isinstance(arg, ast.Name) else '')
                lo = strong_nd_lo if arg_name in strong_names else S.Zero
                return True, lo, 'exact', None
            if func_name in ELEMENTARY_FUNCS and node.args:
                # leading order of f(x0): half the argument's for sqrt, the argument's for
                # sin / sinh / tanh of a small argument, else 0
                is_ex, lo, cat, ref = _terms_lo(node.args[0], S.One)
                if func_name == 'sqrt':
                    flo = lo / 2
                elif func_name in ('sin', 'sinh', 'tanh') and lo > 0:
                    flo = lo
                else:
                    flo = S.Zero
                return is_ex, flo, cat, ref
            if _has_evolved(node):
                unmodeled[0] = True
            return True, S.Zero, 'exact', None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            exp_val = _const_value(node.right)
            if exp_val is None:
                exp_val = S.One

            if isinstance(node.left, ast.BinOp) and not isinstance(node.left.op, ast.Pow):
                return _terms_lo(node.left, exp_val)
            base_exact, base_lo, cat, ref = get_factor_info(node.left)
            return base_exact, base_lo * exp_val, cat, ref
        if isinstance(node, ast.BinOp) and _has_evolved(node):
            # composite factor left unexpanded
            unmodeled[0] = True
            return _terms_lo(node, S.One)
        return True, S.Zero, 'exact', None

    def _has_evolved(node) -> bool:
        return any(isinstance(n, ast.Name) and n.id in evolved_names for n in ast.walk(node))

    unmodeled = [False]  # the input contains evolved operators the analysis does not model

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
            if func_name in ('Pat', 'pattern') and node.args:
                # a pattern: the product of the operator's and its adjoint's powers
                args = node.args[1:]
                if args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, bool):
                    args = args[1:]
                exps = [_const_value(a) for a in args] or [S.One]
                if all(e is not None and e.is_Integer and e >= 0 for e in exps):
                    return [[(node.args[0], 1)] * int(sum(exps))]
                if _has_evolved(node):
                    unmodeled[0] = True
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
        # a name before '(' multiplies unless it is a function (implicit multiplication)
        defined = {*named_map, *sym_orders, *evolved_names, *strong_names,
                   *((op.get('name') or '').strip() for op in (operators or ()))}

        def _callable(name: str) -> bool:
            if name in _INPUT_FUNCS:
                return True
            return name not in defined and callable(getattr(sympy, name, None))

        s = re.sub(r'\b([A-Za-z_]\w*)(\s*)\(',
                   lambda m: m.group(0) if _callable(m.group(1)) else m.group(1) + '*(', s)
        s = re.sub(r'\)\s*(?=[A-Za-z0-9_(])', ')*', s)
        s = re.sub(r'(\d)\s*(?=[A-Za-z_(])', r'\1*', s)
        s = re.sub(r'([A-Za-z0-9_])\s+(?=[A-Za-z0-9_])', r'\1*', s)
        tree = ast.parse(s, mode='eval')
        expanded_terms = expand_additive_products(tree)
    except Exception:
        return None, [], None

    def _expanded_base(node, sign_mult):
        """
        (base node, base leading order) of a factor expanded by expand_func_d (a divisor,
        a non-integer or negative power, or an elementary function), else None.
        """
        lo = get_factor_info(node)[1]
        if sign_mult < 0:
            return node, lo
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            e = _const_value(node.right)
            if e is not None and e.is_Integer and e >= 0:
                return None
            e = S.One if e is None else e  # as in get_factor_info
            return (node.left, lo / e) if e != 0 else None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ELEMENTARY_FUNCS and node.args):
            return node.args[0], _terms_lo(node.args[0], S.One)[1]
        return None

    is_fwm = proc == 'fwm'
    # FWM: the highest orders the evolved operators reach (in the solution of order 1: pump 2,
    # weak fields 1)
    if is_fwm:
        fwm_top = FWM_SOLVER.get_sol(FWM_SOLVER.orders[-1])
        fwm_caps = (fwm_top[0].order, fwm_top[1].order)
    max_evolved_needed = -1
    max_func_needed = -1
    insufficient_items: List[Dict[str, Any]] = []
    seen_warnings = set()

    def _require(terms, target) -> None:
        """
        Records what the terms need for the accuracy `target`: each factor needs target minus
        the leading orders of the others; an expanded factor keeps the relative accuracy of its
        base, which is required recursively.
        """
        nonlocal max_evolved_needed, max_func_needed
        for raw_factors in terms:
            parsed_factors = []
            total_lo_sum = S.Zero
            has_estimated = False

            for f_node, sign_mult in raw_factors:
                is_exact, lo, cat, ref = get_factor_info(f_node)
                eff_lo = lo * sign_mult
                total_lo_sum += eff_lo
                if not is_exact:
                    has_estimated = True
                parsed_factors.append((f_node, sign_mult, is_exact, eff_lo, cat, ref))

            if not has_estimated:
                continue

            for f_node, sign_mult, is_exact, eff_lo, cat, ref in parsed_factors:
                if is_exact:
                    continue
                other_lo_sum = total_lo_sum - eff_lo
                req_order_raw = target - other_lo_sum
                req_order = ceiling(req_order_raw)

                if cat == 'evolved':
                    base = _expanded_base(f_node, sign_mult)
                    if base is not None:
                        base_node, base_lo = base
                        max_func_needed = max(max_func_needed, req_order)
                        _require(expand_additive_products(base_node), base_lo + req_order_raw - eff_lo)
                        continue

                warning = None
                if cat == 'evolved':
                    if is_fwm:
                        # the FWM solution of the required order (get_sol takes the highest
                        # solution not above it); the evolved operators are capped at the orders
                        # of the last one
                        needed_for_op = req_order
                        cap = fwm_caps[0 if ref in strong_names else 1]
                        if req_order > cap:
                            warning = (str(ref), 'evolved_op', cap)
                    elif ref in strong_names:
                        needed_for_op = -1 if req_order <= 0 else (req_order if req_order % 2 == 1 else req_order - 1)
                    else:
                        needed_for_op = -1 if req_order < 0 else (req_order if req_order % 2 == 0 else req_order - 1)
                    max_evolved_needed = max(max_evolved_needed, needed_for_op)
                elif cat == 'named':
                    rec_order = _ref_attr(ref, 'order')
                    if rec_order is not None and as_order(rec_order) < req_order_raw:
                        warning = (_ref_attr(ref, 'name', ''), 'named_expr', rec_order)

                if warning is not None:
                    name, kind, rec_order = warning
                    key = (kind, name, rec_order, req_order)
                    if key not in seen_warnings:
                        seen_warnings.add(key)
                        insufficient_items.append({
                            'name': name,
                            'kind': kind,
                            'order': rec_order,
                            'required_order': req_order,
                        })

    _require(expanded_terms, as_order(target_order))
    if unmodeled[0]:
        return None, insufficient_items, None

    return max_evolved_needed, insufficient_items, max_func_needed


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
    degenerate: bool = False
    # warnings for the user (e.g. saved expressions rebuilt below their order)
    warnings: List[str] = field(default_factory=list)


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
    degenerate: bool = False,
):
    """
    Prepares and algebraically evaluates wave-mixing operator combinations in Tab 3.
    Returns a WMEvalContext holding the parsed wrapper and the evaluation settings.

    degenerate (effective only with single_mode in down-conversion): every input object
    (evolved and initial operators, multimode operators, named expressions) is converted to
    the degenerate single mode (signal = idler = a_1) before parsing, so that all products
    are exact single-mode products of the degenerate process.
    """
    proc = normalize_process(process, dc)
    is_dc = process_is_dc(proc)
    mismatch = bool(mismatch)
    degenerate = is_degenerate(proc, single_mode, degenerate)
    _sm = (lambda w: w.to_single_mode(degenerate=True)) if degenerate else (lambda w: w)
    s = get_wm_solver(proc, mismatch=mismatch)
    p = s.processor

    # seeded expressions of multimode processing in single-mode processing: no modes are left besides
    # the seeded ones; the seed keeps its degeneracy, which must be that of the preset (the
    # expressions of the other one are left out, and must not be referenced)
    seeded_mm, other_deg = set(), set()
    if single_mode:
        for ne in named_expressions:
            seed = (ne.get('assertion') or {}).get('seed_mode')
            if seed is not None and not ne.get('single_mode'):
                (seeded_mm if (proc == 'twm_dc' and int(seed) == 0) == degenerate else other_deg).add(ne['id'])
        for ne in referenced_expressions(text, named_expressions):
            if ne['id'] in other_deg:
                kind = 'degenerate' if degenerate else 'non-degenerate'
                raise ValueError(
                    f"Seeded expression '{ne.get('name') or ne.get('source_input')}' has a "
                    f"{'non-degenerate' if degenerate else 'degenerate'} seed, incompatible with the "
                    f"current {kind} single-mode preset.")
        named_expressions = [ne for ne in named_expressions if ne['id'] not in other_deg]
    # a degenerate seed is evaluated again in degenerate single-mode processing (not convertible)
    direct = seeded_mm if degenerate else set()

    # evolved order needed by the input and, to their recorded order, by the saved expressions
    # to rebuild; an analysis that is not complete falls back to its target order
    targets = [(text, as_order(order))]
    for ne in named_expressions:
        src = (ne.get('source_input') or '').strip()
        if src and (ne['id'] not in WM_VARIABLES or ne['id'] in direct):
            targets.append((src, as_order(ne['order']) if ne.get('order') is not None else as_order(order)))
    evolved_order, func_order = -1, -1
    for src, target in targets:
        ev, _, fn = analyze_expression_terms(
            src,
            target_order=target,
            dc=is_dc,
            symbols=symbols,
            operators=operators,
            named_expressions=named_expressions,
            process=proc,
        )
        if ev is None:
            ev = fn = ceiling(target)
        evolved_order, func_order = max(evolved_order, ev), max(func_order, fn)
    # truncation order of functions / powers of inexact operator expressions
    func_order = max(evolved_order, func_order)
    p._evolved_order = evolved_order
    p._func_order = func_order
    warnings: List[str] = []
    sol = tuple(_sm(w) for w in get_evolved_sol(s, evolved_order))

    init_ops = ({k: _sm(w) for k, w in p.init_ops.items()} if degenerate else p.init_ops)
    # degenerate: the converted evolved operators and their adjoints, as (annihilation wrapper,
    # is_annihilation) pairs, so that patterns of them are recognized
    deg_evolved = ([pair for w in sol for pair in ((w, True), (w.adjoint(), False))]
                   if degenerate else [])

    op_mode_map = {
        'a_0': 'a_0', 'a_p': 'a_0',
        'a_1': 'a_1', 'a_s': 'a_1',
        'a_2': 'a_2', 'a_i': 'a_2',
    }
    mode_kind_map = {k: 'boson' for k in op_mode_map}

    # custom orders keyed by the objects (boson modes as operators; fermions are of order 0)
    custom_orders = {}
    for item_def in (*symbols, *operators):
        name = (item_def.get('name') or '').strip()
        if not name or item_def.get('order') is None or item_def.get('builtin_key'):
            continue
        if item_def in symbols:
            obj = Symbol(name, **symbol_assumptions(item_def))
        elif ((item_def.get('type') == 'operator' or ('kind' in item_def and 'mode_index' not in item_def))
              and item_def.get('kind', 'boson') == 'boson'):
            obj = BosonOp(name)
        else:
            continue
        try:
            custom_orders[obj] = as_order(item_def['order'])
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
        return _sm(WMWrapper.from_val(BosonNum(_resolve_mode_name(item)), p))

    def _make_mmn(name, disp=0):
        return _sm(WMWrapper.from_val(MultimodeNum(_resolve_mode_name(name), _unwrap_scalar(disp)), p))

    def _make_mmn2(disp=0):
        d_val = _unwrap_scalar(disp)
        val = p.mmn2 + d_val * nm if d_val != 0 else p.mmn2
        return _sm(WMWrapper.from_val(val, p))

    def _make_pat(item, *args):
        explicit_ann, exps = _split_pat_args(args)

        # Multimode canonical operator A (or its adjoint)
        is_A = isinstance(item, WMWrapper) and (item is init_ops['A'] or item == init_ops['A'])
        is_dagA = (isinstance(item, WMWrapper) and item.b_left is None
                   and list(item.dict_repr.keys()) == [(False, 1)]
                   and item.dict_repr[(False, 1)].table == {_vpk: S.One})
        if degenerate and isinstance(item, WMWrapper) and not (is_A or is_dagA):
            is_dagA = item == init_ops['A'].adjoint()
        if is_A or is_dagA:
            is_ann = is_A if explicit_ann is None else explicit_ann
            int_exps = [int(sympify(e)) for e in exps]
            if any(e < 0 for e in int_exps):
                raise ValueError("Pattern exponents must be non-negative integers.")
            return _sm(WMWrapper(p, {_tdz(is_ann, int_exps): PF1}))

        # Degenerate single mode: patterns of the converted evolved operators
        deg_match = (next(((w, ann) for w, ann in deg_evolved if item is w or item == w), None)
                     if degenerate and isinstance(item, WMWrapper) else None)
        if deg_match is not None:
            w, ann = deg_match
            op_ann = w if ann else w.adjoint()
            is_a = ann if explicit_ann is None else bool(explicit_ann)
            res = WMWrapper(p, PF1)
            for e in exps:
                res = res * ((op_ann if is_a else op_ann.adjoint()) ** int(sympify(e)))
                is_a = not is_a
            return res

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
        return _sm(WMWrapper(p, PF({(b_key, ()): tail})))

    def _make_op_pat(pat_cls, item, *args):
        return _sm(WMWrapper.from_val(pat_cls(_resolve_mode_name(item), *_raw_args(args)), p))

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, WMWrapper):
                sc = item.as_scalar
                if sc is not None:
                    res = WMWrapper(p, PF({_vpk: sympify(fn(sc))}))
                    if not item.is_exact and res:
                        acc = res.lo + item.rel_order  # relative accuracy is kept
                        if acc.is_number and acc.is_finite:
                            res = res.with_order(acc)
                    return res
                if item.b_left is None:
                    # power handles (e.g. sqrt) follow the power rules of WMWrapper
                    x = Dummy()
                    fx = sympify(fn(x))
                    if fx.is_Pow and fx.base == x:
                        return item ** fx.exp
                    return p.expand_func_d(fn, func_order, item)
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
        'n_m': wrap(S.Half if degenerate else (S.One if single_mode else nm)),
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

    # Saved expressions: the stored object of each definition, or its source input re-evaluated
    # with the definitions it was saved with; the table names are bound to their current definitions
    base_dict = dict(local_dict)
    built: Dict[str, WMWrapper] = {}
    for ne in order_by_refs(named_expressions):
        ne_name = validated_name(ne, 'expression') if ne.get('name') else None
        is_direct = ne['id'] in direct
        w_ne = None if is_direct else WM_VARIABLES.get(ne['id'])
        is_new = w_ne is None
        cfg = ne.get('assertion')
        vacant = ne['id'] in seeded_mm
        replay_mm = vacant and not is_direct
        if is_new:
            ne_str = (ne.get('source_input') or '').strip()
            if not ne_str:
                continue
            scope = {**base_dict, **{n: built[r] for n, r in ne.get('refs', {}).items() if r in built}}
            try:
                validate_input_identifiers(ne_str, scope)
                w_ne = wrap(parse_expr(ne_str, local_dict=scope, transformations=WM_TRANSFORMS))
                if cfg:  # asserted expression: its assertions are applied again
                    w_ne = replay_assertions(WMEvalContext(
                        val=w_ne, solver=s, sol=sol, evolved_order=evolved_order, disp_subs=disp_subs,
                        local_dict=scope, proc=proc, is_dc=is_dc, mismatch=mismatch,
                        single_mode=single_mode and not replay_mm, symbols=list(symbols), degenerate=degenerate,
                    ), cfg)
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name or ne_str}': {e}")
        # a value without an accuracy of its own takes the recorded order of the table entry
        if w_ne.is_exact and ne.get('order') is not None:
            w_ne = w_ne.with_order(as_order(ne['order']))
        if is_new:
            rec = ne.get('order')
            if rec is not None and not w_ne.is_exact and w_ne.order < as_order(rec):
                # not kept, so that it is rebuilt when needed
                warnings.append(
                    f"Saved expression '{ne_name or ne_str}' was rebuilt from its source input "
                    f"(not held by the backend) only up to order {w_ne.order}, below its "
                    f"recorded order {rec}.")
            elif not is_direct:
                save_wm_variable(ne['id'], w_ne, s, order=None if w_ne.is_exact else w_ne.order)
        if single_mode and not (is_new and cfg and not replay_mm):  # replayed assertions convert it
            w_ne = w_ne.to_single_mode(degenerate=degenerate, vacant=vacant)
        built[ne['id']] = w_ne
        if ne_name:
            local_dict[ne_name] = w_ne

    try:
        validate_input_identifiers(text, local_dict)
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=WM_TRANSFORMS)
        wrapper_target = wrap(parsed_target)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    return WMEvalContext(
        val=wrapper_target, solver=s, sol=sol, evolved_order=evolved_order,
        disp_subs=disp_subs, local_dict=local_dict, proc=proc, is_dc=is_dc,
        mismatch=mismatch, single_mode=single_mode, symbols=list(symbols), degenerate=degenerate,
        warnings=warnings,
    )


def _latex_key(text: str) -> str:
    """Normalized LaTeX rendering of a symbol's display text (alpha_0, \\alpha_{0}, α_0, ...)."""
    return ''.join(latex(Symbol(text.replace('α', r'\alpha '))).split())


def coherent_alpha(ctx: WMEvalContext) -> Symbol:
    """
    Coherent amplitude symbol of the strong driving field (alpha_0 for dc, alpha_2 for fc).
    A user-defined commutative symbol displayed as that amplitude (its display text, or its
    name without one, renders as \\alpha_0 / \\alpha_2), whatever its name, is used as is
    (with its assumptions), so that it merges with the amplitude of the assertion.
    """
    name = 'alpha_0' if ctx.is_dc else 'alpha_2'
    target = _latex_key(name)
    for sym_def in ctx.symbols:
        s_name = (sym_def.get('name') or '').strip()
        if not s_name or sym_def.get('builtin_key'):
            continue
        shown = (sym_def.get('display_name') or '').strip() or s_name
        if _latex_key(shown) == target:
            return Symbol(s_name, **symbol_assumptions(sym_def))
    return Symbol(name)


def replay_assertions(ctx: WMEvalContext, cfg: Dict[str, Any]) -> WMWrapper:
    """
    Asserted expression rebuilt from its source input (ctx.val): the assertions of its
    configuration `cfg` (and its cascaded stage, if any) are applied again.
    """
    order = int(cfg.get('order', 1))
    val, times = ctx.val, ()
    cascade = cfg.get('cascade')
    if cascade:
        from .wm_cascaded import cascade_evolve, resolve_cascade_phase, resolve_cascade_symbol, tc
        t_sym = resolve_cascade_symbol(cascade['symbol'], ctx)
        phases = tuple(resolve_cascade_phase(ph, ctx, k)
                       for k, ph in enumerate((list(cascade.get('phases') or []) + ['', '', ''])[:3]))
        val = cascade_evolve(ctx, order, phases).apply(lambda e: e.subs(tc, t_sym), on_npf=True)
        times = (t_sym,)
    val_t, _ = apply_assertions(
        val, ctx.solver, order, process=ctx.proc, mismatch=ctx.mismatch, single_mode=ctx.single_mode,
        degenerate=ctx.degenerate, coherent_drive=bool(cfg.get('coherent_drive')),
        vacuum_signal=bool(cfg.get('vacuum_signal')), vacuum_other_weak=bool(cfg.get('vacuum_other_weak')),
        times=times, alpha=coherent_alpha(ctx), seed_mode=cfg.get('seed_mode'),
        seed_vac_c=cfg.get('seed_vac_c', True), seed_observed=cfg.get('seed_observed', False),
    )
    return val_t


def finalize_wm_result(
    ctx: WMEvalContext,
    val_t: WMWrapper,
    val_g: WMWrapper,
    order: int,
    use_gain: bool,
    assertion: bool = False,
) -> Tuple[str, str, str, Any, int, Any, List[str]]:
    """
    Renders the time / gain results, computes the leading order and records the
    result as the latest Tab 3 evaluation (for saving to the table), or with `assertion` as
    the latest evaluation under input-state assertions (LAST_ASSERT_WM).
    The accuracy order of the result follows the product rule of WMWrapper (the referenced
    variables entering with their recorded orders), so it may fall below the preset `order`.
    Returns: (latex_curr, latex_t, latex_g, leading_order, solver_length, result_order, warnings)
    """
    s, proc = ctx.solver, ctx.proc
    res_order = order if val_t.is_exact else min(val_t.order, order)
    latex_t = latex(safe_subs(val_t.expr, ctx.disp_subs))
    latex_g = latex(safe_subs(val_g.expr, ctx.disp_subs))
    latex_curr = latex_g if use_gain else latex_t

    try:
        leading_order = compute_val_leading_order(val_t, s, dc=ctx.is_dc, process=proc)
    except Exception:
        leading_order = S.Zero

    store = LAST_ASSERT_WM if assertion else LAST_EVAL_WM
    store.clear()
    store.update({'val_t': val_t, 'solver': s, 'order': res_order})
    if assertion:
        store.update({'val_g': val_g, 'disp_subs': ctx.disp_subs})

    return latex_curr, latex_t, latex_g, leading_order, len(s), res_order, ctx.warnings


def eval_wave_mixing_assertions(
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
    degenerate: bool = False,
    seed_mode: Optional[int] = None,
    seed_vac_c: bool = True,
    seed_observed: bool = False,
) -> Tuple[str, str, str, Any, int, Any, List[str]]:
    """
    Evaluates operator combination (optionally under input-state assertions):
    - coherent_drive: strong driving field input is in coherent state |alpha>
    - vacuum_signal: signal field (a_1) input is in vacuum |0>
    - vacuum_other_weak: remaining weak field (a_2 if is_dc else a_0) input is in vacuum |0>
    - seed_mode / seed_vac_c / seed_observed: coherent seeding of a separate weak mode
      (see apply_assertions)
    """
    ctx = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode,
        process=process, degenerate=degenerate,
    )
    val_t, val_g = apply_assertions(
        ctx.val, ctx.solver, order, process=ctx.proc, mismatch=ctx.mismatch, single_mode=single_mode,
        degenerate=ctx.degenerate, coherent_drive=coherent_drive, vacuum_signal=vacuum_signal, vacuum_other_weak=vacuum_other_weak,
        alpha=coherent_alpha(ctx), seed_mode=seed_mode, seed_vac_c=seed_vac_c,
        seed_observed=seed_observed,
    )
    assertion = coherent_drive or vacuum_signal or vacuum_other_weak or seed_mode is not None
    return finalize_wm_result(ctx, val_t, val_g, order, use_gain, assertion)


def eval_wave_mixing(
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
    degenerate: bool = False,
) -> Tuple[str, str, str, Any, int, Any, List[str]]:
    """Evaluates an operator combination without any input-state assertions."""
    return eval_wave_mixing_assertions(
        text, dc, order, use_gain, symbols, operators, mm_operators, named_expressions,
        mismatch=mismatch, single_mode=single_mode, process=process, degenerate=degenerate,
    )
