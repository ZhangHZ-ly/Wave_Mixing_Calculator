from __future__ import annotations
import ast
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from wave_mixing.dsolve_pm import d_p2s
from wave_mixing.pattern_form.op_patterns import BosonNum, BosonPat, FermionPat
from wave_mixing import *
from wave_mixing.wm_processor import t_, _g, nm, _ps, nlc, pm, a_p, a_s, a_i, _tdz
from wave_mixing.pattern_form import PF, PE, PF1
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


def get_wm_solver(process: Optional[str] = None, mismatch: bool = False, dc: Optional[bool] = None) -> WMProcessor:
    proc = normalize_process(process, dc)
    if proc == 'fwm':
        return FWM_SOLVER
    return TWM_SOLVERS[1 if proc == 'twm_dc' else 0][1 if mismatch else 0]


def _solver_process(s: WMProcessor) -> str:
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
        sol_d = s.get_sol(sol_order)[var_index]

        s0 = s.sol[0]
        if len(s0[var_index]) == 1:
            p = S.One
        elif var_index == 0:
            p = s.half_p_factor ** S.NegativeOne
        else:
            p = s.half_p_factor

        expr_t = p * s.to_expr(sol_d)
        sol_d_g = t_to_g(sol_d, process=proc, mismatch=bool(mismatch))
        p_g = t_to_g(p, process=proc, mismatch=bool(mismatch))
        expr_g = p_g * s.to_expr(sol_d_g)

    latex_t = latex(expr_t)
    latex_g = latex(expr_g)

    return latex_t, latex_g


class WmWrapper:
    """
    Algebraic wrapper for Tab 3 (Wave Mixing Operator Combinations).
    Dispatches addition, multiplication, exponentiation, and adjoint across:
    - PF (independent-mode _ind: strong-field initial op, custom ops, number ops)
    - Multimode dict ({tuple: PF}: A and balanced weak-field pairs)
    - Weak-field f_dict ({BosonOp: {tuple: PF}}: weak-field initial ops and evolved weak modes)
    """
    def __init__(self, val, solver: WMProcessor, _factors: Optional[List[Any]] = None):
        self.solver = solver
        if _factors is not None:
            self._factors = [f.val if isinstance(f, WmWrapper) else f for f in _factors]
            self._val = self._factors[0] if len(self._factors) == 1 else None
        else:
            raw_val = val.val if isinstance(val, WmWrapper) else val
            self._factors = [raw_val]
            self._val = raw_val
        self.name: str = ""
        self.val_t = self._val
        self.val_g = None
        self.latex_t: Optional[str] = None
        self.latex_g: Optional[str] = None
        self.order: Optional[int] = None
        self.leading_order: Optional[Union[int, float]] = None
        self.dc: Optional[bool] = solver.dc if solver else True
        self.process: Optional[str] = _solver_process(solver) if solver else 'twm_dc'
        self.single_mode: Optional[bool] = False
        self.notes: str = ""
        self._is_g: bool = False
        self._is_t: bool = False

    @property
    def val(self):
        if self._val is None:
            raw_factors = [f.val if isinstance(f, WmWrapper) else f for f in self._factors]
            self._val = self.solver.op_prod(raw_factors)
            self._factors = [self._val]
        return self._val

    @classmethod
    def from_val(cls, val, solver: Any) -> 'WmWrapper':
        if isinstance(val, cls):
            return val
        s = solver if isinstance(solver, WMProcessor) else TWM_SOLVERS[1 if bool(solver) else 0][0]
        raw_val = val.val if isinstance(val, WmWrapper) else val
        return cls(s.from_expr(raw_val), s)

    def to_single_mode(self) -> 'WmWrapper':
        """Projects this wrapper's value into single-mode PF representation."""
        return WmWrapper(self.solver.to_single_mode(self.val), self.solver)

    def _to_mm_dict(self):
        return self.solver.to_dict(self.val)

    def __add__(self, other):
        other = WmWrapper.from_val(other, self.solver)
        return WmWrapper(self.solver.op_add(self.val, other.val), self.solver)

    def __radd__(self, other):
        return WmWrapper.from_val(other, self.solver).__add__(self)

    def __neg__(self):
        return WmWrapper(self.solver.op_neg(self.val), self.solver)

    def __sub__(self, other):
        other = WmWrapper.from_val(other, self.solver)
        return WmWrapper(self.solver.op_add(self.val, self.solver.op_neg(other.val)), self.solver)

    def __rsub__(self, other):
        return WmWrapper.from_val(other, self.solver).__sub__(self)

    def __mul__(self, other):
        other = WmWrapper.from_val(other, self.solver)
        return WmWrapper(None, self.solver, _factors=self._factors + other._factors)

    def __rmul__(self, other):
        return WmWrapper.from_val(other, self.solver).__mul__(self)

    def __matmul__(self, other):
        other = WmWrapper.from_val(other, self.solver)
        return WmWrapper(self.solver.op_mul(self.val, other.val, single_mode=True), self.solver)

    def __rmatmul__(self, other):
        other = WmWrapper.from_val(other, self.solver)
        return WmWrapper(self.solver.op_mul(other.val, self.val, single_mode=True), self.solver)

    def __truediv__(self, other):
        other = WmWrapper.from_val(other, self.solver)
        return WmWrapper(self.solver.op_div(self.val, other.val), self.solver)

    def __rtruediv__(self, other):
        return WmWrapper.from_val(other, self.solver).__truediv__(self)

    def __pow__(self, power):
        if isinstance(power, WmWrapper):
            if isinstance(power.val, PF) and power.val.npf == S.Zero and set(power.val.table.keys()).issubset({((), ())}):
                power = power.val.table.get(((), ()), S.Zero)
            else:
                raise ValueError("Exponent must be a scalar or number-operator expression.")
        p_sym = sympify(power)
        if p_sym.is_Integer and p_sym >= 0:
            n = int(p_sym)
            f_list = [f for f in self._factors if isinstance(f, f_dict)]
            if f_list:
                parities = [self.solver.f_parity(f) for f in f_list]
                count_0 = sum(1 for p in parities if p == 0)
                count_1 = sum(1 for p in parities if p == 1)
                if count_0 != count_1:
                    return WmWrapper(None, self.solver, _factors=self._factors * n)
        return WmWrapper(self.solver.op_pow(self.val, p_sym), self.solver)

    def adjoint(self):
        adj_factors = [
            self.solver.op_adjoint(f.val if isinstance(f, WmWrapper) else f)
            for f in reversed(self._factors)
        ]
        return WmWrapper(None, self.solver, _factors=adj_factors)

    def diff(self, *symbols, **kwargs):
        s = self.solver
        if not symbols:
            raise ValueError("diff requires at least one variable.")
        var = symbols[0]
        rest_args = symbols[1:]

        is_wrt_g = False
        is_wrt_t = False

        if isinstance(var, WmWrapper):
            if var._is_g:
                is_wrt_g = True
            elif var._is_t:
                is_wrt_t = True
            elif isinstance(var.val, PF) and var.val.npf == S.Zero and set(var.val.table.keys()).issubset({((), ())}):
                var = var.val.table.get(((), ()), S.Zero)
            else:
                raise ValueError(f"Cannot differentiate with respect to operator: {var}")
        elif isinstance(var, str):
            if var == 't':
                is_wrt_t = True
            elif var in ('g', '_g', r'\hat{g}'):
                is_wrt_g = True
            elif var in ('pm', 'Δk', r'\Delta k'):
                var = pm
            elif var == 'nlc':
                var = nlc
            else:
                var = Symbol(var)
        elif var == t_ or str(var) == '_t':
            is_wrt_t = True
        elif var == g_hat or str(var) in ('g', '_g', r'\hat{g}'):
            is_wrt_g = True

        if is_wrt_g:
            order_n = 1
            if rest_args and isinstance(rest_args[0], int):
                order_n = rest_args[0]
                rest_args = rest_args[1:]
            var = t_
            gc_inv_n = s.gc ** (-order_n)
            def _d_g(leaf):
                if isinstance(leaf, PF):
                    sf = {}
                    for k, v in leaf.table.items():
                        dv = diff(v, var, order_n, *rest_args, **kwargs)
                        if dv != 0 and dv != S.Zero:
                            sf[k] = s.n_simp(dv * gc_inv_n)
                    if not sf and leaf.npf == S.Zero:
                        sf = {((), ()): S.Zero}
                    npf = s.n_simp(diff(leaf.npf, var, order_n, *rest_args, **kwargs) * gc_inv_n)
                    return PF(sf, npf)
                return s.n_simp(diff(leaf, var, order_n, *rest_args, **kwargs) * gc_inv_n)
            diff_val = d_apply(self.val, _d_g)
            return WmWrapper(diff_val, s)
        elif is_wrt_t:
            var = t_
            def _d_t(leaf):
                if isinstance(leaf, PF):
                    sf = {}
                    for k, v in leaf.table.items():
                        dv = diff(v, var, *rest_args, **kwargs)
                        if dv != 0 and dv != S.Zero:
                            sf[k] = dv
                    if not sf and leaf.npf == S.Zero:
                        sf = {((), ()): S.Zero}
                    npf = diff(leaf.npf, var, *rest_args, **kwargs)
                    return PF(sf, npf)
                return diff(leaf, var, *rest_args, **kwargs)
            diff_val = d_apply(self.val, _d_t)
            return WmWrapper(diff_val, s)
        else:
            def _d_gen(leaf):
                if isinstance(leaf, PF):
                    sf = {}
                    for k, v in leaf.table.items():
                        dv = diff(v, var, *rest_args, **kwargs)
                        if dv != 0 and dv != S.Zero:
                            sf[k] = dv
                    if not sf and leaf.npf == S.Zero:
                        sf = {((), ()): S.Zero}
                    npf = diff(leaf.npf, var, *rest_args, **kwargs)
                    return PF(sf, npf)
                return diff(leaf, var, *rest_args, **kwargs)
            diff_val = d_apply(self.val, _d_gen)
            return WmWrapper(diff_val, s)

    def Derivative(self, *symbols, **kwargs):
        return self.diff(*symbols, **kwargs)


WM_VARIABLES: Dict[str, WmWrapper] = {}
LAST_EVAL_WM: Dict[str, Any] = {}


def save_wm_variable(
    name: str,
    val_t: Any,
    val_g: Any = None,
    solver: Any = None,
    latex_t: Optional[str] = None,
    latex_g: Optional[str] = None,
    **kwargs
) -> WmWrapper:
    proc = normalize_process(kwargs.get('process'), kwargs.get('dc'))
    s = solver if isinstance(solver, WMProcessor) else get_wm_solver(proc, mismatch=bool(kwargs.get('mismatch', False)))
    w = WmWrapper.from_val(val_t, s)
    w.name = name
    w.val_t = w.val
    w.val_g = val_g if val_g is not None else t_to_g(
        w.val,
        process=kwargs.get('process', _solver_process(s)),
        mismatch=bool(kwargs.get('mismatch', False)),
    )
    w.latex_t = latex_t
    w.latex_g = latex_g
    for k, v in kwargs.items():
        setattr(w, k, v)
    WM_VARIABLES[name] = w
    return w


def save_latest_wm_variable(name: str, **kwargs) -> Optional[WmWrapper]:
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


def get_wm_variable(name: str) -> Optional[WmWrapper]:
    return WM_VARIABLES.get(name)


def clear_wm_variables() -> None:
    WM_VARIABLES.clear()
    LAST_EVAL_WM.clear()


def _format_order_num(val) -> Union[int, float]:
    f = float(val)
    return int(f) if f.is_integer() else f


def compute_val_leading_order(
    val: Any,
    s: WMProcessor,
    dc: Optional[bool] = None,
    process: Optional[str] = None,
) -> Union[int, float]:
    """
    Computes the leading perturbation order of a calculated WM result value `val`.
    Uses s.pe which includes base assumptions (-1 for TWM strong mode, -1/2 for FWM pump)
    and any custom symbol/operator orders.
    """
    proc = normalize_process(process if process is not None else _solver_process(s), dc)
    is_dc = process_is_dc(proc)
    strong_name = a_p.name if is_dc else a_i.name
    strong_ord = -S.Half if proc == 'fwm' else S.NegativeOne
    pe = s.pe

    orders: List[Any] = []

    def _sub_t_for_order(expr_t: Expr) -> Expr:
        if not expr_t.has(t_):
            return expr_t
        if proc == 'fwm' and not expr_t.has(pm):
            return expr_t.subs({t_: _g / (nlc * s.nd)})
        return expr_t.subs({t_: _g / s.gc})

    def _inspect_pf(p: PF, extra_ord=S.Zero):
        if p.npf != 0 and p.npf is not S.Zero:
            try:
                npf_ord = pe.leading_order(_sub_t_for_order(p.npf))[1]
                if npf_ord is not S.Infinity:
                    orders.append(npf_ord + extra_ord)
            except Exception:
                pass
        for (bs, fs), t_expr in p.table.items():
            if t_expr == 0 or t_expr is S.Zero:
                continue
            bs_ord = S.Zero
            for b in bs:
                b_name = b.name
                if b_name in pe.assumptions:
                    bs_ord += sympify(pe.assumptions[b_name]) * int(sum(b.exps))
                elif b_name == strong_name:
                    bs_ord += strong_ord * int(sum(b.exps))
            fs_ord = S.Zero
            for f in fs:
                f_name = f.name
                if f_name in pe.assumptions:
                    fs_ord += sympify(pe.assumptions[f_name]) * int(sum(f.exps))
            try:
                sub_t = _sub_t_for_order(t_expr)
                t_ord = pe.leading_order(sub_t)[1]
                if t_ord is not S.Infinity:
                    orders.append(t_ord + bs_ord + fs_ord + extra_ord)
            except Exception:
                pass

    def _walk(obj, extra_ord=S.Zero):
        if isinstance(obj, PF):
            _inspect_pf(obj, extra_ord)
        elif isinstance(obj, f_dict):
            for b, v in obj.items():
                b_ord = pe.leading_order(b)[1]
                _walk(v, extra_ord + b_ord)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v, extra_ord)

    _walk(val)
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
    Analyzes an expression across each additive term:
    - For a product F_1 * F_2 * ... * F_m with leading orders L_1, ..., L_m,
      each estimated factor F_k (field operator evolution or saved named expression)
      requires accuracy A_k >= target_order - sum_{j != k} L_j.
    - Returns (max_evolved_needed_int, insufficient_items), where insufficient_items
      lists any custom saved expressions (and, under FWM, evolved field operators capped at order 1)
      whose available order is lower than required.
    """
    proc = normalize_process(process, dc)
    is_dc = process_is_dc(proc)
    strong_lo: float = -0.5 if proc == 'fwm' else -1.0
    strong_nd_lo: float = 2.0 * strong_lo  # -1.0 for FWM, -2.0 for TWM

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

    named_map = {}
    if named_expressions:
        for ne in named_expressions:
            name = (ne.get('name') or '').strip()
            if name:
                named_map[name] = ne

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
        # (is_exact, leading_order, category, ref)
        if isinstance(node, ast.Constant):
            return True, 0.0, 'exact', None
        if isinstance(node, ast.Name):
            name = node.id
            if name in named_map:
                ne = named_map[name]
                raw_lo = ne.get('leading_order')
                if raw_lo is None:
                    raw_lo = ne.get('leadingOrder', 0)
                return False, float(raw_lo or 0), 'named', ne
            if name in WM_VARIABLES:
                cvar = WM_VARIABLES[name]
                lo = float(cvar.leading_order if cvar.leading_order is not None else 0)
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
            if func_name in ('dag', 'Dagger', 'dagger', 'diff', 'Derivative'):
                if node.args:
                    return get_factor_info(node.args[0])
            if func_name in ('N_', 'BosonNum'):
                if node.args:
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
                if has_evolved:
                    return False, lo, 'evolved', evolved_ref
                return True, lo, 'exact', None
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
                        is_ex, _, cat, ref = get_factor_info(fn)
                        if cat == 'evolved':
                            has_evolved = True
                            evolved_ref = ref
                lo = min_lo * exp_val
                if has_evolved:
                    return False, lo, 'evolved', evolved_ref
                return True, lo, 'exact', None
            else:
                base_exact, base_lo, cat, ref = get_factor_info(node.left)
                return base_exact, base_lo * exp_val, cat, ref
        return True, 0.0, 'exact', None

    def expand_additive_products(node) -> List[List[Tuple[Any, int]]]:
        # Expand AST into additive terms of [(factor_node, sign_multiplier_for_lo)],
        # distributing *, /, **, and dag() over parenthesized sums (capped at 64 terms).
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
                    # In TWM, strong driving modes evolve at odd orders: -1, 1, 3, 5, ...
                    needed_for_op = -1 if req_order_int <= 0 else (req_order_int if req_order_int % 2 == 1 else req_order_int - 1)
                else:
                    # In TWM, weak signal/idler modes evolve at even orders: 0, 2, 4, ...
                    needed_for_op = -1 if req_order_int < 0 else (req_order_int if req_order_int % 2 == 0 else req_order_int - 1)

                if needed_for_op > max_evolved_needed:
                    max_evolved_needed = needed_for_op
            elif cat == 'named':
                rec_order = ref.get('order')
                if rec_order is not None and float(rec_order) < req_order_raw - 1e-12:
                    name = ref.get('name', '')
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
    If no corresponding variable is stored in local_dict, it is treated as invalid input.
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

    # Automatically analyze required order for evolved field operators
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
        # FWMSolver supports orders -1, 0, 1; cap at 1 when user force-continues higher-order requests
        evolved_order = min(1, evolved_order)
    s._evolved_order = evolved_order
    sol = s.get_sol(evolved_order)

    # In Page 3 FWM mode, include outer SPM phase factor exp(-i * χ * N_0 * t) on evolved pump mode a_p
    if proc == 'fwm' and evolved_order >= 0:
        spm_dict = {(None,): PF({((), ()): exp(-I * nlc * s.nd * t_)})}
        sol = (s.e_mul(sol[0], spm_dict), sol[1], sol[2])

    # Initial-time (t=0) mode wrappers directly from INIT_OPS[is_dc]
    init_wrappers = {k: WmWrapper(v, s) for k, v in INIT_OPS[is_dc].items()}
    evolved_wrappers = [WmWrapper(s_op, s) for s_op in sol]

    # Map operator identifiers -> mode name ('a_0', 'a_1', 'a_2') for N_(...)
    op_mode_map = {
        'a_0': 'a_0', 'a_p': 'a_0',
        'a_1': 'a_1', 'a_s': 'a_1',
        'a_2': 'a_2', 'a_i': 'a_2',
    }
    mode_kind_map = {
        'a_0': 'boson', 'a_1': 'boson', 'a_2': 'boson',
        'a_p': 'boson', 'a_s': 'boson', 'a_i': 'boson'
    }

    # Extract custom orders from symbols and operators to update s.pe
    custom_orders = {}
    for sym_def in symbols:
        s_name = (sym_def.get('name') or '').strip()
        if s_name and 'order' in sym_def and sym_def['order'] is not None:
            try:
                ord_v = sympify(sym_def['order'])
                custom_orders[s_name] = ord_v
                custom_orders[Symbol(s_name)] = ord_v
            except Exception:
                pass
    for op_def in operators:
        op_name = (op_def.get('name') or '').strip()
        if op_name and 'order' in op_def and op_def['order'] is not None:
            try:
                ord_v = sympify(op_def['order'])
                custom_orders[op_name] = ord_v
                custom_orders[Symbol(op_name)] = ord_v
            except Exception:
                pass

    if custom_orders:
        s.pe.assumptions.update(custom_orders)

    def _resolve_mode_name(item) -> str:
        if isinstance(item, WmWrapper):
            for idx, ew in enumerate(evolved_wrappers):
                if item.val is ew.val:
                    return f'a_{idx}'
            for k in ('a_0', 'a_1', 'a_2'):
                if item.val is init_wrappers[k].val:
                    return k
            if isinstance(item.val, f_dict) and len(item.val) == 1:
                name = str(next(iter(item.val.keys())).name)
            elif isinstance(item.val, PF):
                name = str(item.val.expr())
            else:
                name = str(s.to_expr(item.val))
        elif hasattr(item, 'name'):
            name = str(item.name)
        else:
            name = str(item)
        return op_mode_map.get(name, name)

    def _unwrap_scalar(x):
        if isinstance(x, WmWrapper):
            if isinstance(x.val, PF) and x.val.npf == S.Zero and set(x.val.table.keys()).issubset({((), ())}):
                return x.val.table.get(((), ()), S.Zero)
            return x.val.expr() if isinstance(x.val, PF) else s.to_expr(x.val)
        return sympify(x)

    def _make_boson_num(item):
        return WmWrapper.from_val(BosonNum(_resolve_mode_name(item)), s)

    def _make_mmn(name, disp=0):
        return WmWrapper.from_val(MultimodeNum(_resolve_mode_name(name), _unwrap_scalar(disp)), s)

    def _make_mmn2(disp=0):
        d_val = _unwrap_scalar(disp)
        base = MultimodeNum2(a_s.name, a_i.name, 0) if is_dc else MultimodeNum2(a_p.name, a_s.name, 1)
        val = base + d_val * nm if d_val != 0 else base
        return WmWrapper.from_val(val, s)

    def _make_pat(item, *args):
        raw_args = [a if isinstance(a, bool) else _unwrap_scalar(a) for a in args]

        is_A = isinstance(item, WmWrapper) and (item is init_wrappers['A'] or item.val == INIT_OPS[is_dc]['A'])
        is_dagA = (isinstance(item, WmWrapper) and isinstance(item.val, dict) and not isinstance(item.val, f_dict)
                   and list(item.val.keys()) == [(False, 1)] and item.val[(False, 1)].table == {((), ()): S.One})
        if is_A or is_dagA:
            is_ann = True if is_A else False
            exps = list(raw_args)
            if exps and isinstance(exps[0], bool):
                is_ann = exps.pop(0)
            if not exps:
                exps = [1]
            int_exps = [int(sympify(e)) for e in exps]
            if any(e < 0 for e in int_exps):
                raise ValueError("Pattern exponents must be non-negative integers.")
            return WmWrapper({_tdz(is_ann, int_exps): PF1}, s)

        if isinstance(item, WmWrapper) and isinstance(item.val, (f_dict, dict)):
            exps = list(raw_args)
            explicit_ann = exps.pop(0) if (exps and isinstance(exps[0], bool)) else None
            if not exps:
                exps = [1]
            if isinstance(item.val, f_dict) and len(item.val) == 1:
                b_first = next(iter(item.val.keys()))
                op_ann = item if b_first.is_annihilation else item.adjoint()
                is_a = bool(b_first.is_annihilation) if explicit_ann is None else bool(explicit_ann)
            else:
                op_ann = item
                is_a = True if explicit_ann is None else bool(explicit_ann)
            res = WmWrapper(PF1, s)
            for e in exps:
                cur = op_ann if is_a else op_ann.adjoint()
                res = res * (cur ** int(sympify(e)))
                is_a = not is_a
            return res

        val = item.val if isinstance(item, WmWrapper) else item
        if isinstance(val, PF) and val.table:
            bs, fs = next(iter(val.table.keys()))
            op = bs[0] if bs else fs[0]
        else:
            op = val

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

        exps = list(raw_args)
        if exps and isinstance(exps[0], bool):
            is_ann = exps.pop(0)
        if not exps:
            exps = [1]

        is_fermion = (mode_kind_map.get(name) == 'fermion') if mode_kind_map else isinstance(inner, (FermionOp, FermionPat))
        if is_fermion:
            return WmWrapper.from_val(FermionPat(name, is_ann, *exps), s)

        bp = BosonPat(name, is_ann, *exps)
        a_op, tail = bp.num_repr()
        b_key = (a_op,) if a_op is not S.One else ()
        return WmWrapper(PF({(b_key, ()): tail}), s)

    def _make_bp(item, *args):
        raw_args = [a if isinstance(a, bool) else _unwrap_scalar(a) for a in args]
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        return WmWrapper.from_val(BosonPat(name, *raw_args), s)

    def _make_fp(item, *args):
        raw_args = [a if isinstance(a, bool) else _unwrap_scalar(a) for a in args]
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        return WmWrapper.from_val(FermionPat(name, *raw_args), s)

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, WmWrapper):
                if isinstance(item.val, PF):
                    if item.val.npf == S.Zero and set(item.val.table.keys()).issubset({((), ())}):
                        arg = item.val.table.get(((), ()), S.Zero)
                        return WmWrapper(PF({((), ()): sympify(fn(arg))}), s)
                elif isinstance(item.val, dict) and not isinstance(item.val, f_dict):
                    res_dict = s.expand_func_d(fn, evolved_order, item.val)
                    return WmWrapper(res_dict, s)
            raw_item = item.val if isinstance(item, WmWrapper) else item
            return WmWrapper.from_val(fn(sympify(raw_item)), s)
        return _inner

    # Represent g in terms of s.gc * t_ during algebraic evaluation so NumShift on N_d works accurately
    w_g_internal = WmWrapper(PF({((), ()): s.gc * t_}), s)
    w_g_internal._is_g = True

    w_t = WmWrapper.from_val(t_, s)
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
        'dag': lambda item: WmWrapper.from_val(item, s).adjoint(),
        'Dagger': lambda item: WmWrapper.from_val(item, s).adjoint(),
        'dagger': lambda item: WmWrapper.from_val(item, s).adjoint(),
        'N_': _make_boson_num,
        'BosonNum': _make_boson_num,
        'Pat': _make_pat,
        'pattern': _make_pat,
        'BP': _make_bp,
        'BosonPat': _make_bp,
        'FP': _make_fp,
        'FermionPat': _make_fp,
        'MMN': _make_mmn,
        'MMN2': _make_mmn2,
        't': w_t,
        'g': w_g_internal,
        'diff': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'Derivative': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'nlc': WmWrapper.from_val(nlc, s),
        'pm': WmWrapper.from_val(pm_internal, s),
        'n_m': WmWrapper.from_val(S.One if single_mode else nm, s),
        # Initial (t=0) operators & multimode presets
        'a_0': init_wrappers['a_0'],
        'a_1': init_wrappers['a_1'],
        'a_2': init_wrappers['a_2'],
        'A': init_wrappers['A'],
        'N_t': init_wrappers['N_t'],
        # Evolved operators at time t
        'a_p': evolved_wrappers[0],
        'a_s': evolved_wrappers[1],
        'a_i': evolved_wrappers[2],
    }

    disp_subs: Dict[Any, Any] = {}

    # 1. Register user symbols (including customized nlc / pm identifiers)
    for sym_def in symbols:
        s_name = (sym_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(s_name)
        if not valid:
            raise ValueError(f"Invalid symbol identifier: {err_msg}")
        d_name = (sym_def.get('display_name') or '').strip()

        if s_name == 'nlc' or sym_def.get('builtin_key') == 'nlc':
            local_dict[s_name] = WmWrapper.from_val(nlc, s)
            if d_name and d_name not in ('nlc', r'\chi', 'χ'):
                disp_subs[nlc] = Symbol(d_name)
            continue
        if s_name == 'pm' or sym_def.get('builtin_key') == 'pm':
            if not mismatch:
                local_dict[s_name] = WmWrapper.from_val(pm_internal, s)
            else:
                local_dict[s_name] = WmWrapper.from_val(pm, s)
                if d_name and d_name not in ('pm', 'Δk', r'\Delta k'):
                    disp_subs[pm] = Symbol(d_name)
            continue

        kwargs = {}
        if sym_def.get('is_positive'): kwargs['positive'] = True
        elif sym_def.get('is_nonnegative'): kwargs['nonnegative'] = True
        elif sym_def.get('is_integer'): kwargs['integer'] = True
        elif sym_def.get('is_real'): kwargs['real'] = True
        sym_obj = Symbol(s_name, **kwargs)
        local_dict[s_name] = WmWrapper.from_val(sym_obj, s)
        if d_name and d_name != s_name:
            disp_subs[sym_obj] = Symbol(d_name)

    # 2. Register evolved mode operators (a_p, a_s, a_i) and custom boson/fermion modes
    for op_def in operators:
        op_name = (op_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(op_name)
        if not valid:
            raise ValueError(f"Invalid operator identifier: {err_msg}")

        op_type = op_def.get('type')
        if op_type == 'operator' or ('kind' in op_def and 'mode_index' not in op_def):
            # Custom Boson or Fermion mode
            kind = op_def.get('kind', 'boson')
            mode_kind_map[op_name] = kind
            op_mode_map[op_name] = op_name
            op_obj = BosonOp(op_name) if kind == 'boson' else FermionOp(op_name)
            w_op = WmWrapper.from_val(op_obj, s)
            local_dict[op_name] = w_op

            d_name = (op_def.get('display_name') or '').strip()
            if d_name and d_name != op_name:
                disp_subs[Symbol(op_name)] = Symbol(d_name)
        else:
            # Evolved operator mode (pump=0, signal=1, idler=2)
            mode_idx = int(op_def.get('mode_index', 0))
            if 0 <= mode_idx <= 2:
                w_op = evolved_wrappers[mode_idx]
                local_dict[op_name] = w_op

                op_mode_map[op_name] = f'a_{mode_idx}'
                mode_kind_map[op_name] = 'boson'
                mode_kind_map[f'a_{mode_idx}'] = 'boson'

                d_name = (op_def.get('display_name') or '').strip()
                default_mode_name = f'a_{mode_idx}'
                if d_name and d_name != default_mode_name:
                    disp_subs[Symbol(default_mode_name)] = Symbol(d_name)

    # 3. Register multimode operators (A, N_t or renamed identifiers)
    for mm_def in mm_operators:
        mm_name = (mm_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(mm_name)
        if not valid:
            raise ValueError(f"Invalid multimode operator identifier: {err_msg}")
        mm_kind = mm_def.get('mm_kind', 'COP')
        w_mm = init_wrappers['A'] if mm_kind == 'COP' else init_wrappers['N_t']
        local_dict[mm_name] = w_mm

    # 4. Register saved named expressions
    for ne in named_expressions:
        ne_name = (ne.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")

        val_obj = ne.get('val')
        ne_proc = normalize_process(ne.get('process'), ne.get('dc', is_dc))

        if val_obj is not None:
            w_ne = val_obj if isinstance(val_obj, WmWrapper) else WmWrapper(val_obj, s)
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
            save_wm_variable(
                ne_name,
                w_ne.val,
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
            w_cached = WM_VARIABLES[ne_name]
            w_ne = WmWrapper(w_cached.val, s)
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
        elif ne_name in PF_VARIABLES:
            p_cached = PF_VARIABLES[ne_name]
            w_ne = WmWrapper(p_cached, s)
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
                w_ne = WmWrapper.from_val(parsed_ne, s)
                if single_mode:
                    w_ne = w_ne.to_single_mode()
                local_dict[ne_name] = w_ne
                save_wm_variable(
                    ne_name,
                    w_ne.val,
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

    # 5. Parse and evaluate target expression
    try:
        validate_input_identifiers(text, local_dict)
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=WM_TRANSFORMS)
        wrapper_target = WmWrapper.from_val(parsed_target, s)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    return wrapper_target, s, disp_subs, proc, is_dc


def map_pf(e, fn):
    if isinstance(e, f_dict):
        return f_dict({k: map_pf(v, fn) for k, v in e.items()}, b_left=e.b_left)
    if isinstance(e, dict):
        return {k: fn(v) for k, v in e.items()}
    if isinstance(e, PF):
        return fn(e)
    raise TypeError(f"Expected operator container (f_dict, dict, PF), got {type(e)}")


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
    wrapper_target, s, disp_subs, proc, is_dc = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode, process=process
    )
    val = wrapper_target.val

    # 6. Expand up to `order` and simplify via product-to-sum (d_p2s) + n_simp
    def _postprocess_pf(p: PF) -> PF:
        if not p.table and p.npf == S.Zero:
            return p
        p = s.expand(p, order)
        if order >= 0 or p.expr(False).has(t_):
            return p.n_apply(
                lambda t: s.n_simp(d_p2s(_g, s.p2s_group(t)).subs({_g: s.gc * t_}))
            )
        return p.n_apply(s.n_simp)

    val = map_pf(val, _postprocess_pf)

    # 6.1 If FWM + Phase Matched (FWM_zq: SPM compensated by phase mismatch), apply s._zq
    if proc == 'fwm' and not mismatch:
        val = s._zq(val)

    # 6.2 If single-mode processing is requested, convert multimode series to mode_expr with n_m = 1
    if single_mode:
        val = s.to_single_mode(val)

    # 7. Compute both time (t) and gain (g) representations
    val_t = val
    expr_result_t = val_t.expr(False) if isinstance(val_t, PF) else s.to_expr(val_t)
    expr_disp_t = expr_result_t
    if disp_subs:
        try:
            expr_disp_t = expr_result_t.subs(disp_subs)
        except Exception:
            expr_disp_t = expr_result_t
    latex_t = latex(expr_disp_t)

    val_g = t_to_g(val, process=proc, mismatch=mismatch)
    expr_result_g = val_g.expr(False) if isinstance(val_g, PF) else s.to_expr(val_g)
    expr_disp_g = expr_result_g
    if disp_subs:
        try:
            expr_disp_g = expr_result_g.subs(disp_subs)
        except Exception:
            expr_disp_g = expr_result_g
    latex_g = latex(expr_disp_g)

    latex_curr = latex_g if use_gain else latex_t

    try:
        leading_order = compute_val_leading_order(val_t, s, dc=is_dc, process=proc)
    except Exception:
        leading_order = 0

    sol_len = len(s.sol)

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
    Evaluates operator combination under selected input-state assertions:
    - coherent_drive: strong driving field input is in coherent state |alpha>
    - vacuum_signal: signal field (a_1) input is in vacuum |0>
    - vacuum_other_weak: remaining weak field (a_2 if is_dc else a_0) input is in vacuum |0>
    """
    wrapper_target, s, disp_subs, proc, is_dc = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode, process=process
    )
    val = wrapper_target.val

    vac_modes = []
    if vacuum_signal:
        vac_modes.append('a_1')
    if vacuum_other_weak:
        vac_modes.append('a_2' if is_dc else 'a_0')

    if vac_modes:
        val = s.eval_vacuum(val, vac_modes)
    if single_mode:
        val = s.to_single_mode(val)

    if coherent_drive:
        val_t = s.eval_coherent(val, order)
        if proc == 'fwm' and not mismatch:
            alpha_sym = Symbol('alpha_0')
            pm_cohr = 2 * nlc * (Abs(alpha_sym) ** 2)
            val_t = d_apply(val_t, lambda x: bottom_up(x.subs({s.pm: pm_cohr}), _ps))
        val_g = t_to_g(val_t, process=proc, mismatch=mismatch, coherent=True)
    else:
        def _postprocess_pf(p: PF) -> PF:
            if not p.table and p.npf == S.Zero:
                return p
            p = s.expand(p, order)
            if order >= 0 or p.expr(False).has(t_):
                return p.n_apply(
                    lambda t: s.n_simp(d_p2s(_g, s.p2s_group(t)).subs({_g: s.gc * t_}))
                )
            return p.n_apply(s.n_simp)

        val = map_pf(val, _postprocess_pf)
        if proc == 'fwm' and not mismatch:
            val = s._zq(val)
        val_t = val
        val_g = t_to_g(val, process=proc, mismatch=mismatch)

    expr_result_t = val_t.expr(False) if isinstance(val_t, PF) else s.to_expr(val_t)
    expr_result_g = val_g.expr(False) if isinstance(val_g, PF) else s.to_expr(val_g)

    expr_disp_t = expr_result_t
    expr_disp_g = expr_result_g
    if disp_subs:
        try:
            expr_disp_t = expr_result_t.subs(disp_subs)
        except Exception:
            expr_disp_t = expr_result_t
        try:
            expr_disp_g = expr_result_g.subs(disp_subs)
        except Exception:
            expr_disp_g = expr_result_g

    latex_t = latex(expr_disp_t)
    latex_g = latex(expr_disp_g)
    latex_curr = latex_g if use_gain else latex_t

    try:
        leading_order = compute_val_leading_order(val_t, s, dc=is_dc, process=proc)
    except Exception:
        leading_order = 0

    sol_len = len(s.sol)

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

