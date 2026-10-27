from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Tuple

from dsolve_pm import d_p2s
from op_patterns import BosonNum, BosonPat, FermionPat
from parametric import (
    TWMSolver, f_dict, d_apply, t_, _g, nlc, pm,
    a_p, a_s, a_i, MultimodeNum, MultimodeNum2, MultimodeCOP
)
from perturb_eval import pattern_form, PF, PE, Pat
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.numbers import I, Integer, pi
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

from .common import (
    TRANSFORMS,
    is_valid_variable_name,
    expr_to_code_str,
    g_hat,
)
from .twm_transforms import t_to_g, g_to_t

# Cached TWMSolver instances indexed by [dc][mismatch]
SOLVERS = (
    (TWMSolver(False, False), TWMSolver(False, True)),
    (TWMSolver(True, False), TWMSolver(True, True)),
)


def eval_twm_solve(
    dc: bool,
    mismatch: bool,
    var_index: int,
    order: int,
    use_gain: bool = False
) -> Tuple[str, str, str, str, str, str]:
    """
    Evaluates Three-Wave Mixing Heisenberg evolution for Tab 3.
    Returns: (latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g)
    """
    s = SOLVERS[bool(dc)][bool(mismatch)]
    sol_order = max(-1, order - 1)
    sol_d = s.get_sol(sol_order)[var_index]

    s0 = s.sol[0]
    if len(s0[var_index]) == 1:
        p = S.One
    elif var_index == 0:
        p = s.half_p_factor ** S.NegativeOne
    else:
        p = s.half_p_factor

    # Time representation (t)
    expr_t = p * s.to_expr(sol_d)
    latex_t = latex(expr_t)
    expr_str_t = expr_to_code_str(expr_t, dc=bool(dc))

    # Parametric gain representation (g)
    sol_d_g = t_to_g(sol_d, dc=bool(dc), mismatch=bool(mismatch))
    p_g = t_to_g(p, dc=bool(dc), mismatch=bool(mismatch))
    expr_g = p_g * s.to_expr(sol_d_g)
    latex_g = latex(expr_g)
    expr_str_g = expr_to_code_str(expr_g, dc=bool(dc))

    latex_curr = latex_g if use_gain else latex_t
    expr_curr = expr_str_g if use_gain else expr_str_t

    return latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g


def _is_bare_weak(d) -> bool:
    """True if d is an f_dict representing a single bare weak-field operator."""
    if not isinstance(d, f_dict) or len(d) != 1:
        return False
    inner = next(iter(d.values()))
    if len(inner) != 1 or (None,) not in inner:
        return False
    pf = inner[(None,)]
    return pf.npf == S.Zero and pf.table == {((), ()): S.One}


def _is_pure_scalar_pf(pf) -> bool:
    """True if pf has only key ((), ()) and no non-commutative number operators."""
    if not isinstance(pf, PF) or pf.npf != S.Zero:
        return False
    if not set(pf.table.keys()).issubset({((), ())}):
        return False
    val = pf.table.get(((), ()), S.Zero)
    return not val.has(BosonNum, MultimodeNum, MultimodeNum2, g_hat)


def _apply_to_pfs(val, pf_fn):
    """Apply pf_fn(pf: PF) -> PF to all PF leaves in a PF, dict, or f_dict."""
    if isinstance(val, f_dict):
        return f_dict({k: _apply_to_pfs(v, pf_fn) for k, v in val.items()}, b_left=val.b_left)
    if isinstance(val, dict):
        return {k: _apply_to_pfs(v, pf_fn) for k, v in val.items()}
    if isinstance(val, PF):
        return pf_fn(val)
    return val


class ComboWrapper:
    """
    Algebraic wrapper for Tab 4 (Post-Evolution Operator Combinations).
    Dispatches addition, multiplication, exponentiation, and adjoint across:
    - PF (PFTableProcessor, independent-mode _ind)
    - Multimode dict ({tuple: PF})
    - Weak-field f_dict ({BosonOp: {tuple: PF}})
    """
    def __init__(self, val, solver: TWMSolver):
        self.val = val
        self.solver = solver

    @classmethod
    def from_val(cls, val, solver: TWMSolver) -> 'ComboWrapper':
        if isinstance(val, cls):
            return val
        if isinstance(val, (PF, dict, f_dict)):
            return cls(val, solver)
        expr = sympify(val)
        if not expr.has(BosonOp, BosonPat, FermionOp, FermionPat, MultimodeCOP):
            return cls(PF({((), ()): expr}), solver)
        pf = pattern_form(expr, as_dict=None)
        return cls(pf, solver)

    def _to_mm_dict(self):
        if isinstance(self.val, PF):
            return {(None,): self.val}
        return self.val

    def __add__(self, other):
        other = ComboWrapper.from_val(other, self.solver)
        v1, v2 = self.val, other.val
        if isinstance(v1, PF) and isinstance(v2, PF):
            return ComboWrapper(PF.add(v1, v2), self.solver)
        d1 = self._to_mm_dict()
        d2 = other._to_mm_dict()
        return ComboWrapper(self.solver.d_add(d1, d2), self.solver)

    def __radd__(self, other):
        return ComboWrapper.from_val(other, self.solver).__add__(self)

    def __neg__(self):
        return ComboWrapper(d_apply(self.val, lambda x: -x), self.solver)

    def __sub__(self, other):
        return self.__add__(-ComboWrapper.from_val(other, self.solver))

    def __rsub__(self, other):
        return ComboWrapper.from_val(other, self.solver).__sub__(self)

    def __mul__(self, other):
        other = ComboWrapper.from_val(other, self.solver)
        v1, v2 = self.val, other.val
        # 1. Pure scalar PF fast-path
        if _is_pure_scalar_pf(v1):
            c = v1.table.get(((), ()), S.Zero)
            return ComboWrapper(d_apply(v2, lambda t: c * t), self.solver)
        if _is_pure_scalar_pf(v2):
            c = v2.table.get(((), ()), S.Zero)
            return ComboWrapper(d_apply(v1, lambda t: t * c), self.solver)
        # 2. Both PF -> independent-mode PF multiplication
        if isinstance(v1, PF) and isinstance(v2, PF):
            return ComboWrapper(v1.mul_ind(v2), self.solver)
        # 3. Bare weak-field operator on the right of a PF or Multimode dict: M * b
        if not isinstance(v1, f_dict) and _is_bare_weak(v2):
            b = next(iter(v2.keys()))
            d1 = self._to_mm_dict()
            return ComboWrapper(f_dict({b: d1}, b_left=False), self.solver)
        # 4. General dict / f_dict / PF multiplication via solver.e_mul
        d1 = self._to_mm_dict()
        d2 = other._to_mm_dict()
        return ComboWrapper(self.solver.e_mul(d1, d2), self.solver)

    def __rmul__(self, other):
        return ComboWrapper.from_val(other, self.solver).__mul__(self)

    def __truediv__(self, other):
        other = ComboWrapper.from_val(other, self.solver)
        if isinstance(other.val, PF) and other.val.npf == S.Zero and set(other.val.table.keys()).issubset({((), ())}):
            denom = other.val.table.get(((), ()), S.One)
            inv_pf = PF({((), ()): denom ** S.NegativeOne})
            return self.__mul__(ComboWrapper(inv_pf, self.solver))
        raise ValueError("Division is only supported by scalar or number-operator expressions.")

    def __rtruediv__(self, other):
        return ComboWrapper.from_val(other, self.solver).__truediv__(self)

    def __pow__(self, power):
        if isinstance(power, ComboWrapper):
            if isinstance(power.val, PF) and power.val.npf == S.Zero and set(power.val.table.keys()).issubset({((), ())}):
                power = power.val.table.get(((), ()), S.Zero)
            else:
                raise ValueError("Exponent must be a scalar or number-operator expression.")
        p_sym = sympify(power)
        if isinstance(self.val, PF):
            if self.val.npf == S.Zero and set(self.val.table.keys()).issubset({((), ())}):
                base_expr = self.val.table.get(((), ()), S.Zero)
                return ComboWrapper(PF({((), ()): base_expr ** p_sym}), self.solver)
            if p_sym.is_Integer and p_sym >= 0:
                return ComboWrapper(self.val.power_ind(Integer(p_sym)), self.solver)
            raise ValueError("Operator exponentiation only supports non-negative integers.")
        if isinstance(self.val, dict) and not isinstance(self.val, f_dict):
            if p_sym.is_Integer and p_sym >= 0:
                n = int(p_sym)
                if n == 0:
                    return ComboWrapper({(None,): PF({((), ()): S.One})}, self.solver)
                res = self
                for _ in range(n - 1):
                    res = res * self
                return res
        raise ValueError("Exponentiation for this operator type only supports non-negative integers.")

    def adjoint(self):
        if isinstance(self.val, PF):
            return ComboWrapper(self.val.adjoint(res=False), self.solver)
        return ComboWrapper(self.solver.d_adjoint(self.val), self.solver)

    def diff(self, *symbols, **kwargs):
        s = self.solver
        dc = getattr(s, 'dc', True)
        mismatch = (getattr(s, 'pm', S.Zero) != S.Zero)
        if not symbols:
            raise ValueError("diff requires at least one variable.")
        var = symbols[0]
        rest_args = symbols[1:]

        is_wrt_g = False
        is_wrt_t = False

        if isinstance(var, ComboWrapper):
            if getattr(var, '_is_g', False):
                is_wrt_g = True
            elif getattr(var, '_is_t', False):
                is_wrt_t = True
            elif isinstance(var.val, PF) and var.val.npf == S.Zero and set(var.val.table.keys()).issubset({((), ())}):
                val_expr = var.val.table.get(((), ()), S.Zero)
                if val_expr == t_ or str(val_expr) == '_t':
                    is_wrt_t = True
                elif val_expr == g_hat or str(val_expr) in ('g', '_g', r'\hat{g}'):
                    is_wrt_g = True
                else:
                    var = val_expr
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
            val_in_g = t_to_g(self.val, dc=dc, mismatch=mismatch)
            def _d_g(leaf):
                if isinstance(leaf, PF):
                    sf = {}
                    for k, v in leaf.table.items():
                        dv = diff(v, g_hat, *rest_args, **kwargs)
                        if dv != 0 and dv != S.Zero:
                            sf[k] = dv
                    if not sf and leaf.npf == S.Zero:
                        sf = {((), ()): S.Zero}
                    npf = diff(leaf.npf, g_hat, *rest_args, **kwargs)
                    return PF(sf, npf)
                return diff(leaf, g_hat, *rest_args, **kwargs)
            diff_in_g = d_apply(val_in_g, _d_g)
            diff_in_t = g_to_t(diff_in_g, dc=dc, mismatch=mismatch)
            return ComboWrapper(diff_in_t, s)
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
            return ComboWrapper(diff_val, s)
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
            return ComboWrapper(diff_val, s)

    def Derivative(self, *symbols, **kwargs):
        return self.diff(*symbols, **kwargs)


def compute_val_leading_order(val: Any, s: TWMSolver) -> int:
    """
    Computes the leading (lowest) perturbation order of a calculated TWM result value `val`.
    Uses TWMSolver.pe which includes base assumptions and any custom symbol/operator orders.
    Driving mode: order -1 (a_p in dc=True; a_i in dc=False).
    Weak modes, symbols, multimode operators: order 0 (or custom order if defined).
    Nonlinear coupling (nlc): order 1.
    Driving photon number N_d: order -2.
    """
    dc = getattr(s, 'dc', True)
    strong_name = a_p.name if dc else a_i.name
    pe = getattr(s, 'pe', None)
    if pe is None:
        pe = PE({strong_name: S.NegativeOne, nlc: S.One})
        pe.register(MultimodeNum)
        pe.register(MultimodeNum2)
        pe.register(MultimodeCOP)

    orders: List[int] = []

    def _inspect_pf(p: PF, extra_ord: int = 0):
        if p.npf != 0 and p.npf is not S.Zero:
            try:
                npf_ord = pe.leading_order(p.npf.subs({t_: _g / s.gc}))[1]
                if npf_ord is not S.Infinity:
                    orders.append(int(npf_ord) + extra_ord)
            except Exception:
                pass
        for (bs, fs), t_expr in p.table.items():
            if t_expr == 0 or t_expr is S.Zero:
                continue
            bs_ord = 0
            for b in bs:
                b_name = getattr(b, 'name', None)
                if b_name == strong_name:
                    bs_ord += -sum(b.exps)
                elif b_name and b_name in pe.assumptions:
                    bs_ord += int(pe.assumptions[b_name]) * sum(b.exps)
            fs_ord = 0
            for f in fs:
                f_name = getattr(f, 'name', None)
                if f_name and f_name in pe.assumptions:
                    fs_ord += int(pe.assumptions[f_name]) * sum(f.exps)
            try:
                sub_t = t_expr.subs({t_: _g / s.gc})
                t_ord = pe.leading_order(sub_t)[1]
                if t_ord is not S.Infinity:
                    orders.append(int(t_ord) + bs_ord + fs_ord + extra_ord)
            except Exception:
                pass

    def _walk(obj, extra_ord: int = 0):
        if isinstance(obj, PF):
            _inspect_pf(obj, extra_ord)
        elif isinstance(obj, f_dict):
            for k, v in obj.items():
                _walk(v, extra_ord)
        elif isinstance(obj, dict):
            for k, v in obj.items():
                _walk(v, extra_ord)

    _walk(val)
    if not orders:
        return 0
    return min(orders)


# Backward compatibility alias
compute_val_lowest_order = compute_val_leading_order


def compute_leading_order(
    expr_str: str,
    dc: bool = True,
    operators: Optional[List[Dict[str, Any]]] = None,
    named_expressions: Optional[List[Dict[str, Any]]] = None,
) -> int:
    """
    Computes the leading perturbation order of an expression string.
    Driving mode: order -1 (a_p in dc=True; a_i in dc=False).
    Weak modes, symbols, multimode operators: order 0.
    Multiplying factors sums their orders (e.g. a_p * a_p -> -2).
    Adding terms takes the min across terms.
    """
    strong_mode_idx = 0 if dc else 2
    strong_names = {'a_p', 'a_0'} if dc else {'a_i', 'a_2'}

    if operators:
        for op in operators:
            name = (op.get('name') or '').strip()
            mode_idx = int(op.get('mode_index', 0))
            if mode_idx == strong_mode_idx and name:
                strong_names.add(name)

    all_strong = set(strong_names)

    named_leading_orders: Dict[str, int] = {}
    if named_expressions:
        for ne in named_expressions:
            ne_name = (ne.get('name') or '').strip()
            if not ne_name:
                continue
            for k in ('leading_order', 'leadingOrder', 'lowest_order', 'lowestOrder'):
                if k in ne and ne[k] is not None:
                    named_leading_orders[ne_name] = int(ne[k])
                    break

    def get_order(node) -> int:
        if isinstance(node, ast.Expression):
            return get_order(node.body)
        elif isinstance(node, ast.Constant):
            return 0
        elif isinstance(node, ast.Name):
            name = node.id
            if name in named_leading_orders:
                return named_leading_orders[name]
            if name in all_strong:
                return -1
            if name in ('nlc', 'chi', r'\chi', 'χ'):
                return 1
            return 0
        elif isinstance(node, ast.UnaryOp):
            return get_order(node.operand)
        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, (ast.Add, ast.Sub)):
                return min(get_order(node.left), get_order(node.right))
            elif isinstance(node.op, ast.Mult):
                return get_order(node.left) + get_order(node.right)
            elif isinstance(node.op, ast.Div):
                return get_order(node.left) - get_order(node.right)
            elif isinstance(node.op, ast.Pow):
                base_ord = get_order(node.left)
                if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                    return base_ord * node.right.value
                elif isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub) and isinstance(node.right.operand, ast.Constant):
                    return base_ord * (-node.right.operand.value)
                return base_ord
            return 0
        elif isinstance(node, ast.Call):
            func_name = ''
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            if func_name in ('dag', 'Dagger', 'dagger'):
                if node.args:
                    return get_order(node.args[0])
                return 0
            elif func_name in ('N_', 'BosonNum'):
                if node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        arg_name = arg.value
                        if arg_name in all_strong:
                            return -2
                        return 0
                    return 2 * get_order(arg)
                return 0
            elif func_name in ('diff', 'Derivative'):
                if node.args:
                    return get_order(node.args[0])
                return 0
            return 0
        return 0

    try:
        tree = ast.parse(expr_str, mode='eval')
        return get_order(tree)
    except Exception:
        return 0


# Backward compatibility alias
compute_lowest_order = compute_leading_order


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
) -> Tuple[str, str, str, str, str, str, int]:
    """
    Evaluates post-evolution operator combinations in Tab 3 (formerly Tab 4).
    Returns: (latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g, leading_order)
    """
    text = (text or '').strip()
    if not text:
        raise ValueError("Expression input cannot be empty.")

    dc = bool(dc)
    mismatch = bool(mismatch)
    s = SOLVERS[dc][mismatch]
    sol = s.get_sol(max(-1, order))

    # Initial-time (t=0) mode wrappers (driving mode as PF, weak modes as f_dict)
    w_a0 = ComboWrapper.from_val(a_p if dc else f_dict({a_p: {(None,): PF({((), ()): S.One})}}), s)
    w_a1 = ComboWrapper.from_val(f_dict({a_s: {(None,): PF({((), ()): S.One})}}), s)
    w_a2 = ComboWrapper.from_val(f_dict({a_i: {(None,): PF({((), ()): S.One})}}) if dc else a_i, s)

    # Default MultimodeCOP (A) and MultimodeNum2 (N_t)
    w_A = ComboWrapper({(True, 1): PF({((), ()): S.One})}, s)
    mmn2_zero = MultimodeNum2(a_s.name, a_i.name, 0) if dc else MultimodeNum2(a_p.name, a_s.name, 1)
    w_Nt = ComboWrapper.from_val(mmn2_zero, s)

    # Evolved operator wrappers at time t
    evolved_wrappers = [
        ComboWrapper(sol[0], s),
        ComboWrapper(sol[1], s),
        ComboWrapper(sol[2], s),
    ]

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
                ord_v = int(sym_def['order'])
                custom_orders[s_name] = ord_v
                custom_orders[Symbol(s_name)] = ord_v
            except (ValueError, TypeError):
                pass
    for op_def in operators:
        op_name = (op_def.get('name') or '').strip()
        if op_name and 'order' in op_def and op_def['order'] is not None:
            try:
                ord_v = int(op_def['order'])
                custom_orders[op_name] = ord_v
                custom_orders[Symbol(op_name)] = ord_v
            except (ValueError, TypeError):
                pass

    if hasattr(s, 'update_pe'):
        s.update_pe(custom_orders)

    def _make_boson_num(item):
        if isinstance(item, ComboWrapper):
            for idx, ew in enumerate(evolved_wrappers):
                if item.val is ew.val:
                    return ComboWrapper.from_val(BosonNum(f'a_{idx}'), s)
            if item.val is w_a0.val:
                return ComboWrapper.from_val(BosonNum('a_0'), s)
            if item.val is w_a1.val:
                return ComboWrapper.from_val(BosonNum('a_1'), s)
            if item.val is w_a2.val:
                return ComboWrapper.from_val(BosonNum('a_2'), s)
            if isinstance(item.val, PF):
                name = str(item.val.expr())
            else:
                name = str(s.to_expr(item.val))
        elif hasattr(item, 'name'):
            name = str(item.name)
        else:
            name = str(item)
        name = op_mode_map.get(name, name)
        return ComboWrapper.from_val(BosonNum(name), s)

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, ComboWrapper) and isinstance(item.val, PF):
                if item.val.npf == S.Zero and set(item.val.table.keys()).issubset({((), ())}):
                    arg = item.val.table.get(((), ()), S.Zero)
                    return ComboWrapper(PF({((), ()): fn(arg)}), s)
            return ComboWrapper.from_val(fn(sympify(item)), s)
        return _inner

    # Represent g in terms of t_ during algebraic evaluation so NumShift on N_d works accurately
    w_g_internal = ComboWrapper(g_to_t(PF({((), ()): g_hat}), dc=dc, mismatch=mismatch), s)
    w_g_internal._is_g = True

    w_t = ComboWrapper.from_val(t_, s)
    w_t._is_t = True

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
        'dag': lambda item: ComboWrapper.from_val(item, s).adjoint(),
        'Dagger': lambda item: ComboWrapper.from_val(item, s).adjoint(),
        'dagger': lambda item: ComboWrapper.from_val(item, s).adjoint(),
        'N_': _make_boson_num,
        'BosonNum': _make_boson_num,
        'Pat': lambda item, *args: ComboWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), s
        ),
        'pattern': lambda item, *args: ComboWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), s
        ),
        'BP': lambda item, *args: ComboWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), s
        ),
        'FP': lambda item, *args: ComboWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), s
        ),
        'MMN': lambda name, disp=0: ComboWrapper.from_val(MultimodeNum(op_mode_map.get(str(name), str(name)), disp), s),
        'MMN2': lambda disp=0: ComboWrapper.from_val(
            MultimodeNum2(a_s.name, a_i.name, 0, disp) if dc else MultimodeNum2(a_p.name, a_s.name, 1, disp), s
        ),
        't': w_t,
        'g': w_g_internal,
        'diff': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'Derivative': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'nlc': ComboWrapper.from_val(nlc, s),
        'pm': ComboWrapper.from_val(s.pm, s),
        'a_0': w_a0,
        'a_1': w_a1,
        'a_2': w_a2,
        'a_p': evolved_wrappers[0],
        'a_s': evolved_wrappers[1],
        'a_i': evolved_wrappers[2],
        'A': w_A,
        'N_t': w_Nt,
    }

    disp_subs = {}

    # 1. Register commutative symbols from table
    for sym_def in symbols:
        s_name = (sym_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(s_name)
        if not valid:
            raise ValueError(f"Invalid symbol identifier: {err_msg}")
        d_name = (sym_def.get('display_name') or '').strip()

        # Built-in parametric symbols nlc and pm keep their identity in TWMSolver formulas
        if s_name == 'nlc' or sym_def.get('builtin_key') == 'nlc':
            local_dict[s_name] = ComboWrapper.from_val(nlc, s)
            if d_name and d_name not in ('nlc', r'\chi', 'χ'):
                disp_subs[nlc] = Symbol(d_name)
            continue
        if s_name == 'pm' or sym_def.get('builtin_key') == 'pm':
            if not mismatch:
                local_dict[s_name] = ComboWrapper.from_val(S.Zero, s)
            else:
                local_dict[s_name] = ComboWrapper.from_val(pm, s)
                if d_name and d_name not in ('pm', 'Δk'):
                    disp_subs[pm] = Symbol(d_name)
            continue

        kwargs = {}
        if sym_def.get('is_positive'): kwargs['positive'] = True
        elif sym_def.get('is_nonnegative'): kwargs['nonnegative'] = True
        elif sym_def.get('is_integer'): kwargs['integer'] = True
        elif sym_def.get('is_real'): kwargs['real'] = True
        sym_obj = Symbol(s_name, **kwargs)
        local_dict[s_name] = ComboWrapper.from_val(sym_obj, s)
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
            w_op = ComboWrapper.from_val(op_obj, s)
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
        w_mm = w_A if mm_kind == 'COP' else w_Nt
        local_dict[mm_name] = w_mm

    # 4. Register saved named expressions
    for ne in named_expressions:
        ne_name = (ne.get('name') or '').strip()
        ne_str = (ne.get('expr_str_t') or ne.get('expr_str') or ne.get('exprStr') or '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")
        if not ne_str:
            continue
        try:
            parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
            w_ne = ComboWrapper.from_val(parsed_ne, s)
            # Ensure internal representation is in terms of t_ for subsequent multiplications
            w_ne = ComboWrapper(g_to_t(w_ne.val, dc=dc, mismatch=mismatch), s)
            local_dict[ne_name] = w_ne
        except Exception as e:
            raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    # 5. Parse and evaluate target expression
    try:
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS)
        wrapper_target = ComboWrapper.from_val(parsed_target, s)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

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

    val = _apply_to_pfs(val, _postprocess_pf)

    # 7. Compute both time (t) and gain (g) representations
    val_t = g_to_t(val, dc=dc, mismatch=mismatch)
    expr_result_t = val_t.expr(False) if isinstance(val_t, PF) else s.to_expr(val_t)
    expr_str_t = expr_to_code_str(expr_result_t, dc=dc)
    expr_disp_t = expr_result_t
    if disp_subs:
        try:
            expr_disp_t = expr_result_t.subs(disp_subs)
        except Exception:
            expr_disp_t = expr_result_t
    latex_t = latex(expr_disp_t)

    val_g = t_to_g(val, dc=dc, mismatch=mismatch)
    expr_result_g = val_g.expr(False) if isinstance(val_g, PF) else s.to_expr(val_g)
    expr_str_g = expr_to_code_str(expr_result_g, dc=dc)
    expr_disp_g = expr_result_g
    if disp_subs:
        try:
            expr_disp_g = expr_result_g.subs(disp_subs)
        except Exception:
            expr_disp_g = expr_result_g
    latex_g = latex(expr_disp_g)

    latex_curr = latex_g if use_gain else latex_t
    expr_curr = expr_str_g if use_gain else expr_str_t

    leading_order = compute_val_leading_order(val, s)

    return latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g, leading_order

