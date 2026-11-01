from __future__ import annotations
import ast
from typing import Any, Dict, List, Optional, Tuple

from dsolve_pm import d_p2s
from op_patterns import BosonNum, BosonPat, FermionPat, check_num
from parametric import (
    TWMSolver, f_dict, d_apply, t_, _g, _nm, nlc, pm,
    a_p, a_s, a_i, MultimodeNum, MultimodeNum2, MultimodeCOP, _tdz,
    INIT_OPS, PF_ONE
)
from perturb_eval import pattern_form, PF, PE, Pat
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.mul import Mul
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
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex

from .common import (
    TRANSFORMS,
    is_valid_variable_name,
    g_hat,
)
from .twm_transforms import t_to_g, g_to_t

# Cached TWMSolver instances indexed by [dc][mismatch]
SOLVERS = (
    (TWMSolver(False, False), TWMSolver(False, True)),
    (TWMSolver(True, False), TWMSolver(True, True)),
)


def expr_to_code_str(e: Expr, dc: bool = True) -> str:
    """
    Convert a SymPy quantum operator expression (from PF.expr() or TWMSolver.to_expr())
    into a valid Python expression string that can be parsed back by parse_expr in Tab 3.
    """
    if e == 0 or e == S.Zero:
        return "0"
    if e == 1 or e == S.One:
        return "1"

    mmc_ann = MultimodeCOP(a_s.name, a_i.name) if dc else MultimodeCOP(a_p.name, a_s.name, 1)
    mmc_cre = Dagger(mmc_ann)
    mmn2_canonical = MultimodeNum2(a_s.name, a_i.name, 0) if dc else MultimodeNum2(a_p.name, a_s.name, 1)

    def _rec(x):
        if isinstance(x, MultimodeCOP):
            if x == mmc_ann:
                return Symbol('A', commutative=False)
            if x == mmc_cre:
                return Symbol('dag(A)', commutative=False)
        if isinstance(x, MultimodeNum2):
            if x == mmn2_canonical:
                return Symbol('N_t', commutative=False)
            if x.displacement == 0:
                return Symbol('MMN2()', commutative=False)
            return Symbol(f'MMN2({x.displacement})', commutative=False)
        if isinstance(x, MultimodeNum):
            if x.displacement == 0:
                return Symbol(f'MMN({x.name})', commutative=False)
            return Symbol(f'MMN({x.name}, {x.displacement})', commutative=False)
        if isinstance(x, BosonNum):
            return Symbol(f'N_({x.name})', commutative=False)
        if isinstance(x, (BosonOp, FermionOp)):
            return Symbol(str(x.name), commutative=False) if x.is_annihilation else Symbol(f'dag({x.name})', commutative=False)
        if isinstance(x, (BosonPat, FermionPat)):
            factors = []
            is_ann = x.left_base.is_annihilation
            for exp_val in x.exps:
                op_sym = Symbol(str(x.name), commutative=False) if is_ann else Symbol(f'dag({x.name})', commutative=False)
                factors.append(op_sym ** exp_val if exp_val != 1 else op_sym)
                is_ann = not is_ann
            return Mul(*factors)
        if isinstance(x, Dagger) or getattr(x, 'func', None) and x.func.__name__ == 'conjugate':
            inner = _rec(x.args[0])
            return Symbol(f'dag({inner})', commutative=False)
        if x == _nm:
            return Symbol('n_m', nonnegative=True)
        if x == t_:
            return Symbol('t', real=True)
        if x == g_hat or x == _g:
            return Symbol('g', commutative=False)
        if x == nlc:
            return Symbol('nlc', real=True, nonnegative=True)
        if x == pm:
            return Symbol('pm', real=True)
        if not getattr(x, 'args', None):
            return x
        return x.func(*(_rec(a) for a in x.args))

    return str(_rec(sympify(e)))


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


def _mul_two_vals(v1, v2, solver: TWMSolver):
    """Multiply two single resolved values (PF, dict, or f_dict)."""
    # 1. Pure scalar PF fast-path
    if _is_pure_scalar_pf(v1):
        c = v1.table.get(((), ()), S.Zero)
        return d_apply(v2, lambda t: c * t)
    if _is_pure_scalar_pf(v2):
        c = v2.table.get(((), ()), S.Zero)
        return d_apply(v1, lambda t: t * c)
    # 2. Both PF -> independent-mode PF multiplication
    if isinstance(v1, PF) and isinstance(v2, PF):
        return v1.mul_ind(v2)
    # 3. Bare weak-field operator on the right of a PF or Multimode dict: M * b
    if not isinstance(v1, f_dict) and _is_bare_weak(v2):
        b = next(iter(v2.keys()))
        d1 = {(None,): v1} if isinstance(v1, PF) else v1
        return f_dict({b: d1}, b_left=False)
    # 4. General dict / f_dict / PF multiplication via solver.e_mul
    d1 = {(None,): v1} if isinstance(v1, PF) else v1
    d2 = {(None,): v2} if isinstance(v2, PF) else v2
    return solver.e_mul(d1, d2)


def _merge_non_f_dicts(items: List[Any], solver: TWMSolver) -> List[Any]:
    """
    Prioritize multiplying adjacent non-f_dict factors (PF and dict):
    1. First merge adjacent PF factors into a single PF.
    2. Then merge adjacent PF and dict factors into a single dict.
    """
    if len(items) <= 1:
        return items
    # Pass 1: merge adjacent PFs
    merged_pfs: List[Any] = []
    for item in items:
        if merged_pfs and isinstance(merged_pfs[-1], PF) and isinstance(item, PF):
            merged_pfs[-1] = _mul_two_vals(merged_pfs[-1], item, solver)
        else:
            merged_pfs.append(item)
    # Pass 2: merge adjacent non-f_dict factors (PF and dict)
    merged: List[Any] = []
    for item in merged_pfs:
        if merged and not isinstance(merged[-1], f_dict) and not isinstance(item, f_dict):
            merged[-1] = _mul_two_vals(merged[-1], item, solver)
        else:
            merged.append(item)
    return merged


def _dict_pow(d: dict, n: int, solver: TWMSolver) -> dict:
    """Binary exponentiation for a multimode dict (non-f_dict)."""
    if n == 0:
        return {(None,): PF({((), ()): S.One})}
    if n == 1:
        return d
    result = None
    base = d
    exp_n = n
    while exp_n > 0:
        if exp_n & 1:
            result = base if result is None else solver.e_mul(result, base)
        exp_n >>= 1
        if exp_n > 0:
            base = solver.e_mul(base, base)
    return result


def _eval_product_factors(factors: List[Any], solver: TWMSolver):
    """
    Evaluate a product chain of factors:
    1. Checks the parities of all f_dict factors and counts them;
       if the count difference between the two parities is > 1, raises ValueError.
    2. Prioritizes evaluating PF and dict factors first (_merge_non_f_dicts).
    3. Prioritizes multiplying complementary f_dict pairs that cancel into dict,
       immediately merging resulting dicts with adjacent dict/PF factors.
    """
    if not factors:
        return PF({((), ()): S.One})
    if len(factors) == 1:
        return factors[0]

    items = list(factors)
    f_list = [f for f in items if isinstance(f, f_dict)]

    if f_list:
        parities = [solver.f_dict_parity(f) for f in f_list]
        count_0 = sum(1 for p in parities if p == 0)
        count_1 = sum(1 for p in parities if p == 1)
        if abs(count_0 - count_1) > 1:
            raise ValueError('Cannot use multimode operators to represent the product')

    # Prioritize evaluating all adjacent PF and dict factors first
    items = _merge_non_f_dicts(items, solver)

    if f_list:
        # Prioritize multiplying complementary f_dict pairs that cancel into dict
        while True:
            f_positions = [
                (idx, solver.f_dict_parity(item))
                for idx, item in enumerate(items)
                if isinstance(item, f_dict)
            ]
            candidates = []
            for k in range(len(f_positions) - 1):
                idx_l, p_l = f_positions[k]
                idx_r, p_r = f_positions[k + 1]
                if p_l is not None and p_r is not None and p_l != p_r:
                    f_l = items[idx_l]
                    f_r = items[idx_r]
                    swaps = int(bool(f_l.b_left)) + int(not bool(f_r.b_left))
                    candidates.append((idx_r - idx_l, swaps, idx_l, idx_r))
            if not candidates:
                break
            _, _, best_l, best_r = min(candidates)
            sub_val = items[best_l]
            for step_idx in range(best_l + 1, best_r + 1):
                sub_val = _mul_two_vals(sub_val, items[step_idx], solver)
            items[best_l : best_r + 1] = [sub_val]
            # Immediately merge the resulting dict with any adjacent dict/PF factors
            items = _merge_non_f_dicts(items, solver)

    res = items[0]
    for item in items[1:]:
        res = _mul_two_vals(res, item, solver)
    return res


class TwmWrapper:
    """
    Algebraic wrapper for Tab 3 (Post-Evolution Operator Combinations).
    Dispatches addition, multiplication, exponentiation, and adjoint across:
    - PF (PFTableProcessor, independent-mode _ind: strong-field initial op, custom ops, number ops)
    - Multimode dict ({tuple: PF}: A and balanced weak-field pairs)
    - Weak-field f_dict ({BosonOp: {tuple: PF}}: weak-field initial ops and evolved weak modes)
    """
    def __init__(self, val, solver: TWMSolver, _factors: Optional[List[Any]] = None):
        self.solver = solver
        if _factors is not None:
            self._factors = list(_factors)
            self._val = self._factors[0] if len(self._factors) == 1 else None
        else:
            self._factors = [val]
            self._val = val
        self.name: str = ""
        self.val_t = val
        self.val_g = None
        self.latex_t: Optional[str] = None
        self.latex_g: Optional[str] = None
        self.order: Optional[int] = None
        self.leading_order: Optional[int] = None
        self.dc: Optional[bool] = getattr(solver, 'dc', True) if solver else True
        self.single_mode: Optional[bool] = False
        self.notes: str = ""

    @property
    def val(self):
        if self._val is None:
            self._val = _eval_product_factors(self._factors, self.solver)
            self._factors = [self._val]
        return self._val

    @classmethod
    def from_val(cls, val, solver: Any) -> 'TwmWrapper':
        if isinstance(val, cls):
            return val

        s = solver if isinstance(solver, TWMSolver) else SOLVERS[bool(solver)][0]
        dc = s.dc

        if isinstance(val, str) and val in INIT_OPS[dc]:
            return cls(INIT_OPS[dc][val], s)

        for k, v in INIT_OPS[dc].items():
            if val is v:
                return cls(v, s)

        if isinstance(val, (PF, dict, f_dict)):
            return cls(val, solver)

        expr = sympify(val)
        if isinstance(expr, MultimodeCOP):
            return cls({(bool(expr.annih1), 1): PF_ONE}, solver)

        weak_names = set(solver.a[0].names)
        if isinstance(expr, BosonOp) and expr.name in weak_names:
            return cls(f_dict({expr: {(None,): PF_ONE}}, b_left=True), solver)

        if not expr.has(MultimodeCOP) and (
            check_num(expr) or not expr.has(BosonOp, BosonPat, FermionOp, FermionPat)
        ):
            return cls(PF({((), ()): expr}), solver)

        if isinstance(expr, BosonPat) and not isinstance(expr, (BosonOp, BosonNum)) and expr.name in weak_names:
            is_a = bool(expr.left_base.is_annihilation) if hasattr(expr, 'left_base') else bool(expr.args[1])
            res = cls(PF_ONE, solver)
            for e in expr.exps:
                b = BosonOp(expr.name, is_a)
                w_b = cls(f_dict({b: {(None,): PF_ONE}}, b_left=True), solver)
                res = res * (w_b ** int(e))
                is_a = not is_a
            return res

        if expr.has(MultimodeCOP) or any(
            getattr(b, 'name', None) in weak_names
            for b in expr.atoms(BosonOp, BosonPat)
            if not isinstance(b, BosonNum)
        ):
            if expr.is_Add:
                res = cls.from_val(expr.args[0], solver)
                for arg in expr.args[1:]:
                    res = res + cls.from_val(arg, solver)
                return res
            if expr.is_Mul:
                res = cls.from_val(expr.args[0], solver)
                for arg in expr.args[1:]:
                    res = res * cls.from_val(arg, solver)
                return res
            if expr.is_Pow:
                return cls.from_val(expr.base, solver) ** expr.exp

        pf = pattern_form(expr, as_dict=None)
        return cls(pf, solver)

    def to_single_mode(self) -> 'TwmWrapper':
        """Projects this wrapper's value into single-mode PF representation."""
        return TwmWrapper(self.solver.to_single_mode(self.val), self.solver)

    def _to_mm_dict(self):
        v = self.val
        if isinstance(v, PF):
            return {(None,): v}
        return v

    def __add__(self, other):
        other = TwmWrapper.from_val(other, self.solver)
        v1, v2 = self.val, other.val
        if isinstance(v1, PF) and isinstance(v2, PF):
            return TwmWrapper(PF.add(v1, v2), self.solver)
        d1 = self._to_mm_dict()
        d2 = other._to_mm_dict()
        return TwmWrapper(self.solver.d_add(d1, d2), self.solver)

    def __radd__(self, other):
        return TwmWrapper.from_val(other, self.solver).__add__(self)

    def __neg__(self):
        return TwmWrapper(d_apply(self.val, lambda x: -x), self.solver)

    def __sub__(self, other):
        return self.__add__(-TwmWrapper.from_val(other, self.solver))

    def __rsub__(self, other):
        return TwmWrapper.from_val(other, self.solver).__sub__(self)

    def __mul__(self, other):
        other = TwmWrapper.from_val(other, self.solver)
        return TwmWrapper(None, self.solver, _factors=self._factors + other._factors)

    def __rmul__(self, other):
        return TwmWrapper.from_val(other, self.solver).__mul__(self)

    def __truediv__(self, other):
        other = TwmWrapper.from_val(other, self.solver)
        if isinstance(other.val, PF) and other.val.npf == S.Zero and set(other.val.table.keys()).issubset({((), ())}):
            denom = other.val.table.get(((), ()), S.One)
            inv_pf = PF({((), ()): denom ** S.NegativeOne})
            return self.__mul__(TwmWrapper(inv_pf, self.solver))
        raise ValueError("Division is only supported by scalar or number-operator expressions.")

    def __rtruediv__(self, other):
        return TwmWrapper.from_val(other, self.solver).__truediv__(self)

    def __pow__(self, power):
        if isinstance(power, TwmWrapper):
            if isinstance(power.val, PF) and power.val.npf == S.Zero and set(power.val.table.keys()).issubset({((), ())}):
                power = power.val.table.get(((), ()), S.Zero)
            else:
                raise ValueError("Exponent must be a scalar or number-operator expression.")
        p_sym = sympify(power)
        if len(self._factors) == 1 and isinstance(self._factors[0], PF):
            pf_val = self._factors[0]
            if pf_val.npf == S.Zero and set(pf_val.table.keys()).issubset({((), ())}):
                base_expr = pf_val.table.get(((), ()), S.Zero)
                return TwmWrapper(PF({((), ()): base_expr ** p_sym}), self.solver)
            if p_sym.is_Integer and p_sym >= 0:
                return TwmWrapper(pf_val.power_ind(Integer(p_sym)), self.solver)
            raise ValueError("Operator exponentiation only supports non-negative integers.")
        if p_sym.is_Integer and p_sym >= 0:
            n = int(p_sym)
            if n == 0:
                return TwmWrapper({(None,): PF({((), ()): S.One})}, self.solver)
            if len(self._factors) == 1 and isinstance(self._factors[0], dict) and not isinstance(self._factors[0], f_dict):
                return TwmWrapper(_dict_pow(self._factors[0], n, self.solver), self.solver)
            f_list = [f for f in self._factors if isinstance(f, f_dict)]
            if not f_list:
                base_val = self.val
                if isinstance(base_val, PF):
                    return TwmWrapper(base_val.power_ind(Integer(n)), self.solver)
                return TwmWrapper(_dict_pow(base_val, n, self.solver), self.solver)
            parities = [self.solver.f_dict_parity(f) for f in f_list]
            count_0 = sum(1 for p in parities if p == 0)
            count_1 = sum(1 for p in parities if p == 1)
            if count_0 == count_1:
                base_val = self.val
                return TwmWrapper(_dict_pow(base_val, n, self.solver), self.solver)
            return TwmWrapper(None, self.solver, _factors=self._factors * n)
        raise ValueError("Exponentiation for this operator type only supports non-negative integers.")

    def adjoint(self):
        adj_factors = [
            f.adjoint(res=False) if isinstance(f, PF) else self.solver.d_adjoint(f)
            for f in reversed(self._factors)
        ]
        return TwmWrapper(None, self.solver, _factors=adj_factors)

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

        if isinstance(var, TwmWrapper):
            if getattr(var, '_is_g', False):
                is_wrt_g = True
            elif getattr(var, '_is_t', False):
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
            return TwmWrapper(diff_in_t, s)
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
            return TwmWrapper(diff_val, s)
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
            return TwmWrapper(diff_val, s)

    def Derivative(self, *symbols, **kwargs):
        return self.diff(*symbols, **kwargs)


TWM_VARIABLES: Dict[str, TwmWrapper] = {}
LAST_EVAL_TWM: Dict[str, Any] = {}


def save_twm_variable(
    name: str,
    val_t: Any,
    val_g: Any = None,
    solver: Any = None,
    latex_t: Optional[str] = None,
    latex_g: Optional[str] = None,
    **kwargs
) -> TwmWrapper:
    s = solver if isinstance(solver, TWMSolver) else SOLVERS[bool(kwargs.get('dc', True))][0]
    w = TwmWrapper.from_val(val_t, s)
    w.name = name
    w.val_t = w.val
    w.val_g = val_g if val_g is not None else t_to_g(w.val, dc=kwargs.get('dc', s.dc), mismatch=kwargs.get('mismatch', False))
    w.latex_t = latex_t
    w.latex_g = latex_g
    for k, v in kwargs.items():
        setattr(w, k, v)
    TWM_VARIABLES[name] = w
    return w


def save_latest_twm_variable(name: str, **kwargs) -> Optional[TwmWrapper]:
    if not LAST_EVAL_TWM:
        return None
    s = LAST_EVAL_TWM.get('solver') or SOLVERS[bool(LAST_EVAL_TWM.get('dc', True))][0]
    return save_twm_variable(
        name,
        LAST_EVAL_TWM['val_t'],
        val_g=LAST_EVAL_TWM.get('val_g'),
        solver=s,
        latex_t=LAST_EVAL_TWM.get('latex_t'),
        latex_g=LAST_EVAL_TWM.get('latex_g'),
        order=LAST_EVAL_TWM.get('order'),
        leading_order=LAST_EVAL_TWM.get('leading_order'),
        dc=LAST_EVAL_TWM.get('dc'),
        single_mode=LAST_EVAL_TWM.get('single_mode'),
        **kwargs
    )


def get_twm_variable(name: str) -> Optional[TwmWrapper]:
    return TWM_VARIABLES.get(name)


def clear_twm_variables() -> None:
    TWM_VARIABLES.clear()
    LAST_EVAL_TWM.clear()


def compute_val_leading_order(val: Any, s: TWMSolver, dc: Optional[bool] = None) -> int:
    """
    Computes the leading perturbation order of a calculated TWM result value `val`.
    Uses TWMSolver.pe which includes base assumptions and any custom symbol/operator orders.
    Driving mode: order -1 (a_p in dc=True; a_i in dc=False).
    Weak modes, symbols, multimode operators: order 0 (or custom order if defined).
    Nonlinear coupling (nlc): order 1.
    Driving photon number N_d: order -2.
    """
    if dc is None:
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
                    orders.append(int(npf_ord) + int(extra_ord))
            except Exception:
                pass
        for (bs, fs), t_expr in p.table.items():
            if t_expr == 0 or t_expr is S.Zero:
                continue
            bs_ord = 0
            for b in bs:
                b_name = getattr(b, 'name', None)
                if b_name == strong_name:
                    bs_ord += -int(sum(b.exps))
                elif b_name and b_name in pe.assumptions:
                    bs_ord += int(pe.assumptions[b_name]) * int(sum(b.exps))
                elif b and b in pe.assumptions:
                    bs_ord += int(pe.assumptions[b]) * int(sum(b.exps))
            fs_ord = 0
            for f in fs:
                f_name = getattr(f, 'name', None)
                if f_name and f_name in pe.assumptions:
                    fs_ord += int(pe.assumptions[f_name]) * int(sum(f.exps))
                elif f and f in pe.assumptions:
                    fs_ord += int(pe.assumptions[f]) * int(sum(f.exps))
            try:
                sub_t = t_expr.subs({t_: _g / s.gc})
                t_ord = pe.leading_order(sub_t)[1]
                if t_ord is not S.Infinity:
                    orders.append(int(t_ord) + int(bs_ord) + int(fs_ord) + int(extra_ord))
            except Exception:
                pass

    def _walk(obj, extra_ord: int = 0):
        if isinstance(obj, PF):
            _inspect_pf(obj, extra_ord)
        elif isinstance(obj, f_dict):
            for v in obj.values():
                _walk(v, extra_ord)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v, extra_ord)

    _walk(val)
    if not orders:
        return 0
    try:
        return int(min(orders))
    except Exception:
        return 0



def analyze_expression_terms(
    expr_str: str,
    target_order: int,
    dc: bool,
    symbols: Optional[List[Dict[str, Any]]] = None,
    operators: Optional[List[Dict[str, Any]]] = None,
    named_expressions: Optional[List[Dict[str, Any]]] = None
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Analyzes an expression across each additive term:
    - If a term contains estimated items (field operator evolution or named expressions)
      multiplied by exact items (symbols, initial ops, number ops, etc.), the effective accuracy
      of that term is min(order of estimated items) + sum(leading orders of exact items).
    - If field operator evolutions (a_p, a_s, a_i) are multiplied by negative leading-order exact items,
      their required accuracy is target_order - sum(leading orders of exact items).
      Returns the maximum required order for field operator evolutions (to call s.get_sol).
    - If a custom saved expression has a recorded order less than required, records it in
      insufficient_named for UI warning.
    """
    strong_mode_idx = 0 if dc else 2
    strong_names = {'a_p', 'a_0'} if dc else {'a_i', 'a_2'}
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

    sym_orders = {}
    if symbols:
        for s_def in symbols:
            name = (s_def.get('name') or '').strip()
            if name:
                if name in ('nlc', 'chi', r'\chi', 'χ') or s_def.get('builtin_key') == 'nlc':
                    sym_orders[name] = 1
                else:
                    try:
                        sym_orders[name] = int(s_def.get('order', 0))
                    except (ValueError, TypeError):
                        sym_orders[name] = 0

    def get_factor_info(node) -> Tuple[bool, int, str, Any]:
        # (is_exact, leading_order, category, ref)
        if isinstance(node, ast.Constant):
            return True, 0, 'exact', None
        if isinstance(node, ast.Name):
            name = node.id
            if name in named_map:
                ne = named_map[name]
                lo = int(ne.get('leading_order') if ne.get('leading_order') is not None else (ne.get('leadingOrder') or 0))
                return False, lo, 'named', ne
            if name in TWM_VARIABLES:
                cvar = TWM_VARIABLES[name]
                lo = int(cvar.leading_order if cvar.leading_order is not None else 0)
                return False, lo, 'named', cvar
            if name in evolved_names:
                lo = -1 if name in strong_names else 0
                return False, lo, 'evolved', name
            if name in sym_orders:
                return True, sym_orders[name], 'exact', name
            if name in ('nlc', 'chi', r'\chi', 'χ'):
                return True, 1, 'exact', name
            lo = -1 if name in strong_names else 0
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
                    lo = -2 if arg_name in strong_names else 0
                    return True, lo, 'exact', None
            return True, 0, 'exact', None
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
            base_exact, base_lo, cat, ref = get_factor_info(node.left)
            exp_val = 1
            if isinstance(node.right, ast.Constant) and isinstance(node.right.value, int):
                exp_val = node.right.value
            elif isinstance(node.right, ast.UnaryOp) and isinstance(node.right.op, ast.USub) and isinstance(node.right.operand, ast.Constant):
                exp_val = -node.right.operand.value
            return base_exact, base_lo * exp_val, cat, ref
        return True, 0, 'exact', None

    def flatten_product(node) -> List[Any]:
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return flatten_product(node.left) + flatten_product(node.right)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return flatten_product(node.left) + [node.right]
        if isinstance(node, ast.UnaryOp):
            return flatten_product(node.operand)
        return [node]

    def split_additive_terms(node) -> List[Any]:
        if isinstance(node, ast.Expression):
            return split_additive_terms(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return split_additive_terms(node.left) + split_additive_terms(node.right)
        return [node]

    try:
        tree = ast.parse(expr_str, mode='eval')
        additive_terms = split_additive_terms(tree)
    except Exception:
        return target_order, []

    max_evolved_needed = target_order
    insufficient_named = []
    seen_warnings = set()

    for term in additive_terms:
        factors = flatten_product(term)
        exact_lo_sum = 0
        has_estimated = False
        term_evolved_count = 0
        term_named = []

        for f in factors:
            is_exact, lo, cat, ref = get_factor_info(f)
            if is_exact:
                exact_lo_sum += lo
            else:
                has_estimated = True
                if cat == 'evolved':
                    term_evolved_count += 1
                elif cat == 'named':
                    term_named.append((ref, lo))

        if not has_estimated:
            continue

        req_order = target_order - exact_lo_sum

        if term_evolved_count > 0:
            if req_order > max_evolved_needed:
                max_evolved_needed = req_order

        for ref, lo in term_named:
            rec_order = ref.get('order') if isinstance(ref, dict) else getattr(ref, 'order', None)
            if rec_order is not None and rec_order < req_order:
                name = ref.get('name') if isinstance(ref, dict) else getattr(ref, 'name', '')
                key = (name, rec_order, req_order)
                if key not in seen_warnings:
                    seen_warnings.add(key)
                    insufficient_named.append({
                        'name': name,
                        'order': rec_order,
                        'required_order': req_order
                    })

    return max_evolved_needed, insufficient_named


def _prepare_twm_evaluation(
    text: str,
    dc: bool,
    order: int,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    mm_operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    mismatch: bool = False,
    single_mode: bool = False,
):
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

    # Automatically analyze required order for evolved field operators
    # If evolved operators are multiplied by negative leading-order exact items,
    # append get_sol until the accuracy of the evolved operators is sufficient.
    needed_evolved_order, _ = analyze_expression_terms(
        text,
        target_order=order,
        dc=dc,
        symbols=symbols,
        operators=operators,
        named_expressions=named_expressions
    )
    evolved_order = max(-1, order, needed_evolved_order)
    sol = s.get_sol(evolved_order)

    # Initial-time (t=0) mode wrappers directly from INIT_OPS[dc]
    init_wrappers = {k: TwmWrapper(v, s) for k, v in INIT_OPS[dc].items()}
    evolved_wrappers = [TwmWrapper(s_op, s) for s_op in sol]

    if single_mode:
        init_wrappers = {k: w.to_single_mode() for k, w in init_wrappers.items()}
        evolved_wrappers = [w.to_single_mode() for w in evolved_wrappers]

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

    def _resolve_mode_name(item) -> str:
        if isinstance(item, TwmWrapper):
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
        if isinstance(x, TwmWrapper):
            if isinstance(x.val, PF) and x.val.npf == S.Zero and set(x.val.table.keys()).issubset({((), ())}):
                return x.val.table.get(((), ()), S.Zero)
            return x.val.expr() if isinstance(x.val, PF) else s.to_expr(x.val)
        return sympify(x)

    def _make_boson_num(item):
        return TwmWrapper.from_val(BosonNum(_resolve_mode_name(item)), s)

    def _make_mmn(name, disp=0):
        return TwmWrapper.from_val(MultimodeNum(_resolve_mode_name(name), _unwrap_scalar(disp)), s)

    def _make_mmn2(disp=0):
        d_val = _unwrap_scalar(disp)
        return TwmWrapper.from_val(
            MultimodeNum2(a_s.name, a_i.name, 0, d_val) if dc else MultimodeNum2(a_p.name, a_s.name, 1, d_val), s
        )

    def _make_pat(item, *args):
        raw_args = []
        for a in args:
            if isinstance(a, (tuple, list)):
                raw_args.extend(x if isinstance(x, bool) else _unwrap_scalar(x) for x in a)
            else:
                raw_args.append(a if isinstance(a, bool) else _unwrap_scalar(a))

        is_A = isinstance(item, TwmWrapper) and (item is init_wrappers['A'] or item.val == INIT_OPS[dc]['A'])
        is_dagA = (isinstance(item, TwmWrapper) and isinstance(item.val, dict) and not isinstance(item.val, f_dict)
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
            return TwmWrapper({_tdz(is_ann, int_exps): PF_ONE}, s)

        if isinstance(item, TwmWrapper) and isinstance(item.val, (f_dict, dict)):
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
            res = TwmWrapper(PF_ONE, s)
            for e in exps:
                cur = op_ann if is_a else op_ann.adjoint()
                res = res * (cur ** int(sympify(e)))
                is_a = not is_a
            return res

        return TwmWrapper.from_val(
            Pat(item, *raw_args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), s
        )

    def _make_bp(item, *args):
        raw_args = []
        for a in args:
            if isinstance(a, (tuple, list)):
                raw_args.extend(x if isinstance(x, bool) else _unwrap_scalar(x) for x in a)
            else:
                raw_args.append(a if isinstance(a, bool) else _unwrap_scalar(a))
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        return TwmWrapper.from_val(BosonPat(name, *raw_args), s)

    def _make_fp(item, *args):
        raw_args = []
        for a in args:
            if isinstance(a, (tuple, list)):
                raw_args.extend(x if isinstance(x, bool) else _unwrap_scalar(x) for x in a)
            else:
                raw_args.append(a if isinstance(a, bool) else _unwrap_scalar(a))
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        return TwmWrapper.from_val(FermionPat(name, *raw_args), s)

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, TwmWrapper) and isinstance(item.val, PF):
                if item.val.npf == S.Zero and set(item.val.table.keys()).issubset({((), ())}):
                    arg = item.val.table.get(((), ()), S.Zero)
                    return TwmWrapper(PF({((), ()): fn(arg)}), s)
            return TwmWrapper.from_val(fn(sympify(item)), s)
        return _inner

    # Represent g in terms of t_ during algebraic evaluation so NumShift on N_d works accurately
    w_g_internal = TwmWrapper(g_to_t(PF({((), ()): g_hat}), dc=dc, mismatch=mismatch), s)
    w_g_internal._is_g = True

    w_t = TwmWrapper.from_val(t_, s)
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
        'dag': lambda item: TwmWrapper.from_val(item, s).adjoint(),
        'Dagger': lambda item: TwmWrapper.from_val(item, s).adjoint(),
        'dagger': lambda item: TwmWrapper.from_val(item, s).adjoint(),
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
        'nlc': TwmWrapper.from_val(nlc, s),
        'pm': TwmWrapper.from_val(s.pm, s),
        **init_wrappers,
        'a_p': evolved_wrappers[0],
        'a_s': evolved_wrappers[1],
        'a_i': evolved_wrappers[2],
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
            local_dict[s_name] = TwmWrapper.from_val(nlc, s)
            if d_name and d_name not in ('nlc', r'\chi', 'χ'):
                disp_subs[nlc] = Symbol(d_name)
            continue
        if s_name == 'pm' or sym_def.get('builtin_key') == 'pm':
            if not mismatch:
                local_dict[s_name] = TwmWrapper.from_val(S.Zero, s)
            else:
                local_dict[s_name] = TwmWrapper.from_val(pm, s)
                if d_name and d_name not in ('pm', 'Δk'):
                    disp_subs[pm] = Symbol(d_name)
            continue

        kwargs = {}
        if sym_def.get('is_positive'): kwargs['positive'] = True
        elif sym_def.get('is_nonnegative'): kwargs['nonnegative'] = True
        elif sym_def.get('is_integer'): kwargs['integer'] = True
        elif sym_def.get('is_real'): kwargs['real'] = True
        sym_obj = Symbol(s_name, **kwargs)
        local_dict[s_name] = TwmWrapper.from_val(sym_obj, s)
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
            w_op = TwmWrapper.from_val(op_obj, s)
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

        if val_obj is not None:
            if isinstance(val_obj, TwmWrapper):
                w_ne = val_obj
            else:
                w_ne = TwmWrapper(val_obj, s)
            w_ne = TwmWrapper(g_to_t(w_ne.val, dc=dc, mismatch=mismatch), s)
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
            save_twm_variable(
                ne_name,
                w_ne.val,
                solver=s,
                latex_t=ne.get('latex_t'),
                latex_g=ne.get('latex_g'),
                order=ne.get('order'),
                leading_order=ne.get('leading_order'),
                dc=dc,
                single_mode=single_mode
            )
        elif ne_name in TWM_VARIABLES:
            w_ne = TWM_VARIABLES[ne_name]
            w_ne = TwmWrapper(g_to_t(w_ne.val, dc=dc, mismatch=mismatch), s)
            if single_mode:
                w_ne = w_ne.to_single_mode()
            local_dict[ne_name] = w_ne
        else:
            ne_str = (ne.get('expr_str_t') or ne.get('expr_str') or ne.get('source_input') or ne.get('sourceInput') or '').strip()
            if not ne_str:
                continue
            try:
                parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
                w_ne = TwmWrapper.from_val(parsed_ne, s)
                # Ensure internal representation is in terms of t_ for subsequent multiplications
                w_ne = TwmWrapper(g_to_t(w_ne.val, dc=dc, mismatch=mismatch), s)
                # If single_mode is enabled, project any multimode named expressions into single mode (E -> P)
                if single_mode:
                    w_ne = w_ne.to_single_mode()
                local_dict[ne_name] = w_ne
                save_twm_variable(
                    ne_name,
                    w_ne.val,
                    solver=s,
                    latex_t=ne.get('latex_t'),
                    latex_g=ne.get('latex_g'),
                    order=ne.get('order'),
                    leading_order=ne.get('leading_order'),
                    dc=dc,
                    single_mode=single_mode
                )
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    # 5. Parse and evaluate target expression
    try:
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS)
        wrapper_target = TwmWrapper.from_val(parsed_target, s)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    return wrapper_target, s, disp_subs


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
) -> Tuple[str, str, str, str, str, str, int, int]:
    wrapper_target, s, disp_subs = _prepare_twm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode
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

    val = _apply_to_pfs(val, _postprocess_pf)

    # 6.1 If single-mode processing is requested, convert multimode series to mode_expr with n_m = 1
    if single_mode:
        val = s.to_single_mode(val)

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

    try:
        leading_order = int(compute_val_leading_order(val, s, dc=dc))
    except Exception:
        leading_order = 0

    sol_len = len(s.sol)

    global LAST_EVAL_TWM
    LAST_EVAL_TWM = {
        'val_t': val_t,
        'val_g': val_g,
        'val': val_t,
        'solver': s,
        'latex_t': latex_t,
        'latex_g': latex_g,
        'order': order,
        'leading_order': leading_order,
        'dc': dc,
        'single_mode': single_mode,
    }

    return latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g, leading_order, sol_len


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
) -> Tuple[str, str, str, str, str, str, int, int]:
    """
    Evaluates operator combination under selected input-state assertions:
    - coherent_drive: strong driving field input is in coherent state |alpha>
    - vacuum_signal: signal field (a_1) input is in vacuum |0>
    - vacuum_other_weak: remaining weak field (a_2 if dc else a_0) input is in vacuum |0>
    """
    wrapper_target, s, disp_subs = _prepare_twm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode
    )
    val = wrapper_target.val

    vac_modes = []
    if vacuum_signal:
        vac_modes.append('a_1')
    if vacuum_other_weak:
        vac_modes.append('a_2' if dc else 'a_0')

    if vac_modes:
        val = s.eval_vacuum(val, vac_modes, single_mode=single_mode)
    elif single_mode:
        val = s.to_single_mode(val)

    if coherent_drive:
        val_t, val_g = s.eval_coherent(val, order, return_both=True)
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

        val = _apply_to_pfs(val, _postprocess_pf)
        val_t = g_to_t(val, dc=dc, mismatch=mismatch)
        val_g = t_to_g(val, dc=dc, mismatch=mismatch)

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
    expr_str_t = expr_to_code_str(expr_result_t, dc=dc)
    expr_str_g = expr_to_code_str(expr_result_g, dc=dc)

    latex_curr = latex_g if use_gain else latex_t
    expr_curr = expr_str_g if use_gain else expr_str_t

    try:
        leading_order = int(compute_val_leading_order(val_t, s, dc=dc))
    except Exception:
        leading_order = 0

    sol_len = len(s.sol)

    global LAST_EVAL_TWM
    LAST_EVAL_TWM = {
        'val_t': val_t,
        'val_g': val_g,
        'val': val_t,
        'solver': s,
        'latex_t': latex_t,
        'latex_g': latex_g,
        'order': order,
        'leading_order': leading_order,
        'dc': dc,
        'single_mode': single_mode,
    }

    return latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g, leading_order, sol_len



# Backwards compatibility aliases
ComboWrapper = TwmWrapper
COMBO_VARIABLES = TWM_VARIABLES
save_combo_variable = save_twm_variable
save_latest_combo_variable = save_latest_twm_variable
get_combo_variable = get_twm_variable
clear_combo_variables = clear_twm_variables
