from __future__ import annotations
import keyword
import re
from typing import Any, Dict, List, Optional, Tuple

from op_patterns import BosonNum, BosonPat, FermionPat
from perturb_eval import pattern_form, PF, PFTableProcessor
from parametric import (
    TWMSolver, f_dict, d_apply, t_, _g, _nd, _ps, nlc, pm,
    a_p, a_s, a_i, MultimodeNum, MultimodeNum2, MultimodeCOP
)
from dsolve_pm import d_p2s
from sympy.core.expr import Expr
from sympy.core.function import diff
from sympy.core.mul import Mul
from sympy.core.numbers import I, Integer, pi
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.traversal import bottom_up
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.hyperbolic import cosh, sinh, tanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.fermion import FermionOp
from sympy.printing.latex import latex
from sympy.core.sympify import sympify

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# Cached TWMSolver instances indexed by [dc][mismatch]
SOLVERS = (
    (TWMSolver(False, False), TWMSolver(False, True)),
    (TWMSolver(True, False), TWMSolver(True, True)),
)

# Parametric gain operator symbol \hat{g} (non-commutative to preserve operator ordering)
g_hat = Symbol(r'\hat{g}', commutative=False)


RESERVED_NAMES = {
    # Constants
    'i', 'I', 'pi', 'PI', 'e', 'E',
    # Elementary functions
    'exp', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
    'sinh', 'cosh', 'tanh', 'coth',
    'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
    'sqrt', 'log', 'ln',
    # Quantum macros / operators
    'dag', 'Dagger', 'dagger', 'BosonNum', 'N_', 'BP', 'FP',
    'BosonPat', 'FermionPat', 'BosonOp', 'FermionOp',
    'MMN', 'MMN2', 'MMC',
}


def is_valid_variable_name(name: str) -> Tuple[bool, str]:
    """Validate whether name conforms to Python identifier rules and does not conflict with keywords or built-in functions/constants."""
    if not name or not isinstance(name, str):
        return False, "Identifier cannot be empty."
    name = name.strip()
    if not name.isidentifier():
        return False, f"'{name}' is not a valid Python identifier."
    if keyword.iskeyword(name):
        return False, f"'{name}' is a reserved Python keyword."
    if name in RESERVED_NAMES:
        return False, f"'{name}' conflicts with a built-in constant or mathematical function ({name})."
    return True, ""


def is_valid_python_name(name: str) -> bool:
    """Validate whether name conforms to Python identifier rules and does not conflict with keywords or built-ins."""
    return is_valid_variable_name(name)[0]


def expr_to_code_str(e: Expr, dc: bool = True) -> str:
    """
    Convert a SymPy quantum operator expression (from PF.expr() or TWMSolver.to_expr())
    into a valid Python expression string that can be parsed back by parse_expr.
    """
    if e == 0 or e == S.Zero:
        return "0"
    if e == 1 or e == S.One:
        return "1"

    s_Nt = Symbol('N_t', commutative=False)
    s_A = Symbol('A', commutative=False)
    s_dagA = Symbol('dag(A)', commutative=False)
    s_t = Symbol('t', real=True)
    s_g = Symbol('g', commutative=False)
    s_nlc = Symbol('nlc', real=True, nonnegative=True)
    s_pm = Symbol('pm', real=True)

    mmc_ann = MultimodeCOP(a_s.name, a_i.name) if dc else MultimodeCOP(a_p.name, a_s.name, 1)
    mmc_cre = Dagger(mmc_ann)

    def _rec(x):
        if isinstance(x, MultimodeCOP):
            if x == mmc_ann:
                return s_A
            if x == mmc_cre:
                return s_dagA
        if isinstance(x, MultimodeNum2):
            if x.displacement == 0:
                return s_Nt
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
        if isinstance(x, Dagger):
            inner = _rec(x.args[0])
            return Symbol(f'dag({inner})', commutative=False)
        if x == t_:
            return s_t
        if x == g_hat or x == _g:
            return s_g
        if x == nlc:
            return s_nlc
        if x == pm:
            return s_pm
        if not getattr(x, 'args', None):
            return x
        return x.func(*(_rec(a) for a in x.args))

    return str(_rec(sympify(e)))


def t_to_g(obj, dc: bool = True):
    """
    Convert time variable t_ to parametric gain operator \\hat{g} / (\\chi \\sqrt{\\hat{N}_d})
    and simplify commutative number-operator expressions.
    Independent of TWMSolver; only requires dc to identify the driving mode.
    """
    nd = BosonNum(a_p.name if dc else a_i.name)

    def _simp_n(expr):
        if not isinstance(expr, Expr):
            return expr
        expr = expr.subs({t_: g_hat / (nlc * nd ** S.Half), _g: g_hat})

        def _combine(e):
            if e.is_Mul:
                n_part = []
                other = []
                for arg in Mul.make_args(e.subs({nd: _nd, g_hat: _g})):
                    if arg.has(_nd, _g):
                        n_part.append(arg)
                    else:
                        other.append(arg)
                if n_part:
                    simp_n = _ps(Mul(*n_part)).subs({_nd: nd, _g: g_hat})
                    e = Mul(*other, simp_n)
            return _ps(e)

        return bottom_up(expr, _combine)

    if isinstance(obj, (PF, dict, f_dict)):
        return d_apply(obj, _simp_n)
    return _simp_n(obj)


def g_to_t(obj, dc: bool = True):
    """
    Inverse of t_to_g: convert parametric gain operator \\hat{g} back to \\chi \\sqrt{\\hat{N}_d} t_
    and simplify commutative number-operator expressions.
    Independent of TWMSolver; only requires dc to identify the driving mode.
    """
    nd = BosonNum(a_p.name if dc else a_i.name)

    def _simp_n(expr):
        if not isinstance(expr, Expr):
            return expr
        expr = expr.subs({g_hat: nlc * nd ** S.Half * t_, _g: nlc * nd ** S.Half * t_})

        def _combine(e):
            if e.is_Mul:
                n_part = []
                other = []
                for arg in Mul.make_args(e.subs({nd: _nd})):
                    if arg.has(_nd):
                        n_part.append(arg)
                    else:
                        other.append(arg)
                if n_part:
                    simp_n = _ps(Mul(*n_part)).subs({_nd: nd})
                    e = Mul(*other, simp_n)
            return _ps(e)

        return bottom_up(expr, _combine)

    if isinstance(obj, (PF, dict, f_dict)):
        return d_apply(obj, _simp_n)
    return _simp_n(obj)


def eval_single_mode(mode_type: str, name: str, first_op: str, exp_str: str, action: str) -> Tuple[str, bool, str]:
    """
    Evaluates single-mode quantum operator product (BosonPat or FermionPat).
    Returns: (latex_str, is_hermitian, expr_str)
    """
    name = (name or '').strip()
    if not name:
        name = 'a' if mode_type == 'boson' else 'c'
    is_ann = (first_op == 'annihilation')
    exps = [int(x) for x in re.findall(r'\d+', exp_str or '')]

    if mode_type == 'boson':
        if not exps or all(e == 0 for e in exps):
            orig = patform = norm = S.One
            is_hermitian = True
        else:
            orig = BosonPat(name, is_ann, *exps, eval_exp=False)
            if isinstance(orig, BosonOp):
                patform = norm = orig
                is_hermitian = False
            elif isinstance(orig, BosonNum):
                patform = norm = orig
                is_hermitian = True
            else:
                patform = orig.rewrite(BosonNum)
                norm = orig.to_normal_order(via_dict=True).rewrite(BosonOp)
                is_hermitian = (getattr(orig, 'parity_diff', None) == 0)

        target = orig if action == 'original' else (patform if action == 'patform' else norm)
        return latex(target), is_hermitian, expr_to_code_str(target)

    else:  # fermion
        if not exps or all(e == 0 for e in exps):
            orig = simp = norm = S.One
            is_hermitian = True
        else:
            orig = FermionPat(name, is_ann, *exps, eval_exp=False)
            simp = orig.doit()
            if simp == 0:
                norm = S.Zero
                is_hermitian = True
            elif simp == 1:
                norm = S.One
                is_hermitian = True
            elif isinstance(simp, FermionOp):
                norm = simp
                is_hermitian = False
            elif isinstance(simp, FermionPat) and simp.exps == (1, 1):
                if simp.left_base.is_annihilation:
                    # c c† -> 1 - c† c
                    norm = 1 - FermionPat(name, False, 1, 1)
                    is_hermitian = True
                else:
                    norm = simp
                    is_hermitian = True
            else:
                norm = simp
                is_hermitian = (getattr(simp, 'is_hermitian', None) is True)

        target = orig if action == 'original' else (simp if action == 'patform' else norm)
        target_str = _fermion_target_to_expr_str(target, name)
        return latex(target), is_hermitian, target_str


def _fermion_target_to_expr_str(target, name: str) -> str:
    """Format single-mode fermion result into a valid Python expression string for Tab 2."""
    if target == 0 or target == S.Zero:
        return "0"
    if target == 1 or target == S.One:
        return "1"
    if isinstance(target, FermionOp):
        return name if target.is_annihilation else f"dag({name})"
    if isinstance(target, FermionPat) and target.exps == (1, 1):
        if target.left_base.is_annihilation:
            return f"{name} * dag({name})"
        else:
            return f"dag({name}) * {name}"
    try:
        norm_test = 1 - FermionPat(name, False, 1, 1)
        if target == norm_test:
            return f"1 - dag({name}) * {name}"
    except Exception:
        pass
    return expr_to_code_str(target)


class PFWrapper:
    """
    Wrapper around PFTableProcessor to support operator algebra expressions
    via PFTableProcessor's add, mul_ind/mul_res, and power_ind/power_res methods.
    """
    def __init__(self, pf: PFTableProcessor, res=S.Zero, use_res: bool = False):
        self.pf = pf
        self.res = res
        self.use_res = use_res

    @classmethod
    def from_val(cls, val, use_res: bool = False):
        if isinstance(val, PFWrapper):
            val.use_res = use_res
            return val
        if isinstance(val, PFTableProcessor):
            return cls(val, use_res=use_res)
        expr = sympify(val)
        if getattr(expr, 'is_commutative', True) or not expr.has(BosonOp, BosonPat, FermionOp, FermionPat):
            return cls(PFTableProcessor({((), ()): expr}), use_res=use_res)
        pf = pattern_form(expr, as_dict=None)
        return cls(pf, use_res=use_res)

    def __add__(self, other):
        other = PFWrapper.from_val(other, self.use_res)
        new_pf = PFTableProcessor.add(self.pf, other.pf)
        return PFWrapper(new_pf, self.res + other.res, self.use_res)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = PFWrapper.from_val(other, self.use_res)
        neg_table = {k: -v for k, v in other.pf.table.items()}
        neg_pf = PFTableProcessor(neg_table, -other.pf.npf, other.pf._res_cache)
        new_pf = PFTableProcessor.add(self.pf, neg_pf)
        return PFWrapper(new_pf, self.res - other.res, self.use_res)

    def __rsub__(self, other):
        return PFWrapper.from_val(other, self.use_res).__sub__(self)

    def __neg__(self):
        neg_table = {k: -v for k, v in self.pf.table.items()}
        neg_pf = PFTableProcessor(neg_table, -self.pf.npf, self.pf._res_cache)
        return PFWrapper(neg_pf, -self.res, self.use_res)

    def __mul__(self, other):
        other = PFWrapper.from_val(other, self.use_res)
        if self.use_res or other.use_res:
            new_pf, r = self.pf.mul_res(other.pf)
            total_res = self.res + other.res + r
            return PFWrapper(new_pf, total_res, True)
        else:
            new_pf = self.pf.mul_ind(other.pf)
            return PFWrapper(new_pf, S.Zero, False)

    def __rmul__(self, other):
        other = PFWrapper.from_val(other, self.use_res)
        return other.__mul__(self)

    def __pow__(self, power):
        if isinstance(power, PFWrapper):
            if power.pf.npf == S.Zero and set(power.pf.table.keys()).issubset({((), ())}):
                power = power.pf.table.get(((), ()), S.Zero)
            else:
                raise ValueError("Exponent must be a scalar or number-operator expression.")
        p_sym = sympify(power)
        if self.pf.npf == S.Zero and set(self.pf.table.keys()).issubset({((), ())}):
            base_expr = self.pf.table.get(((), ()), S.Zero)
            return PFWrapper(PFTableProcessor({((), ()): base_expr ** p_sym}), self.res, self.use_res)
        if p_sym.is_Integer and p_sym >= 0:
            exp_int = int(p_sym)
            if self.use_res:
                new_pf, r = self.pf.power_res(exp_int, self.res)
                return PFWrapper(new_pf, r, True)
            else:
                new_pf = self.pf.power_ind(exp_int)
                return PFWrapper(new_pf, S.Zero, False)
        raise ValueError("Operator exponentiation only supports non-negative integers.")

    def __truediv__(self, other):
        if isinstance(other, PFWrapper):
            if other.pf.npf == S.Zero and set(other.pf.table.keys()).issubset({((), ())}):
                val = other.pf.table.get(((), ()), S.One)
                return self.__mul__(PFWrapper.from_val(val ** S.NegativeOne, self.use_res))
            raise ValueError("Division is only supported by scalar or number-operator expressions.")
        return self.__mul__(PFWrapper.from_val(sympify(other) ** S.NegativeOne, self.use_res))

    def __rtruediv__(self, other):
        return PFWrapper.from_val(other, self.use_res).__truediv__(self)

    def adjoint(self):
        new_pf = self.pf.adjoint(res=self.use_res)
        return PFWrapper(new_pf, self.res, self.use_res)

    def diff(self, *symbols, **kwargs):
        resolved = []
        for s in symbols:
            if isinstance(s, PFWrapper):
                if s.pf.npf == S.Zero and set(s.pf.table.keys()).issubset({((), ())}):
                    resolved.append(s.pf.table.get(((), ()), S.Zero))
                else:
                    raise ValueError(f"Cannot differentiate with respect to operator: {s}")
            elif isinstance(s, str):
                resolved.append(Symbol(s))
            else:
                resolved.append(s)

        new_table = {}
        for k, v in self.pf.table.items():
            dv = diff(v, *resolved, **kwargs)
            if dv != 0 and dv != S.Zero:
                new_table[k] = dv
        if not new_table and self.pf.npf == S.Zero:
            new_table = {((), ()): S.Zero}
        new_npf = diff(self.pf.npf, *resolved, **kwargs)
        new_res = diff(self.res, *resolved, **kwargs) if self.res != S.Zero else S.Zero
        return PFWrapper(PFTableProcessor(new_table, new_npf), new_res, self.use_res)

    def Derivative(self, *symbols, **kwargs):
        return self.diff(*symbols, **kwargs)

    def expr(self):
        return self.pf.expr()


def eval_pattern_form(
    text: str,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    use_res: bool = False
) -> Tuple[str, str, Optional[str], Optional[str]]:
    """
    Evaluates quantum expressions using PFTableProcessor add/mul/power arithmetic.
    Returns: (latex_result, expr_str, residue_latex, residue_str)
    """
    text = (text or '').strip()
    if not text:
        raise ValueError("Expression input cannot be empty.")

    def _make_boson_num(item):
        if isinstance(item, PFWrapper):
            name = str(item.pf.expr())
        elif hasattr(item, 'name'):
            name = str(item.name)
        else:
            name = str(item)
        return PFWrapper.from_val(BosonNum(name), use_res)

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, PFWrapper):
                if item.pf.npf == S.Zero and set(item.pf.table.keys()).issubset({((), ())}):
                    arg = item.pf.table.get(((), ()), S.Zero)
                    return PFWrapper(PFTableProcessor({((), ()): fn(arg)}), item.res, use_res)
                return PFWrapper.from_val(fn(item.pf.expr()), use_res)
            return PFWrapper.from_val(fn(sympify(item)), use_res)
        return _inner

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
        'dag': lambda item: PFWrapper.from_val(item, use_res).adjoint(),
        'Dagger': lambda item: PFWrapper.from_val(item, use_res).adjoint(),
        'dagger': lambda item: PFWrapper.from_val(item, use_res).adjoint(),
        'N_': _make_boson_num,
        'BosonNum': _make_boson_num,
        'BP': lambda item, *args: PFWrapper.from_val(BosonPat(str(item), True, *args), use_res),
        'FP': lambda item, *args: PFWrapper.from_val(FermionPat(str(item), True, *args), use_res),
        'diff': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'Derivative': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
    }

    disp_subs = {}

    # Register symbols as scalar PFWrapper
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
        local_dict[s_name] = PFWrapper.from_val(sym_obj, use_res)

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
        w_op = PFWrapper.from_val(op_obj, use_res)
        w_dag = w_op.adjoint()

        local_dict[op_name] = w_op
        local_dict[f'{op_name}d'] = w_dag
        local_dict[f'{op_name}_dag'] = w_dag
        local_dict[f'dag_{op_name}'] = w_dag

        d_name = (op.get('display_name') or '').strip()
        if d_name and d_name != op_name:
            disp_subs[Symbol(op_name)] = Symbol(d_name)

    # Register saved named expressions (evaluate them to obtain their PFTableProcessor)
    for ne in named_expressions:
        ne_name = ne.get('name', '').strip()
        ne_str = (ne.get('expr_str') or ne.get('exprStr') or '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")
        if not ne_str:
            continue
        try:
            parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
            w_ne = PFWrapper.from_val(parsed_ne, use_res)
            local_dict[ne_name] = w_ne
        except Exception as e:
            raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    try:
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS)
        wrapper_target = PFWrapper.from_val(parsed_target, use_res)
    except Exception as e:
        raise ValueError(f"Calculation failed: {e}")

    pf_result = wrapper_target.pf
    expr_result = pf_result.expr()

    # Apply display name substitutions to LaTeX output
    expr_disp = expr_result
    if disp_subs:
        try:
            expr_disp = expr_result.subs(disp_subs)
        except Exception:
            expr_disp = expr_result

    latex_str = latex(expr_disp)
    expr_str = expr_to_code_str(expr_result)

    if use_res:
        residue_expr = wrapper_target.res
        res_disp = residue_expr
        if disp_subs:
            try:
                res_disp = residue_expr.subs(disp_subs)
            except Exception:
                res_disp = residue_expr
        return latex_str, expr_str, latex(res_disp), expr_to_code_str(residue_expr)
    else:
        return latex_str, expr_str, None, None


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
    sol_order = order - 1
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
    sol_d_g = t_to_g(sol_d, dc=bool(dc))
    p_g = t_to_g(p, dc=bool(dc))
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
        dc = s.dc
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
            else:
                var = Symbol(var)
        elif var == t_ or str(var) == '_t':
            is_wrt_t = True
        elif var == g_hat or str(var) in ('g', '_g', r'\hat{g}'):
            is_wrt_g = True

        if is_wrt_g:
            val_in_g = t_to_g(self.val, dc=dc)
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
            diff_in_t = g_to_t(diff_in_g, dc=dc)
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


def eval_operator_combination(
    text: str,
    dc: bool,
    order: int,
    use_gain: bool,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    mm_operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
) -> Tuple[str, str, str, str, str, str]:
    """
    Evaluates post-evolution operator combinations in Tab 4.
    Returns: (latex_str, expr_str)
    """
    text = (text or '').strip()
    if not text:
        raise ValueError("Expression input cannot be empty.")
    if order < -1:
        raise ValueError("Order must be an integer >= -1.")

    dc = bool(dc)
    s = SOLVERS[dc][False]
    sol = s.get_sol(order)

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
    w_g_internal = ComboWrapper(g_to_t(PF({((), ()): g_hat}), dc=dc), s)
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
        'BP': lambda item, *args: ComboWrapper.from_val(BosonPat(op_mode_map.get(str(item), str(item)), True, *args), s),
        'MMN': lambda name, disp=0: ComboWrapper.from_val(MultimodeNum(op_mode_map.get(str(name), str(name)), disp), s),
        'MMN2': lambda disp=0: ComboWrapper.from_val(
            MultimodeNum2(a_s.name, a_i.name, 0, disp) if dc else MultimodeNum2(a_p.name, a_s.name, 1, disp), s
        ),
        't': w_t,
        'g': w_g_internal,
        'diff': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'Derivative': lambda expr, *args, **kwargs: expr.diff(*args, **kwargs) if hasattr(expr, 'diff') else diff(expr, *args, **kwargs),
        'nlc': ComboWrapper.from_val(nlc, s),
        'pm': ComboWrapper.from_val(pm, s),
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

    # 2. Register evolved mode operators (a_p, a_s, a_i or renamed identifiers)
    for op_def in operators:
        op_name = (op_def.get('name') or '').strip()
        valid, err_msg = is_valid_variable_name(op_name)
        if not valid:
            raise ValueError(f"Invalid operator identifier: {err_msg}")
        mode_idx = int(op_def.get('mode_index', 0))
        if 0 <= mode_idx <= 2:
            w_op = evolved_wrappers[mode_idx]
            w_dag = w_op.adjoint()
            local_dict[op_name] = w_op
            local_dict[f'{op_name}d'] = w_dag
            local_dict[f'{op_name}_dag'] = w_dag
            local_dict[f'dag_{op_name}'] = w_dag
            op_mode_map[op_name] = f'a_{mode_idx}'

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
        w_mm_dag = w_mm.adjoint()
        local_dict[mm_name] = w_mm
        local_dict[f'{mm_name}d'] = w_mm_dag
        local_dict[f'{mm_name}_dag'] = w_mm_dag
        local_dict[f'dag_{mm_name}'] = w_mm_dag

    # 4. Register saved named expressions
    for ne in named_expressions:
        ne_name = (ne.get('name') or '').strip()
        ne_str = (ne.get('expr_str') or ne.get('exprStr') or '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")
        if not ne_str:
            continue
        try:
            parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
            w_ne = ComboWrapper.from_val(parsed_ne, s)
            # Ensure internal representation is in terms of t_ for subsequent multiplications
            w_ne = ComboWrapper(g_to_t(w_ne.val, dc=dc), s)
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
        if order >= 0 or p.expr(False).has(t_):
            p = s.expand(p, order)
            return p.n_apply(
                lambda t: s.n_simp(d_p2s(_g, s.p2s_group(t)).subs({_g: s.gc * t_}))
            )
        return p.n_apply(s.n_simp)

    val = _apply_to_pfs(val, _postprocess_pf)

    # 7. Compute both time (t) and gain (g) representations
    val_t = g_to_t(val, dc=dc)
    expr_result_t = val_t.expr(False) if isinstance(val_t, PF) else s.to_expr(val_t)
    expr_str_t = expr_to_code_str(expr_result_t, dc=dc)
    expr_disp_t = expr_result_t
    if disp_subs:
        try:
            expr_disp_t = expr_result_t.subs(disp_subs)
        except Exception:
            expr_disp_t = expr_result_t
    latex_t = latex(expr_disp_t)

    val_g = t_to_g(val, dc=dc)
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

    return latex_curr, expr_curr, latex_t, expr_str_t, latex_g, expr_str_g
