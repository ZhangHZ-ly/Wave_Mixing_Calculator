from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from op_patterns import BosonNum, BosonPat, FermionPat
from perturb_eval import pattern_form, PFTableProcessor, Pat
from sympy.core.function import diff
from sympy.core.numbers import I, Integer, pi
from sympy.core.power import Pow
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

from .common import TRANSFORMS, is_valid_variable_name, expr_to_code_str


class PFWrapper:
    """
    Wrapper around PFTableProcessor to support operator algebra expressions
    via PFTableProcessor's add, mul_ind/mul_rem, and power_ind/power_rem methods.
    """
    def __init__(self, pf: PFTableProcessor, rem=None, use_rem: bool = False):
        if rem is None:
            rem = S.Zero

        self.pf = pf
        self.rem = rem
        self.use_rem = use_rem
        self.name: str = ""
        self.latex_str: Optional[str] = None
        self.remainder_latex: Optional[str] = None
        self.notes: str = ""

    @classmethod
    def from_val(cls, val, use_rem: bool = False):
        if isinstance(val, PFWrapper):
            val.use_rem = use_rem
            return val
        if isinstance(val, PFTableProcessor):
            return cls(val, use_rem=use_rem)
        expr = sympify(val)
        if getattr(expr, 'is_commutative', True) or not expr.has(BosonOp, BosonPat, FermionOp, FermionPat):
            return cls(PFTableProcessor({((), ()): expr}), use_rem=use_rem)
        pf = pattern_form(expr, as_dict=None)
        return cls(pf, use_rem=use_rem)

    def __add__(self, other):
        other = PFWrapper.from_val(other, self.use_rem)
        new_pf = PFTableProcessor.add(self.pf, other.pf)
        return PFWrapper(new_pf, self.rem + other.rem, self.use_rem)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = PFWrapper.from_val(other, self.use_rem)
        neg_table = {k: -v for k, v in other.pf.table.items()}
        neg_pf = PFTableProcessor(neg_table, -other.pf.npf, other.pf._rem_cache)
        new_pf = PFTableProcessor.add(self.pf, neg_pf)
        return PFWrapper(new_pf, self.rem - other.rem, self.use_rem)

    def __rsub__(self, other):
        return PFWrapper.from_val(other, self.use_rem).__sub__(self)

    def __neg__(self):
        neg_table = {k: -v for k, v in self.pf.table.items()}
        neg_pf = PFTableProcessor(neg_table, -self.pf.npf, self.pf._rem_cache)
        return PFWrapper(neg_pf, -self.rem, self.use_rem)

    def __mul__(self, other):
        other = PFWrapper.from_val(other, self.use_rem)
        if self.use_rem or other.use_rem:
            new_pf, r = self.pf.mul_rem(other.pf)
            total_rem = self.rem + other.rem + r
            return PFWrapper(new_pf, total_rem, True)
        else:
            new_pf = self.pf.mul_ind(other.pf)
            return PFWrapper(new_pf, S.Zero, False)

    def __rmul__(self, other):
        other = PFWrapper.from_val(other, self.use_rem)
        return other.__mul__(self)

    def __pow__(self, power):
        if isinstance(power, PFWrapper):
            if power.pf.npf == S.Zero and set(power.pf.table.keys()).issubset({((), ())}):
                power = power.pf.table.get(((), ()), S.Zero)
            else:
                exp_expr = power.pf.expr()
                return PFWrapper(PFTableProcessor(npf=Pow(self.pf.expr(), exp_expr)), S.Zero, self.use_rem)
        p_sym = sympify(power)
        if self.pf.npf == S.Zero and set(self.pf.table.keys()).issubset({((), ())}):
            base_expr = self.pf.table.get(((), ()), S.Zero)
            return PFWrapper(PFTableProcessor({((), ()): base_expr ** p_sym}), self.rem, self.use_rem)
        if getattr(p_sym, 'is_Integer', False) and p_sym >= 0:
            exp_int = int(p_sym)
            try:
                if self.use_rem:
                    new_pf, r = self.pf.power_rem(exp_int, self.rem)
                    return PFWrapper(new_pf, r, True)
                else:
                    new_pf = self.pf.power_ind(exp_int)
                    return PFWrapper(new_pf, S.Zero, False)
            except Exception:
                return PFWrapper(PFTableProcessor(npf=Pow(self.pf.expr(), p_sym)), S.Zero, self.use_rem)
        return PFWrapper(PFTableProcessor(npf=Pow(self.pf.expr(), p_sym)), S.Zero, self.use_rem)

    def __truediv__(self, other):
        if isinstance(other, PFWrapper):
            if other.pf.npf == S.Zero and set(other.pf.table.keys()).issubset({((), ())}):
                val = other.pf.table.get(((), ()), S.One)
                return self.__mul__(PFWrapper.from_val(val ** S.NegativeOne, self.use_rem))
            raise ValueError("Division is only supported by scalar or number-operator expressions.")
        return self.__mul__(PFWrapper.from_val(sympify(other) ** S.NegativeOne, self.use_rem))

    def __rtruediv__(self, other):
        return PFWrapper.from_val(other, self.use_rem).__truediv__(self)

    def adjoint(self):
        new_pf = self.pf.adjoint(rem=self.use_rem)
        return PFWrapper(new_pf, self.rem, self.use_rem)

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
        new_rem = diff(self.rem, *resolved, **kwargs) if self.rem != S.Zero else S.Zero
        return PFWrapper(PFTableProcessor(new_table, new_npf), new_rem, self.use_rem)

    def Derivative(self, *symbols, **kwargs):
        return self.diff(*symbols, **kwargs)

    def expr(self):
        return self.pf.expr()


PF_VARIABLES: Dict[str, PFWrapper] = {}
LAST_EVAL_PF: Dict[str, Any] = {}


def save_pf_variable(name: str, pf: Any, rem: Any = None, use_rem: bool = False, res: Any = None, **kwargs) -> PFWrapper:
    val_rem = rem if rem is not None else (res if res is not None else S.Zero)
    w = PFWrapper.from_val(pf, use_rem=use_rem)
    w.rem = val_rem
    w.res = val_rem
    w.name = name
    for k, v in kwargs.items():
        setattr(w, k, v)
    PF_VARIABLES[name] = w
    return w


def save_latest_pf_variable(name: str, **kwargs) -> Optional[PFWrapper]:
    if not LAST_EVAL_PF:
        return None
    return save_pf_variable(
        name,
        LAST_EVAL_PF['pf'],
        rem=LAST_EVAL_PF.get('rem', LAST_EVAL_PF.get('res', S.Zero)),
        latex_str=LAST_EVAL_PF.get('latex_str'),
        remainder_latex=LAST_EVAL_PF.get('remainder_latex'),
        **kwargs
    )


def get_pf_variable(name: str) -> Optional[PFWrapper]:
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
    Evaluates quantum expressions using PFTableProcessor add/mul/power arithmetic.
    Returns: (latex_result, expr_str, remainder_latex, remainder_str)
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
        return PFWrapper.from_val(BosonNum(name), use_rem)

    mode_kind_map = {op.get('name', '').strip(): op.get('kind', 'boson') for op in operators if op.get('name')}
    op_mode_map = {
        op.get('name', '').strip(): (op.get('display_name') or op.get('displayName') or '').strip()
        for op in operators if op.get('name')
    }

    def _wrap_fn(fn):
        def _inner(item):
            if isinstance(item, PFWrapper):
                if item.pf.npf == S.Zero and set(item.pf.table.keys()).issubset({((), ())}):
                    arg = item.pf.table.get(((), ()), S.Zero)
                    return PFWrapper(PFTableProcessor({((), ()): sympify(fn(arg))}), item.rem, use_rem)
                return PFWrapper.from_val(fn(item.pf.expr()), use_rem)
            return PFWrapper.from_val(fn(sympify(item)), use_rem)
        return _inner

    def _make_bp(item, *args):
        raw_args = []
        for a in args:
            if isinstance(a, (tuple, list)):
                raw_args.extend(x if isinstance(x, bool) else (x.pf.expr() if isinstance(x, PFWrapper) else sympify(x)) for x in a)
            else:
                raw_args.append(a if isinstance(a, bool) else (a.pf.expr() if isinstance(a, PFWrapper) else sympify(a)))
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        return PFWrapper.from_val(BosonPat(name, *raw_args), use_rem)

    def _make_fp(item, *args):
        raw_args = []
        for a in args:
            if isinstance(a, (tuple, list)):
                raw_args.extend(x if isinstance(x, bool) else (x.pf.expr() if isinstance(x, PFWrapper) else sympify(x)) for x in a)
            else:
                raw_args.append(a if isinstance(a, bool) else (a.pf.expr() if isinstance(a, PFWrapper) else sympify(a)))
        name = item.name if hasattr(item, 'name') else str(item)
        if op_mode_map and name in op_mode_map:
            name = str(op_mode_map[name])
        return PFWrapper.from_val(FermionPat(name, *raw_args), use_rem)

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
        'dag': lambda item: PFWrapper.from_val(item, use_rem).adjoint(),
        'Dagger': lambda item: PFWrapper.from_val(item, use_rem).adjoint(),
        'dagger': lambda item: PFWrapper.from_val(item, use_rem).adjoint(),
        'N_': _make_boson_num,
        'BosonNum': _make_boson_num,
        'Pat': lambda item, *args: PFWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), use_rem
        ),
        'pattern': lambda item, *args: PFWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), use_rem
        ),
        'BP': _make_bp,
        'BosonPat': _make_bp,
        'FP': _make_fp,
        'FermionPat': _make_fp,
        'Pow': lambda b, e: (b ** e) if isinstance(b, PFWrapper) else ((PFWrapper.from_val(b, use_rem)) ** e),
        'pow': lambda b, e: (b ** e) if isinstance(b, PFWrapper) else ((PFWrapper.from_val(b, use_rem)) ** e),
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
        local_dict[s_name] = PFWrapper.from_val(sym_obj, use_rem)

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
        w_op = PFWrapper.from_val(op_obj, use_rem)

        local_dict[op_name] = w_op

        d_name = (op.get('display_name') or '').strip()
        if d_name and d_name != op_name:
            disp_subs[Symbol(op_name)] = Symbol(d_name)

    # Register saved named expressions directly from PFTableProcessor
    for ne in named_expressions:
        ne_name = ne.get('name', '').strip()
        valid, err_msg = is_valid_variable_name(ne_name)
        if not valid:
            raise ValueError(f"Invalid expression identifier: {err_msg}")

        if 'val' in ne and ne['val'] is not None:
            val_obj = ne['val']
            if isinstance(val_obj, PFWrapper):
                w_ne = val_obj
            elif isinstance(val_obj, PFTableProcessor):
                w_ne = PFWrapper(val_obj, rem=ne.get('rem', S.Zero), use_rem=use_rem)
            elif isinstance(val_obj, tuple):
                w_ne = PFWrapper(val_obj[0], rem=val_obj[1], use_rem=use_rem)
            else:
                w_ne = PFWrapper.from_val(val_obj, use_rem)
            local_dict[ne_name] = w_ne
            save_pf_variable(ne_name, w_ne.pf, rem=w_ne.rem, latex_str=ne.get('latex'))
        elif ne_name in PF_VARIABLES:
            local_dict[ne_name] = PF_VARIABLES[ne_name]
        else:
            ne_str = (ne.get('expr_str') or ne.get('source_input') or ne.get('sourceInput') or '').strip()
            if not ne_str:
                continue
            try:
                parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
                w_ne = PFWrapper.from_val(parsed_ne, use_rem)
                local_dict[ne_name] = w_ne
                save_pf_variable(ne_name, w_ne.pf, rem=w_ne.rem, latex_str=ne.get('latex'))
            except Exception as e:
                raise ValueError(f"Failed to evaluate named expression '{ne_name}': {e}")

    try:
        parsed_target = parse_expr(text, local_dict=local_dict, transformations=TRANSFORMS)
        wrapper_target = PFWrapper.from_val(parsed_target, use_rem)
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

    remainder_expr = wrapper_target.rem if use_rem else S.Zero
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
        rem_str = expr_to_code_str(remainder_expr)

    global LAST_EVAL_PF
    LAST_EVAL_PF = {
        'pf': pf_result,
        'rem': remainder_expr,
        'res': remainder_expr,
        'latex_str': latex_str,
        'remainder_latex': rem_latex,
    }

    return latex_str, expr_str, rem_latex, rem_str



# Backwards compatibility alias
eval_pattern_form = eval_pf_form
