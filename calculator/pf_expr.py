from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

from op_patterns import BosonNum, BosonPat, FermionPat
from perturb_eval import pattern_form, PFTableProcessor, Pat
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

from .common import TRANSFORMS, is_valid_variable_name, expr_to_code_str


class PFWrapper:
    """
    Wrapper around PFTableProcessor to support operator algebra expressions
    via PFTableProcessor's add, mul_ind/mul_res, and power_ind/power_res methods.
    """
    def __init__(self, pf: PFTableProcessor, res=S.Zero, use_res: bool = False):
        self.pf = pf
        self.res = res
        self.use_res = use_res
        self.name: str = ""
        self.latex_str: Optional[str] = None
        self.residue_latex: Optional[str] = None
        self.notes: str = ""

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


PF_VARIABLES: Dict[str, PFWrapper] = {}
LAST_EVAL_PF: Dict[str, Any] = {}


def save_pf_variable(name: str, pf: Any, res: Any = S.Zero, use_res: bool = False, **kwargs) -> PFWrapper:
    w = PFWrapper.from_val(pf, use_res=use_res)
    w.res = res
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
        res=LAST_EVAL_PF.get('res', S.Zero),
        latex_str=LAST_EVAL_PF.get('latex_str'),
        residue_latex=LAST_EVAL_PF.get('residue_latex'),
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
                    return PFWrapper(PFTableProcessor({((), ()): fn(arg)}), item.res, use_res)
                return PFWrapper.from_val(fn(item.pf.expr()), use_res)
            return PFWrapper.from_val(fn(sympify(item)), use_res)
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
        return PFWrapper.from_val(BosonPat(name, *raw_args), use_res)

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
        return PFWrapper.from_val(FermionPat(name, *raw_args), use_res)

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
        'Pat': lambda item, *args: PFWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), use_res
        ),
        'pattern': lambda item, *args: PFWrapper.from_val(
            Pat(item, *args, mode_kind_map=mode_kind_map, op_mode_map=op_mode_map), use_res
        ),
        'BP': _make_bp,
        'BosonPat': _make_bp,
        'FP': _make_fp,
        'FermionPat': _make_fp,
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
                w_ne = PFWrapper(val_obj, res=ne.get('res', S.Zero), use_res=use_res)
            elif isinstance(val_obj, tuple):
                w_ne = PFWrapper(val_obj[0], res=val_obj[1], use_res=use_res)
            else:
                w_ne = PFWrapper.from_val(val_obj, use_res)
            local_dict[ne_name] = w_ne
            save_pf_variable(ne_name, w_ne.pf, res=w_ne.res, latex_str=ne.get('latex'))
        elif ne_name in PF_VARIABLES:
            local_dict[ne_name] = PF_VARIABLES[ne_name]
        else:
            ne_str = (ne.get('expr_str') or ne.get('source_input') or ne.get('sourceInput') or '').strip()
            if not ne_str:
                continue
            try:
                parsed_ne = parse_expr(ne_str, local_dict=local_dict, transformations=TRANSFORMS)
                w_ne = PFWrapper.from_val(parsed_ne, use_res)
                local_dict[ne_name] = w_ne
                save_pf_variable(ne_name, w_ne.pf, res=w_ne.res, latex_str=ne.get('latex'))
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

    residue_expr = wrapper_target.res if use_res else S.Zero
    res_latex = None
    res_str = None
    if use_res:
        res_disp = residue_expr
        if disp_subs:
            try:
                res_disp = residue_expr.subs(disp_subs)
            except Exception:
                res_disp = residue_expr
        res_latex = latex(res_disp)
        res_str = expr_to_code_str(residue_expr)

    global LAST_EVAL_PF
    LAST_EVAL_PF = {
        'pf': pf_result,
        'res': residue_expr,
        'latex_str': latex_str,
        'residue_latex': res_latex,
    }

    return latex_str, expr_str, res_latex, res_str



# Backwards compatibility alias
eval_pattern_form = eval_pf_form
