"""
Cascaded wave-mixing evolution for Tab 3.

A calculated expression E (written in the t = 0 input operators a_0, a_1, a_2) is evolved
for a further time t_c by substituting every input-operator object with its evolved
counterpart at t_c:
- E is split into terms of definite perturbative order by WMWrapper.expand(..., with_order=True);
- in each term, the parts built from a_0, a_1, a_2 inside the number-operator tails
  (n_drive, multimode numbers) are replaced by dummies and lambdified into a handle,
  which is expanded by WMProcessor.expand_func_d around the evolved number operators;
- the operator parts (single-mode keys, multimode COP products, driving-field patterns)
  are multiplied out from the evolved operators.
Evolved multimode objects and n_drive are computed by multimode products (@) and cached
in evo_cache, per solver and solution order.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

from cas_qwm.op_pattern import BosonNum, BosonPat
from cas_qwm.pattern_perturb import PF, PF1
from cas_qwm.pattern_perturb.pattern_form import _vpk
from cas_qwm.parametric_multimode import (
    FWMSolver, MultimodeCOP, MultimodeNum, MultimodeNum2, WMSolver, WMWrapper,
)
from cas_qwm.parametric_multimode.wm_processor import t_, nm, a_p, a_s, a_i
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.physics.quantum.boson import BosonOp
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.printing.pycode import SymPyPrinter
from sympy.utilities.lambdify import lambdify

from .common import is_valid_variable_name, symbol_assumptions
from .wave_mixing import WMEvalContext, _prepare_wm_evaluation, finalize_wm_result, get_evolved_sol
from .wm_transforms import apply_assertions

__all__ = ['tc', 'evo_cache', 'cascade_evolve', 'resolve_cascade_symbol', 'eval_cascaded_combination']

tc = Dummy('t_c', nonnegative=True)  # further evolution time of the cascaded stage
# {(solver, solution order): {input-operator object: evolved WMWrapper at t_c}}
evo_cache = defaultdict(dict)

_MODE_INDEX = {a_p.name: 0, a_s.name: 1, a_i.name: 2}
_MODE_ATOMS = (BosonNum, MultimodeNum, MultimodeNum2)


class _ExactPrinter(SymPyPrinter):
    """
    lambdify printer keeping numbers exact: the default prints rationals as Python
    float divisions and I as 1j, which would break the order homogeneity of the handle.
    """
    def _print_Integer(self, expr):
        return f'Integer({expr.p})'

    def _print_Rational(self, expr):
        return f'Rational({expr.p}, {expr.q})'

    _print_Half = _print_Rational
    _print_One = _print_Zero = _print_NegativeOne = _print_Integer

    def _print_ImaginaryUnit(self, expr):
        return 'I'

    def _print_Pi(self, expr):
        return 'pi'

    def _print_Exp1(self, expr):
        return 'E'


_LAMBDIFY_PRINTER = _ExactPrinter({'fully_qualified_modules': False, 'inline': True,
                                   'allow_unknown_functions': True, 'user_functions': {}})
_LAMBDIFY_MODULES = [{'Integer': Integer, 'Rational': Rational}, 'sympy']


class _Evolver:
    """
    Evolves input-operator objects of one solver to time t_c, caching the results.

    Every evolved object X of leading order lo_X is truncated at the absolute order
    lo_X + rel, where `rel` is the largest relative accuracy needed by any term
    (target order minus the lowest term order), so that no product carries terms
    beyond the retained order.
    """

    def __init__(self, s: WMSolver, sol_order: int, rel):
        self.s = s
        self.p = s.processor
        self.sol_order = sol_order
        self.rel = sympify(rel)
        self.cache = evo_cache[(s, sol_order, self.rel)]
        self._sol = None
        self.weak_names = set(self.p.a[0].names)

    def lo(self, name):
        """Leading order of a single-mode operator of the given mode."""
        return self.p.base_orders.get(name, S.Zero)

    def trunc(self, w: WMWrapper, lo) -> WMWrapper:
        return w.expand(lo + self.rel)

    @property
    def sol(self) -> tuple:
        if self._sol is None:
            self._sol = tuple(w.apply(lambda e: e.subs(t_, tc), on_npf=True)
                              for w in get_evolved_sol(self.s, self.sol_order))
        return self._sol

    def _cached(self, key, build):
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]

    def op(self, b: BosonOp) -> WMWrapper:
        """Single-mode input operator a_k or a_k^dagger."""
        def build():
            w = self.sol[_MODE_INDEX[b.name]]
            return self.trunc(w if b.is_annihilation else w.adjoint(), self.lo(b.name))
        return self._cached(b, build)

    def cop(self, c: MultimodeCOP) -> WMWrapper:
        """Multimode canonical operator, as the multimode product of its two evolved modes."""
        def build():
            o1, o2 = Mul.make_args(c.mode_expr)
            return self.trunc(self.op(o1) @ self.op(o2), S.Zero)
        return self._cached(c, build)

    def mm_num(self, name) -> WMWrapper:
        """Multimode number operator Σ N_name (zero displacement)."""
        return self._cached(MultimodeNum(name), lambda: self.trunc(
            self.op(BosonOp(name, False)) @ self.op(BosonOp(name)), S.Zero))

    def atom(self, x) -> WMWrapper:
        """Number-like input-operator object appearing in a PF tail."""
        def build():
            if isinstance(x, BosonNum):
                dag, ann = self.op(BosonOp(x.name, False)), self.op(BosonOp(x.name))
                # n_drive is a multimode product; a weak-field BosonNum is single-mode
                prod = dag * ann if x.name in self.weak_names else dag @ ann
                return self.trunc(prod, 2 * self.lo(x.name))
            if isinstance(x, MultimodeNum):
                return self.mm_num(x.name) + x.displacement * nm
            # MultimodeNum2: decomposed into multimode number operators
            coeffs = x.bn_coeffs
            const = sympify(x.mode_expr - Add(*(c * BosonNum(n) for n, c in coeffs.items())))
            return self.p.d_add(*(c * self.mm_num(n) for n, c in coeffs.items()), const * nm)
        return self._cached(x, build)

    def pattern(self, b: BosonPat) -> WMWrapper:
        """Boson pattern a^e1 a^dagger^e2 ... of a PF key, as single-mode products."""
        def build():
            lo = self.lo(b.name)
            res, k = None, 0
            is_ann = bool(b.left_base.is_annihilation)
            for j, e in enumerate(b.exps):
                op = self.op(BosonOp(b.name, is_ann ^ bool(j & 1)))
                for _ in range(int(e)):
                    k += 1  # a partial product of k factors is needed up to order k * lo + rel
                    res = op if res is None else self.trunc(res * op, k * lo)
            return res if res is not None else WMWrapper(self.p, PF1)
        return self._cached(b, build)

    def key(self, k: tuple) -> WMWrapper:
        """Multimode exponent tuple T_k = Π COP^e (alternating)."""
        if k == (None,):
            return WMWrapper(self.p, PF1)
        def build():
            i, *exps = k
            res = WMWrapper(self.p, PF1)
            for e in exps:
                for _ in range(int(e)):
                    res = self.trunc(res @ self.cop(self.p.a[i]), S.Zero)
                i = not i
            return res
        return self._cached(k, build)

    def is_mode_atom(self, x) -> bool:
        return getattr(x, 'name', None) in _MODE_INDEX or isinstance(x, (MultimodeNum, MultimodeNum2))

    def atom_leading(self, x) -> Tuple[bool, Any]:
        """
        (whether the leading part of the evolved atom is a scalar / number-operator PF,
        leading order of the evolved atom).
        """
        def build():
            w = self.atom(x)
            if w.b_left is not None or not w.dict_repr:
                return False, S.Zero
            los = {k: self.p.pe.lo_pf(v) for k, v in w.dict_repr.items()}
            m = min(o for _, o in los.values())
            lead = {k: lp for k, (lp, o) in los.items() if o == m}
            lp = lead.get((None,))
            scalar = (set(lead) == {(None,)} and lp.npf == 0 and set(lp.table) <= {_vpk})
            return scalar, m
        return self._cached(('leading', x), build)

    def func(self, expr: Expr, atoms: dict, order) -> WMWrapper:
        """
        Lambdifies expr in the dummies of `atoms` ({dummy: atom}, scalar-leading evolved atoms)
        into a handle and expands it around the evolved atoms by expand_func_d.
        """
        p = self.p
        xs = tuple(sorted(expr.free_symbols & set(atoms), key=default_sort_key))
        if not xs:
            return WMWrapper(p, PF({_vpk: expr.xreplace(atoms)}))
        others = tuple(sorted(expr.free_symbols - set(xs), key=default_sort_key))
        f = lambdify((*xs, *others), expr, modules=_LAMBDIFY_MODULES,
                     printer=_LAMBDIFY_PRINTER, dummify=True)
        handle = lambda *args: f(*args, *others)
        return p.expand_func_d(handle, order, *(self.atom(atoms[x]) for x in xs))

    def tail(self, tail: Expr, order) -> WMWrapper:
        """
        Evolves a number-operator tail up to the absolute order `order`. The input-operator
        objects are replaced by dummies: those whose evolved forms have a scalar leading part
        (e.g. n_drive) enter the lambdified handle of expand_func_d; the others (weak-field and
        multimode numbers, mixing with the multimode COPs) must appear polynomially and are
        multiplied out from their evolved forms.
        """
        p = self.p
        atoms = sorted((x for x in tail.atoms(*_MODE_ATOMS) if self.is_mode_atom(x)),
                       key=default_sort_key)
        if not atoms:
            return WMWrapper(p, PF({_vpk: tail}))
        func_atoms, poly_atoms = {}, {}
        for x in atoms:
            (func_atoms if self.atom_leading(x)[0] else poly_atoms)[Dummy()] = x
        expr = tail.xreplace({x: d for d, x in (*func_atoms.items(), *poly_atoms.items())})
        if not poly_atoms:
            return self.func(expr, func_atoms, order)

        ys = tuple(poly_atoms)
        try:
            terms = Poly(expr, *ys).terms()
        except PolynomialError:
            raise ValueError('Cascaded evolution requires the weak-field and multimode number '
                             'operators to appear polynomially in the expression')
        parts = []
        for monom, coeff in terms:
            w_mon = WMWrapper(p, PF1)
            mon_o = S.Zero
            for y, m in zip(ys, monom):
                if m:
                    w_mon = w_mon @ (self.atom(poly_atoms[y]) ** m)
                    mon_o += m * self.atom_leading(poly_atoms[y])[1]
            parts.append(self.func(coeff, func_atoms, order - mon_o) @ w_mon)
        return p.d_add(*parts)

    def npf(self, npf: Expr) -> WMWrapper:
        if any(self.is_mode_atom(x) for x in npf.atoms(BosonOp, BosonPat, MultimodeNum,
                                                         MultimodeNum2, MultimodeCOP)):
            raise ValueError('Cascaded evolution does not support non-pattern-form (npf) terms '
                             'containing the wave-mixing modes')
        return WMWrapper(self.p, PF(npf=npf))

    def tuple_dict(self, d: dict, order) -> WMWrapper:
        """Evolves a {tuple: PF} dict term by term: T_k · bs · tail · fs."""
        p, pe = self.p, self.p.pe
        parts = []
        for k, pf_v in d.items():
            wk = self.key(k)
            if pf_v.npf != 0:
                parts.append(wk @ self.npf(pf_v.npf))
            for (bs, fs), tail in pf_v.table.items():
                bs_los = [pe.leading_order(b)[1] for b in bs]
                fs_lo = Add(*(pe.leading_order(f)[1] for f in fs))
                tail_lo = pe.leading_order(tail)[1]
                ops_o = Add(*bs_los, fs_lo)
                # Every partial product is truncated at the order left by the factors still
                # to come. The patterns of a PF key form a single-mode product (*); the
                # multimode key, the pattern product, the tail and the fermion patterns are
                # joined by multimode products (@).
                w = wk
                if bs:
                    w_bs = None
                    for j, b in enumerate(bs):
                        w_b = (self.pattern(b) if self.is_mode_atom(b)
                               else WMWrapper(p, PF({((b,), ()): S.One})))
                        rest = Add(*bs_los[j + 1:], tail_lo, fs_lo)
                        w_bs = w_b if w_bs is None else (w_bs * w_b).expand(order - rest)
                    w = (w @ w_bs).expand(order - tail_lo - fs_lo)
                if w:
                    w = (w @ self.tail(tail, order - ops_o)).expand(order - fs_lo)
                if w and fs:
                    w = (w @ WMWrapper(p, PF({((), fs): S.One}))).expand(order)
                if w:
                    parts.append(w)
        return p.d_add(*parts)

    def wrapper(self, w: WMWrapper, order) -> WMWrapper:
        if w.b_left is None:
            return self.tuple_dict(w.dict_repr, order)
        parts = []
        for b, d in w.dict_repr.items():
            lo_b = self.lo(b.name)
            wb, wd = self.op(b), self.tuple_dict(d, order - lo_b)
            parts.append((wb @ wd if w.b_left else wd @ wb).expand(order))
        return self.p.d_add(*parts)


def _required_sol_order(s: WMSolver, pieces: dict, order) -> int:
    """
    Solution order of the evolved operators needed for an absolute accuracy `order`:
    an evolved operator Y of leading order lo_Y inside a term of order o is needed up to
    order - o + lo_Y (lo_Y = 0 for the weak fields, the drive order for drive-only terms).
    """
    p = s.processor
    weak_names = set(p.a[0].names)
    drive_lo = p.base_orders[p.nd.name]
    req = -1
    for o, ws in pieces.items():
        has_weak = any(
            (w.b_left is not None) or any(k != (None,) for k in w.dict_repr)
            or any(getattr(x, 'name', None) in weak_names or isinstance(x, (MultimodeNum, MultimodeNum2))
                   for x in w.expr.atoms(BosonOp, BosonPat, MultimodeNum, MultimodeNum2))
            for w in ws
        )
        need = order - o + (S.Zero if has_weak else drive_lo)
        req = max(req, int(math.ceil(float(need) - 1e-12)))
    if isinstance(s, FWMSolver):
        req = min(1, req)
    return req


def cascade_evolve(ctx: WMEvalContext, order: int) -> WMWrapper:
    """
    Evolves the parsed expression of ctx for a further time t_c (the Dummy `tc`),
    retaining terms up to the given absolute perturbative order.
    """
    s = ctx.solver
    p = s.processor
    pieces = defaultdict(list)
    for w, o in ctx.val.expand(order, with_order=True):
        pieces[o].append(w)
    if not pieces:
        return WMWrapper(p)

    rel = order - min(pieces)  # largest relative accuracy needed by any term
    ev = _Evolver(s, _required_sol_order(s, pieces, order), rel)
    return p.d_add(*(ev.wrapper(p.d_add(*ws), order) for ws in pieces.values()))


def resolve_cascade_symbol(name: str, ctx: WMEvalContext) -> Expr:
    """
    Returns the symbol substituted for t_c: the time t, a user-defined commutative
    symbol, or a new nonnegative commutative symbol. Names of other preset objects
    and of defined expressions / operators are rejected.
    """
    name = (name or '').strip()
    if name == 't':
        return t_
    valid, err_msg = is_valid_variable_name(name)
    if not valid:
        raise ValueError(f"Invalid cascaded evolution time: {err_msg}")
    for sym_def in ctx.symbols:
        if (sym_def.get('name') or '').strip() == name:
            if sym_def.get('builtin_key') or name in ('nlc', 'pm'):
                raise ValueError(f"Cascaded evolution time '{name}' conflicts with a built-in parameter.")
            return Symbol(name, **symbol_assumptions(sym_def))
    if name in ctx.local_dict:
        raise ValueError(
            f"Cascaded evolution time '{name}' conflicts with a preset object or a defined "
            f"expression / operator. Use 't', a custom commutative symbol, or a new name."
        )
    return Symbol(name, nonnegative=True)


def eval_cascaded_combination(
    text: str,
    dc: bool,
    order: int,
    use_gain: bool,
    symbols: List[Dict[str, Any]],
    operators: List[Dict[str, Any]],
    mm_operators: List[Dict[str, Any]],
    named_expressions: List[Dict[str, Any]],
    cascade_symbol: str,
    mismatch: bool = False,
    single_mode: bool = False,
    coherent_drive: bool = False,
    vacuum_signal: bool = False,
    vacuum_other_weak: bool = False,
    process: Optional[str] = None,
) -> Tuple[str, str, str, Union[int, float], int]:
    """
    Evaluates an operator combination evolved for a further time `cascade_symbol`
    (cascaded system), optionally under input-state assertions.
    """
    ctx = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode, process=process
    )
    t_sym = resolve_cascade_symbol(cascade_symbol, ctx)
    val = cascade_evolve(ctx, order).apply(lambda e: e.subs(tc, t_sym), on_npf=True)
    val_t, val_g = apply_assertions(
        val, ctx.solver, order, process=ctx.proc, mismatch=ctx.mismatch, single_mode=single_mode,
        coherent_drive=coherent_drive, vacuum_signal=vacuum_signal, vacuum_other_weak=vacuum_other_weak,
        times=(t_sym,),
    )
    return finalize_wm_result(ctx, val_t, val_g, order, use_gain)
