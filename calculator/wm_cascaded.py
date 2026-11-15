"""
Cascaded wave-mixing evolution for Tab 3.

A calculated expression E (written in the t = 0 input operators a_0, a_1, a_2) is evolved
for a further time t_c by substituting every input-operator object with its evolved
counterpart at t_c, each input operator a_k carrying an optional real phase phi_k
(a_k -> exp(i phi_k) a_k(t_c); the multimode COPs take the phases of their two modes, i.e. the
phases are assumed to be the same for all the modes of a field):
- E is split into terms of definite perturbative order by WMWrapper.expand(..., with_order=True);
- in each term, the parts built from a_0, a_1, a_2 inside the number-operator tails
  (n_drive, multimode numbers) are replaced by dummies and lambdified into a handle,
  which is expanded by WMProcessor.expand_func_d around the evolved number operators;
- the operator parts (single-mode keys, multimode COP products, driving-field patterns)
  are multiplied out from the evolved operators.
Evolved multimode objects and n_drive are computed by multimode products (@) and cached
in evo_cache, per solver, solution order, relative accuracy and phases.
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
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import E, Float, I, Integer, Rational, pi
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import exp
from sympy.parsing.sympy_parser import parse_expr
from sympy.physics.quantum.boson import BosonOp
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.printing.pycode import SymPyPrinter
from sympy.utilities.lambdify import lambdify

from .common import is_valid_variable_name, symbol_assumptions
from .wave_mixing import (WMEvalContext, _prepare_wm_evaluation, coherent_alpha, finalize_wm_result,
                          get_evolved_sol)
from .wm_transforms import apply_assertions

__all__ = ['tc', 'evo_cache', 'cascade_evolve', 'resolve_cascade_symbol', 'resolve_cascade_phase',
           'eval_cascaded_combination']

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

    With `degenerate`, the evolved operators are those of the degenerate single mode a_1
    (signal = idler), which carries the phase phi_1 (phi_2 is not used).
    """

    def __init__(self, s: WMSolver, sol_order: int, rel, phases=(S.Zero, S.Zero, S.Zero),
                 degenerate: bool = False):
        self.s = s
        self.p = s.processor
        self.sol_order = sol_order
        self.rel = sympify(rel)
        self.phases = tuple(sympify(ph) for ph in phases)
        # degenerate: the evolved operators are those of the degenerate single mode a_1
        self.degenerate = bool(degenerate)
        self.cache = evo_cache[(s, sol_order, self.rel, self.phases, self.degenerate)]
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
            sol = get_evolved_sol(self.s, self.sol_order)
            if self.degenerate:
                sol = tuple(w.to_single_mode(degenerate=True) for w in sol)
            self._sol = tuple(w.apply(lambda e: e.subs(t_, tc), on_npf=True) for w in sol)
        return self._sol

    def _cached(self, key, build):
        if key not in self.cache:
            self.cache[key] = build()
        return self.cache[key]

    def op(self, b: BosonOp) -> WMWrapper:
        """
        Single-mode input operator a_k or a_k^dagger, evolved and with its phase:
        exp(i phi_k) a_k(t_c) or exp(-i phi_k) a_k^dagger(t_c).
        """
        def build():
            k = _MODE_INDEX[b.name]
            w = self.sol[k]
            w = self.trunc(w if b.is_annihilation else w.adjoint(), self.lo(b.name))
            phi = self.phases[k]
            if phi != 0:
                factor = exp(I * phi if b.is_annihilation else -I * phi)
                w = w.apply(lambda e: e * factor, on_npf=True)
            return w
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
            lead, m = self.atom(x).leading()
            return lead is not None and lead.as_scalar is not None, (m if lead is not None else S.Zero)
        return self._cached(('leading', x), build)

    def has_nonscalar(self, expr: Expr) -> bool:
        """Whether expr contains atoms whose evolved forms do not have a scalar leading part."""
        return any(self.is_mode_atom(a) and not self.atom_leading(a)[0]
                   for a in expr.atoms(*_MODE_ATOMS))

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
        Evolves a number-operator expression up to the absolute order `order`. The input-operator
        objects are replaced by dummies: those whose evolved forms have a scalar leading part
        (e.g. n_drive) enter a lambdified handle of expand_func_d; the others (weak-field and
        multimode numbers, mixing with the multimode COPs) are multiplied out from their evolved
        forms where they appear polynomially. Otherwise the expression is decomposed (see
        `structural`).
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
            return self.structural(tail, order)
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

    def structural(self, x: Expr, order) -> WMWrapper:
        """
        Evolves a number-operator expression that is not polynomial in the atoms without scalar
        leading parts, node by node up to the absolute order `order`:
        - sums and products are evolved term / factor-wise (the factors free of such atoms are
          kept together in one handle);
        - positive integer powers are multiplied out;
        - other powers are expanded by expand_func_d, which requires the leading part of the
          evolved base to be a scalar / number-operator expression;
        - functions are expanded by expand_func_d around their evolved arguments.
        """
        p = self.p
        lo = lambda e: p.pe.leading_order(e)[1]
        if x.is_Add:
            return p.d_add(*(self.tail(a, order) for a in x.args))
        if x.is_Mul:
            plain = [f for f in x.args if not self.has_nonscalar(f)]
            factors = ([Mul(*plain)] if plain else []) + [f for f in x.args if self.has_nonscalar(f)]
            los = [lo(f) for f in factors]
            w = None
            for j, f in enumerate(factors):
                # each factor is needed up to the order left by the other factors
                wf = self.tail(f, order - Add(*los[:j], *los[j + 1:]))
                w = wf if w is None else (w @ wf).expand(order - Add(*los[j + 1:]))
            return w
        if x.is_Pow:
            b, e = x.args
            if any(self.is_mode_atom(a) for a in e.atoms(*_MODE_ATOMS)):
                raise ValueError(f'Cascaded evolution does not support exponents containing '
                                 f'the wave-mixing modes: {x}')
            lb = lo(b)
            if e.is_Integer and e > 0:
                wb = self.tail(b, order - (e - 1) * lb)
                w = wb
                for k in range(2, int(e) + 1):
                    w = (w @ wb).expand(order - (e - k) * lb)
                return w
            wb = self.tail(b, order - lo(x) + lb)
            lead, _ = wb.leading()
            if lead is None or lead.as_scalar is None:
                raise ValueError(
                    f'Cascaded evolution of {x}: the leading part of the evolved base is not a '
                    f'scalar or number-operator expression, so the power cannot be expanded')
            return p.expand_func_d(lambda y: y ** e, order, wb)
        if isinstance(x, Function):
            lx = lo(x)
            args = [self.tail(a, order - lx + lo(a)) for a in x.args]
            try:
                return p.expand_func_d(lambda *ys: x.func(*ys), order, *args)
            except ValueError as err:
                raise ValueError(f'Cascaded evolution of {x}: {err}')
        raise ValueError(f'Cascaded evolution does not support {x}')

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


def cascade_evolve(ctx: WMEvalContext, order: int, phases=(S.Zero, S.Zero, S.Zero)) -> WMWrapper:
    """
    Evolves the parsed expression of ctx for a further time t_c (the Dummy `tc`), with the
    real phases (phi_0, phi_1, phi_2) of the input operators (a_k -> exp(i phi_k) a_k(t_c)),
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
    ev = _Evolver(s, _required_sol_order(s, pieces, order), rel, phases, ctx.degenerate)
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


_PHASE_CONSTANTS = {'pi': pi, 'E': E}


def resolve_cascade_phase(text: str, ctx: WMEvalContext, k: int) -> Expr:
    """
    Returns the phase phi_k of the input operator a_k in the cascaded system: a real numeric
    expression (e.g. 0, -1.5, pi/2), a user-defined real commutative symbol, or a new real
    symbol (empty: 0). Names of preset objects, built-in parameters, non-real symbols and
    defined expressions / operators are rejected.
    """
    text = (text or '').strip()
    if not text:
        return S.Zero
    label = f'Phase phi_{k}'
    if is_valid_variable_name(text)[0] and text not in _PHASE_CONSTANTS:
        for sym_def in ctx.symbols:
            if (sym_def.get('name') or '').strip() == text:
                if sym_def.get('builtin_key') or text in ('nlc', 'pm'):
                    raise ValueError(f"{label} '{text}' conflicts with a built-in parameter.")
                sym = Symbol(text, **symbol_assumptions(sym_def))
                if not sym.is_real:
                    raise ValueError(f"{label} '{text}' must be a real symbol (set it as real, "
                                     f"integer, nonnegative or positive).")
                return sym
        if text in ctx.local_dict:
            raise ValueError(f"{label} '{text}' conflicts with a preset object or a defined "
                             f"expression / operator.")
        return Symbol(text, real=True)
    try:
        val = parse_expr(text, local_dict=dict(_PHASE_CONSTANTS), global_dict={
            'Integer': Integer, 'Float': Float, 'Rational': Rational, 'Symbol': Symbol})
    except Exception:
        raise ValueError(f"{label} '{text}' is neither a number nor a symbol name.")
    if not isinstance(val, Expr) or val.free_symbols or val.is_real is not True:
        raise ValueError(f"{label} '{text}' must be a real number or a real symbol.")
    return val


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
    cascade_phases: Optional[List[str]] = None,
    mismatch: bool = False,
    single_mode: bool = False,
    coherent_drive: bool = False,
    vacuum_signal: bool = False,
    vacuum_other_weak: bool = False,
    process: Optional[str] = None,
    degenerate: bool = False,
) -> Tuple[str, str, str, Union[int, float], int, Union[int, float]]:
    """
    Evaluates an operator combination evolved for a further time `cascade_symbol`
    (cascaded system), with the input-operator phases `cascade_phases` (phi_0, phi_1, phi_2;
    real numbers or symbols, default 0), optionally under input-state assertions.
    """
    ctx = _prepare_wm_evaluation(
        text, dc, order, symbols, operators, mm_operators, named_expressions, mismatch, single_mode,
        process=process, degenerate=degenerate,
    )
    t_sym = resolve_cascade_symbol(cascade_symbol, ctx)
    phases = tuple(resolve_cascade_phase(ph, ctx, k)
                   for k, ph in enumerate((list(cascade_phases or []) + ['', '', ''])[:3]))
    val = cascade_evolve(ctx, order, phases).apply(lambda e: e.subs(tc, t_sym), on_npf=True)
    val_t, val_g = apply_assertions(
        val, ctx.solver, order, process=ctx.proc, mismatch=ctx.mismatch, single_mode=single_mode,
        degenerate=ctx.degenerate, coherent_drive=coherent_drive, vacuum_signal=vacuum_signal, vacuum_other_weak=vacuum_other_weak,
        times=(t_sym,), alpha=coherent_alpha(ctx),
    )
    return finalize_wm_result(ctx, val_t, val_g, order, use_gain)
