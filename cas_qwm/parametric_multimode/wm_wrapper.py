import copy
from collections import defaultdict
from itertools import product
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, Dummy
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import diff, _mexpand
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from ..pattern_perturb import PF, PF0, PF1, pf
from ..op_pattern import BosonNum, BosonPat, FermionPat
from ..op_pattern.boson_pattern import fac
from ..pattern_perturb.pattern_form import pf_add, _vpk, bn_simp
from .multimode import MultimodeNum, MultimodeNum2, MultimodeCOP
from .wm_processor import (WMProcessor, vac_pf, _dr_prep, exceeds,
                           _t_mul, _tdz, t_, _g, nlc, nm)

__all__ = ['WMWrapper', 'EXACT']

EXACT = S.Infinity  # order attribute of an exact (untruncated) expression: accurate to all orders

def _is_exact(order) -> bool:
    return order is None or order == S.Infinity

def _omin(*orders):
    """Lowest of the given accuracy orders (None counts as exact, i.e. +oo)."""
    m = EXACT
    for o in orders:
        if o is not None and exceeds(m, o):
            m = sympify(o)
    return m

def _lower(a, b):
    """The lower of two (relative) orders (+oo for an exact factor)."""
    return b if exceeds(a, b) else a

def _sum_dl(dl: dict) -> dict:
    return {k: vs for k, ps in dl.items() if (vs := pf_add(*ps))}

_lx = Dummy('x')  # ladder variable

def _ladder(k: tuple, down: bool, start: int, sg: int):
    """
    Scans the multimode exponent tuple k block by block from the right, starting at
    level `start` (`down`: index of the lowering COP). Returns (final level, P0, P1),
    or None if the level goes below 0, with P0 / P1 the ladder products of
    the j=0 / j=1 lowest-weight families.
    """
    m, f0, f1 = start, [], []
    if k != (None,):
        i, *exps = k
        for j in range(len(exps) - 1, -1, -1):
            e = int(exps[j])
            if bool(i) ^ bool(j & 1) == down:
                if (m := m - e) < 0:
                    return None
            else:
                for l in range(m, m + e):
                    f0.append((l + 1) * (_lx + sg * l))
                    f1.append(l * (_lx + sg * (l + 1)))
                m += e
    return m, Mul(*f0), Mul(*f1)

_pw = lambda b, e: PF({((b**e,), ()): S.One}) if e else PF1  # PF of b**e

def _as_processor(dc) -> WMProcessor:
    """Returns dc itself if it is a WMProcessor, else the TWM processor for the given dc flag."""
    if isinstance(dc, WMProcessor):
        return dc
    from .wm_solver import twm_processors  # deferred: wm_solver imports this module
    return twm_processors[bool(dc)]

def _mmn_only(expr, weak_names) -> bool:
    """True if expr contains no unpaired weak-field single-mode operator or multimode COP."""
    if isinstance(expr, BosonPat):
        return expr.is_hermitian or not expr.has(*weak_names)
    if isinstance(expr, (FermionPat, MultimodeNum, MultimodeNum2)):
        return True
    if isinstance(expr, MultimodeCOP):
        return False
    if hasattr(expr, 'args'):
        return all(_mmn_only(arg, weak_names) for arg in expr.args)
    return True

class WMWrapper:
    """
    Algebraic wrapper for quantum multimode wave-mixing expressions.

    Stores:
    - processor: PMProcessor instance (for dc=True or dc=False)
    - dict_repr: dict structure ({tuple: PF} when b_left is None,
                 or {BosonOp: {tuple: PF}} when b_left is True/False)
    - b_left: True (weak-field BosonOp left-multiplied),
              False (weak-field BosonOp right-multiplied),
              or None (multimode/strong-field/scalar dict without unpaired weak BosonOp)
    - order: absolute perturbative order up to which the expression is accurate
             (S.Infinity, the default, for an exact expression). A product of
             factors with leading orders lo_1, lo_2 is accurate to
             lo_1 + lo_2 + min(order_1 - lo_1, order_2 - lo_2) and truncated there.
    """

    def __init__(self, dc, dict_repr=None, b_left=None, order=EXACT, truncate=True):
        """
        A finite `order` truncates the expression at that order, unless `truncate` is False
        (for content known to carry no terms beyond it).
        """
        self.processor = _as_processor(dc)
        # absolute perturbative order up to which the expression is accurate (EXACT: +oo)
        self.order = EXACT if _is_exact(order) else sympify(order)
        self._lo = None
        if isinstance(dict_repr, PF):
            dict_repr = {(None,): dict_repr} if dict_repr else dict()
        elif dict_repr is None:
            dict_repr = dict()
        dr = _dr_prep(dict_repr)
        if dr is None:
            new_dr = dict()
            for k, v in dict_repr.items():
                if not (isinstance(k, BosonOp) and k.name in self.processor.a[0].names):
                    raise ValueError(f'{k} is not a legal weak-field key for this instance')
                v = _dr_prep(v)
                if v is None:
                    raise ValueError("Invalid dict_repr structure")
                elif v:
                    new_dr[k] = v
            self.dict_repr = new_dr
            if new_dr:
                parity = self.processor.f_parity(tuple(new_dr))
                self.b_left = (self.dc ^ parity) if b_left is None else bool(b_left)
            else:
                self.b_left = None
        else:
            self.dict_repr = dr
            self.b_left = None
        if truncate and not self.is_exact:
            self._truncate()

    def _truncate(self):
        """Drops the terms beyond self.order in place (no-op for a non-numeric order)."""
        if not (self.dict_repr and self.order.is_number):
            return
        expand_pf, n = self.processor.pe.expand_pf, self.order
        trunc = lambda d: {k: tv for k, v in d.items() if (tv := expand_pf(v, n))}
        if self.b_left is None:
            self.dict_repr = trunc(self.dict_repr)
        else:
            self.dict_repr = {b: td for b, d in self.dict_repr.items() if (td := trunc(d))}
            if not self.dict_repr:
                self.b_left = None
        self._lo = None

    @property
    def dc(self) -> bool:
        return self.processor.dc

    @classmethod
    def from_val(cls, val, dc=True) -> 'WMWrapper':
        """
        Converts an Expr, PF, dict, WMWrapper, or string/number into a WMWrapper.
        """
        proc = _as_processor(dc)
        if isinstance(val, cls):
            return val if val.dc == proc.dc else cls(proc, val.dict_repr, val.b_left, val.order, False)
        if isinstance(val, (PF, dict)):
            return cls(proc, val, None)
        if isinstance(val, str) and val in proc.init_ops:
            return proc.init_ops[val]

        expr = sympify(val)
        if isinstance(expr, MultimodeCOP):
            return cls(proc, {(bool(expr.annih1), 1): PF1}, None)

        weak_names = proc.a[0].names
        if isinstance(expr, BosonOp) and expr.name in weak_names:
            return cls(proc, {expr: {(None,): PF1}})

        if _mmn_only(expr, weak_names):
            return cls(proc, {(None,): pf(expr, as_dict=None,
                                          ignore=(MultimodeNum, MultimodeNum2, MultimodeCOP))}, None)

        if isinstance(expr, BosonPat) and expr.name in weak_names:
            pp, np = expr.num_repr()
            if isinstance(pp, BosonPat):
                if pp.exps[0] > 1:
                    raise ValueError('Solution parity unbalanced')
                b = BosonOp(*pp.args[:2])
                w = cls(proc, {b: {(None,): PF({_vpk: np})}}, b_left=True)
                return w if w.b_left == (proc.dc ^ w.parity) else w.flip()
            return cls(proc, {(None,): PF({_vpk: np})}, None)

        if expr.has(MultimodeCOP) or any(
            getattr(b, 'name', None) in weak_names
            for b in expr.atoms(BosonOp, BosonPat)
            if not isinstance(b, BosonNum)
        ):
            if expr.is_Add:
                return proc.d_add(*(cls.from_val(arg, proc) for arg in expr.args))
            if expr.is_Mul:
                res = cls.from_val(expr.args[0], proc)
                for arg in expr.args[1:]:
                    res = res * cls.from_val(arg, proc)
                return res
            if expr.is_Pow:
                return cls.from_val(expr.base, proc) ** expr.exp

        return cls(proc, {(None,): pf(expr, as_dict=None)}, None)

    def __bool__(self) -> bool:
        return bool(self.dict_repr)

    def __repr__(self) -> str:
        o = '' if _is_exact(self.order) else f', order={self.order}'
        return f"WMWrapper(dc={self.dc}, {self.dict_repr}, b_left={self.b_left}{o})"

    def with_order(self, order, truncate=True) -> 'WMWrapper':
        """
        A shallow copy of this wrapper with the accuracy order set to `order`; when the order
        is lowered, the copy is truncated there unless `truncate` is False (for content known
        to carry no terms beyond it).
        """
        w = copy.copy(self)
        w.order = EXACT if _is_exact(order) else sympify(order)
        if truncate and exceeds(self.order, w.order):
            w._truncate()
        return w

    @property
    def is_exact(self) -> bool:
        return _is_exact(self.order)

    @property
    def lo(self):
        """Leading perturbative order of the expression (S.Infinity if it vanishes)."""
        if self._lo is None:
            lo_pf = self.processor.pe.lo_order_pf
            pfs = (self.dict_repr.values() if self.b_left is None
                   else (v for d in self.dict_repr.values() for v in d.values()))
            m = S.Infinity
            for v in pfs:
                if exceeds(m, o := lo_pf(v)):
                    m = o
            self._lo = m
        return self._lo

    @property
    def rel_order(self):
        """Relative accuracy: order - leading order (S.Infinity if exact)."""
        return S.Infinity if self.is_exact else self.order - self.lo

    def _zero_order(self, other: 'WMWrapper'):
        """Accuracy order of the vanishing product of self and other (one of them is zero)."""
        cands = []
        for a, b in ((self, other), (other, self)):
            if not a and not a.is_exact:
                cands.append(a.order + (b.lo if b else S.Zero))
        return _omin(*cands)

    def _prod_plan(self, other: 'WMWrapper'):
        """
        (target order, or None if the product is exact; left factor; right factor) of the
        product of two non-zero wrappers. The target is lo_1 + lo_2 + min(rel_1, rel_2),
        rel_i = order_i - lo_i; the factor more accurate than needed is truncated beforehand.
        """
        if self.is_exact and other.is_exact:
            return None, self, other
        lo1, lo2 = self.lo, other.lo
        r1, r2 = self.rel_order, other.rel_order
        r = _lower(r1, r2)
        target = lo1 + lo2 + r
        if not (target.is_number and target.is_finite):
            return None, self, other
        f1 = self.expand(lo1 + r) if exceeds(r1, r) else self
        f2 = other.expand(lo2 + r) if exceeds(r2, r) else other
        return target, f1, f2

    def flip(self) -> 'WMWrapper':
        """
        Returns an algebraically equivalent WMWrapper with b_left inverted
        (True <-> False) using bd_swap, or self if b_left is None.
        """
        if self.b_left is None:
            return self
        p = self.processor
        target_left = not self.b_left
        swapped = [
            WMWrapper(p, res, target_left)
            for b, d in self.dict_repr.items()
            if (res := p.bd_swap(b, d, to_left=target_left))
        ]
        if not swapped:
            return WMWrapper(p, {}, None, self.order)
        return p.d_add(*swapped).with_order(self.order, False)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WMWrapper):
            return self.__eq__(self.from_val(other, self.processor))
        if isinstance(other, WMWrapper):
            if self.dc != other.dc:
                return False
            if self.b_left == other.b_left:
                return self.dict_repr == other.dict_repr
            if self.b_left is None or other.b_left is None:
                return False
            return self.flip().dict_repr == other.dict_repr

    @property
    def as_scalar(self):
        """Returns the scalar/number-operator Expr if this is a pure scalar PF wrapper, else None."""
        if self.b_left is not None or not set(self.dict_repr.keys()).issubset({(None,)}):
            return None
        p = self.dict_repr.get((None,), PF0)
        if p.npf == 0 and set(p.table.keys()).issubset({_vpk}):
            return p.table.get(_vpk, S.Zero)
        return None

    def leading(self):
        """
        (leading part, leading order) of a wrapper without single-mode weak-field keys:
        the terms of its lowest perturbative order. (None, oo) if it has single-mode keys
        or vanishes.
        """
        if self.b_left is not None or not self.dict_repr:
            return None, S.Infinity
        pe = self.processor.pe
        los = {k: pe.lo_pf(v) for k, v in self.dict_repr.items()}
        m = min(o for _, o in los.values())
        return WMWrapper(self.processor, {k: lp for k, (lp, o) in los.items() if o == m}, None), m

    @property
    def parity(self):
        """Returns the weak-field parity (0 or 1), or None if b_left is None."""
        if self.b_left is None or not self.dict_repr:
            return None
        return self.processor.f_parity(tuple(self.dict_repr))

    def apply(self, f, on_npf=False, self_update=False) -> 'WMWrapper':
        """
        Applies f to every scalar/number-operator tail of the leaf PFs (and npf if on_npf);
        falls back to applying f to every leaf PF, then to every {tuple: PF} dict
        when f raises TypeError / AttributeError. Returns a new WMWrapper.
        """
        def _apply_node(e):
            try:
                return {k: nv for k, v in e.items() if (nv := v.apply(f, on_npf))}
            except (TypeError, AttributeError):
                try:
                    return {k: nv for k, v in e.items() if (nv := f(v))}
                except (TypeError, AttributeError):
                    return f(e)

        res = (_apply_node(self.dict_repr) if self.b_left is None
               else {b: dv for b, d in self.dict_repr.items()
                     if (dv := _apply_node(d))})
        
        if self_update:
            self.dict_repr = res
            self._lo = None
            if not res:
                self.b_left = None
            return self
        return WMWrapper(self.processor, res, self.b_left if res else None, self.order, False)

    def expand(self, order, with_order=False):
        """
        Expands this WMWrapper to the given perturbative order.
        If with_order=True, returns a generator of (WMWrapper, order) pairs, one per
        perturbative term (orders may repeat). As in the plain expansion, the order is
        counted on the PF parts; the weak-field and multimode keys are of order 0.
        """
        pe = self.processor.pe
        if with_order:
            return self._expand_terms(order)
        return self.apply(lambda p: pe.expand_pf(p, order), on_npf=True).with_order(
            _omin(self.order, order), False)

    def _expand_terms(self, order):
        """Yields (WMWrapper, order) pairs of the perturbative terms up to the given order."""
        p = self.processor
        acc = _omin(self.order, order)
        if self.b_left is None:
            for k, v in self.dict_repr.items():
                for pt, o in p.pe.expand_pf(v, order, True):
                    yield WMWrapper(p, {k: pt}, None, acc, False), o
            return
        for b, d in self.dict_repr.items():
            for k, v in d.items():
                for pt, o in p.pe.expand_pf(v, order, True):
                    yield WMWrapper(p, {b: {k: pt}}, self.b_left, acc, False), o

    def eval_vacuum(self, vac_modes) -> 'WMWrapper':
        """
        Evaluates the partial or full expectation value of this WMWrapper
        under vacuum input assertions for the specified mode(s).
        The npf parts are kept untouched.
        """
        p = self.processor
        vac_sets = set(), set()
        weak_names = p.a[0].names
        for m in (vac_modes or ()):
            m = sympify(m)
            if not m.is_Symbol and hasattr(m, 'name'):
                m = m.name
            vac_sets[m in weak_names].add(m)
        if not vac_sets[0] and not vac_sets[1]:
            return self
        return self._eval_vacuum(p, vac_sets).with_order(self.order, False)

    def _eval_vacuum(self, p, vac_sets) -> 'WMWrapper':
        weak_names = p.a[0].names

        _strong = lambda d: ({k: vp for k, v in d.items() if (vp := vac_pf(v, vac_sets[0]))}
                             if vac_sets[0] else d)
        if not vac_sets[1]:
            return self.apply(_strong)
        if self.b_left is not None and len(vac_sets[1]) == 2:
            return WMWrapper(p, {}, None)  # odd parity

        vb = weak_names[1] if weak_names[1] in vac_sets[1] else weak_names[0]  # in vacuum
        va = weak_names[weak_names[0] == vb]
        full = va in vac_sets[1]
        down = BosonOp(vb) in p.a[1].mode_expr.args  # index of the COP lowering vb
        a_bar = next(b for b in p.a[down].mode_expr.args if b.name == va)
        sg = 1 if p.dc else -1
        x_op = MultimodeNum(va, int(p.dc))

        def _collapse(expr, occ=S.Zero):
            """Collapses vacuum-field numbers; mode vb_1 holds `occ` quanta."""
            occ = sympify(occ)  # keep replace() results as sympy objects
            oc = lambda n: occ if n == vb else S.Zero
            _mm = lambda n, d: d * nm + oc(n) if n in vac_sets[1] else MultimodeNum(n, d)

            def _mm2(m1, m2, q):
                in1, in2 = m1 in vac_sets[1], m2 in vac_sets[1]
                if not (in1 or in2):
                    return MultimodeNum2(m1, m2, q)
                if q:  # Σ(N_2 - N_1)
                    return ((oc(m2) if in2 else MultimodeNum(m2))
                            - (oc(m1) if in1 else MultimodeNum(m1)))
                if in1 and in2:
                    return nm + oc(m1) + oc(m2)
                return MultimodeNum(m2 if in1 else m1, 1) + oc(m1 if in1 else m2)

            return expr if expr.is_number else (
                expr.replace(lambda e: isinstance(e, BosonNum) and e.name in vac_sets[1],
                             lambda e: oc(e.name))
                .replace(MultimodeNum, _mm).replace(MultimodeNum2, _mm2))

        def _key_terms(k, q, p_left):
            """⟨0_b| [b_1] T_k b_1†^q |0_b⟩ as a tuple of a-field PFs."""
            scan = _ladder(k, down, q, sg)
            if scan is None or scan[0] != p_left:  # final level 0, or 1 under b_1
                return ()
            _, p0, p1 = scan
            pairs = ([(((p0 - p1) / _lx).expand(), _pw(Dagger(a_bar), 1) * _pw(a_bar, q)),
                      (p1, _pw(a_bar, q - 1) if q else PF0)] if p_left
                     else [(p0, _pw(a_bar, q))])
            x = x_op - sg if p_left else x_op
            return tuple(PF({_vpk: fac(q) * c.subs(_lx, x)}) * ops for c, ops in pairs if c != 0)

        def _reduce(d: dict, occ: int = 0, p_left: bool = False):
            """
            Reduces {tuple: PF} (with b_1†^occ on the right and b_1 on the left if p_left)
            into an a-field PF; returns it together with the untouched npf parts.
            """
            out, npfs = [], {}
            for k, pf_v in _strong(d).items():
                if pf_v.npf != 0:
                    npfs[k] = PF(npf=pf_v.npf)
                for (bs, fs), tail in pf_v.table.items():
                    q, bs_rest = occ, []
                    for b in bs:
                        if b.name != vb:
                            bs_rest.append(b)
                        elif not b.left_base.is_annihilation:
                            q += b.exps[0]
                        elif b.exps[0] > occ:  # b_1^e b_1†^occ |0_b⟩ = 0
                            break
                        else:                  # e == occ == 1
                            q = 0
                    else:
                        if (t := _collapse(tail, occ)) != 0:
                            rest = PF({(tuple(bs_rest), fs): t})
                            out.extend(kp * rest for kp in _key_terms(k, q, p_left))
            red = pf_add(*out) if out else PF0
            return (vac_pf(red, [va]).apply(_collapse) if full else red).apply(_mexpand), npfs

        def _merged(d: dict) -> dict:
            red, npfs = _reduce(d)
            return {**npfs, (None,): pf_add(red, npfs.get((None,), PF0))}

        if self.b_left is None:
            return self.apply(_merged)

        def _extract(red: PF) -> 'WMWrapper':
            """Pulls the unpaired a-field operator out of a (None,)-keyed PF as a single-mode key."""
            grouped = defaultdict(list)
            for (bs, fs), t in red.table.items():
                j = next((j for j, b in enumerate(bs) if b.name == va), None)
                if j is None:
                    raise ValueError('Solution parity unbalanced')
                o, e = bs[j].left_base, bs[j].exps[0]
                grouped[o].append(PF({(bs[:j] + ((o**(e - 1),) if e > 1 else ()) + bs[j + 1:], fs): t}))
            w = WMWrapper(p, {o: {(None,): pf_add(*ps)} for o, ps in grouped.items()}, True)
            return w if w.b_left == self.b_left else w.flip().apply(_mexpand)

        ws = []
        for b, d in self.dict_repr.items():
            if b.name != vb:
                ws.append(WMWrapper(p, {b: _merged(d)}, self.b_left))
            elif bool(b.is_annihilation) == bool(self.b_left):  # else ⟨0_b|b_1† = b_1|0_b⟩ = 0
                red, npfs = _reduce(d, p_left=True) if self.b_left else _reduce(d, occ=1)
                ws += [_extract(red), WMWrapper(p, {b: npfs}, self.b_left)]
        return p.d_add(*ws)

    def eval_coherent(self, order: int = 0, alpha=None) -> 'WMWrapper':
        """
        Evaluates this WMWrapper under coherent state assertion for the strong driving field,
        with the coherent amplitude `alpha` (default: the symbol alpha_0 for dc, alpha_2 for fc).
        """
        p = self.processor
        if alpha is None:
            alpha = Symbol('alpha_0') if p.dc else Symbol('alpha_2')
        sub_dict = {p.nd.name: alpha}

        def _cohr_dict(d: dict) -> dict:
            return {
                k: cp for k, v in d.items()
                if (cp := p.pe.cohr_pf(v, sub_dict, order))
            }

        acc = _omin(self.order, order)
        if self.b_left is None:
            return WMWrapper(p, _cohr_dict(self.dict_repr), None, acc, False)
        res = {b: cd for b, d in self.dict_repr.items() if (cd := _cohr_dict(d))}
        return WMWrapper(p, res, self.b_left if res else None, acc, False)

    @property
    def expr(self) -> Expr:
        """Converts the wrapper representation into a SymPy Expr."""
        p = self.processor
        if self.b_left is None:
            return p._d_expr(self.dict_repr)
        if self.b_left:
            return Add(*(b * p._d_expr(d) for b, d in self.dict_repr.items()))
        return Add(*(p._d_expr(d) * b for b, d in self.dict_repr.items()))

    def to_single_mode(self, degenerate=False) -> 'WMWrapper':
        """
        Converts multimode operator representations into single-mode boson
        operators (mode_expr) with n_m = 1, returning a single-mode WMWrapper.

        degenerate=True (effective only for dc=True) converts into the degenerate
        down-conversion with signal = idler = a_1, governed by
        H = χ (a_0† a_1² + a_0 a_1†²) / 2. The map
            Σ(a_1 a_2) -> a_1²/2,  Σ(1 + N_1 + N_2) -> N_1 + 1/2,
            Σ(N_j + d) -> (N_1 + d)/2,  n_m -> 1/2,  a_2 -> a_1
        preserves every commutator between the single weak-field operator and the
        multimode / driving-field operators, so it carries the multimode Heisenberg
        solutions (and expressions linear in one weak-field operator, or built by `@`)
        onto the degenerate ones; a multimode sum Σ N_j maps to N_1/2 (half of the
        degenerate photon number). Single-mode products (`*`) of weak-field operators
        must be taken after this conversion, not before.
        """
        p = self.processor
        deg = bool(degenerate) and p.dc
        def _dict_sm(d: dict) -> PF:
            return pf_add(*(
                p._smk(k, deg) * p._sm_pf(v, deg)
                for k, v in d.items()
            ))

        if self.b_left is None:
            pf_res = _dict_sm(self.dict_repr)
        elif self.b_left:
            pf_res = pf_add(*(p._smk(b, deg) * _dict_sm(d) for b, d in self.dict_repr.items()))
        else:
            pf_res = pf_add(*(_dict_sm(d) * p._smk(b, deg) for b, d in self.dict_repr.items()))
        return WMWrapper(p, pf_res.apply(bn_simp), None, self.order, False)

    def adjoint(self) -> 'WMWrapper':
        """Takes the Hermitian adjoint of the operator representation."""
        p = self.processor

        def _adj_dict(d: dict) -> dict:
            ad = defaultdict(list)
            for k, v in d.items():
                i, *exps = k
                if exps:
                    i = bool(i ^ (len(exps) & 1))
                    exps.reverse()
                for k_new, vp in p.tp_swap((i, *exps), v.adjoint(), to_left=True):
                    ad[k_new].append(vp)
            return {k: vs for k, v in ad.items() if (vs := pf_add(*v))}

        if self.b_left is None:
            return WMWrapper(p, _adj_dict(self.dict_repr), None, self.order, False)
        res = {Dagger(b): ad for b, d in self.dict_repr.items() if (ad := _adj_dict(d))}
        return WMWrapper(p, res, (not self.b_left) if res else None, self.order, False)

    def __add__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        return self.processor.d_add(self, other)

    def __radd__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__add__(self)

    def __neg__(self) -> 'WMWrapper':
        return self.apply(lambda p: -p, on_npf=True)

    def __sub__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        return self.processor.d_add(self, -other)

    def __rsub__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__sub__(self)

    def __mul__(self, other) -> 'WMWrapper':
        """Single-mode product, truncated at the accuracy order of the product."""
        other = WMWrapper.from_val(other, self.processor)
        if not self or not other:
            return WMWrapper(self.processor, {}, None, self._zero_order(other))
        order, f1, f2 = self._prod_plan(other)
        return f1._mul(f2, order).with_order(order, False)  # truncated in _mul

    def _mul(self, other: 'WMWrapper', order=None) -> 'WMWrapper':
        """Single-mode product of non-zero wrappers, truncated at `order` (None: exact)."""
        p = self.processor
        if self.b_left is None or other.b_left is None:
            return self._matmul(other, order)
        if self.parity == other.parity:
            raise ValueError('Solution parity unbalanced')
        res_l = defaultdict(list)
        if other.b_left:
            e1 = (self.flip() if self.b_left else self).dict_repr
            for (b1, d1), (b2, d2) in product(e1.items(), other.dict_repr.items()):
                if b1 == Dagger(b2):
                    bn = BosonNum(b1.name) + int(b1.is_annihilation)
                    p.d_mul(res_l, d1, d2, bn, order=order)
                elif b1.compare(b2) == 1:
                    p.d_mul(res_l, d1, d2, b2.is_annihilation, True, order=order)
                else:
                    p.d_mul(res_l, d1, d2, b1.is_annihilation, True, order=order)
            return WMWrapper(p, _sum_dl(res_l), None)
        else:
            if self.b_left:
                for (b1, d1), (b2, d2) in product(self.dict_repr.items(),
                                                other.dict_repr.items()):
                    mid_dl = defaultdict(list)
                    p.d_mul(mid_dl, d1, d2, order=order)
                    for b, d in p.bd_swap(b1, _sum_dl(mid_dl), False).items():
                        for k, v in d.items():
                            res_l[k].append(v*(b*b2))
            else:
                for (b1, d1), (b2, d2) in product(self.dict_repr.items(),
                                                other.dict_repr.items()):
                    for k2, v2 in d2.items():
                        i, *exps = k2
                        for s, b, nes in p.bt_swap(b1, i, exps):
                            for k1, v1 in d1.items():
                                res_l[_t_mul(k1, _tdz(i, nes))].append(v1*s*b*v2*b2)
        res = WMWrapper(p, _sum_dl(res_l), None)
        return res if order is None else res.expand(order)

    def __rmul__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__mul__(self)

    def __matmul__(self, other) -> 'WMWrapper':
        """Multimode product, truncated at the accuracy order of the product."""
        other = WMWrapper.from_val(other, self.processor)
        if not self or not other:
            return WMWrapper(self.processor, {}, None, self._zero_order(other))
        order, f1, f2 = self._prod_plan(other)
        return f1._matmul(f2, order).with_order(order, False)  # truncated in _matmul

    def _matmul(self, other: 'WMWrapper', order=None) -> 'WMWrapper':
        """Multimode product of non-zero wrappers, truncated at `order` (None: exact)."""
        e1, e2 = self.dict_repr, other.dict_repr
        p = self.processor

        is_f1 = self.b_left is not None
        is_f2 = other.b_left is not None
        result = dict()

        if is_f1 and is_f2:
            if self.parity == other.parity:
                raise ValueError('Solution parity unbalanced')
            res_l = defaultdict(list)
            e1 = (self.flip() if self.b_left else self).dict_repr
            e2 = (other if other.b_left else other.flip()).dict_repr
            for (b1, d1), (b2, d2) in product(e1.items(), e2.items()):
                if b1 == Dagger(b2):
                    mn = MultimodeNum(b1.name, int(b1.is_annihilation))
                    p.d_mul(res_l, d1, d2, mn, order=order)
                elif b1.compare(b2) == 1:
                    p.d_mul(res_l, d1, d2, b2.is_annihilation, order=order)
                else:
                    p.d_mul(res_l, d1, d2, b1.is_annihilation, order=order)
            return WMWrapper(p, _sum_dl(res_l), None)

        if is_f1:
            if self.b_left:
                for b, d in e1.items():
                    res_dl = defaultdict(list)
                    p.d_mul(res_dl, d, e2, order=order)
                    if (res_b := _sum_dl(res_dl)):
                        result[b] = res_b
                return WMWrapper(p, result, True)
            res_dl = defaultdict(lambda: defaultdict(list))
            for b, d1 in e1.items():
                for nb, p2_prime in p.bd_swap(b, e2, to_left=False).items():
                    p.d_mul(res_dl[nb], d1, p2_prime, order=order)
            result = {nb: rb for nb, dl in res_dl.items() if (rb := _sum_dl(dl))}
            return WMWrapper(p, result, False)

        if is_f2:
            if not other.b_left:
                for b, d in e2.items():
                    res_dl = defaultdict(list)
                    p.d_mul(res_dl, e1, d, order=order)
                    if (res_b := _sum_dl(res_dl)):
                        result[b] = res_b
                return WMWrapper(p, result, False)
            res_dl = defaultdict(lambda: defaultdict(list))
            for b, d2 in e2.items():
                for nb, p1_prime in p.bd_swap(b, e1, to_left=True).items():
                    p.d_mul(res_dl[nb], p1_prime, d2, order=order)
            result = {nb: rb for nb, dl in res_dl.items() if (rb := _sum_dl(dl))}
            return WMWrapper(p, result, True)

        res_dl = defaultdict(list)
        p.d_mul(res_dl, e1, e2, order=order)
        return WMWrapper(p, _sum_dl(res_dl), None)

    def __rmatmul__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__matmul__(self)

    def __truediv__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        if other.as_scalar is not None:
            return self.__matmul__(other ** S.NegativeOne)
        if other.b_left is None:
            # expanded by expand_func_d around the scalar leading part of the denominator
            return self.__matmul__(other ** S.NegativeOne)
        raise ValueError("Division by single-mode weak-field operators is not supported.")

    def __rtruediv__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__truediv__(self)

    def __pow__(self, power) -> 'WMWrapper':
        p = self.processor
        if isinstance(power, WMWrapper):
            sc = power.as_scalar
            if sc is None:
                raise ValueError("Exponent must be a scalar or number-operator expression.")
            power = sc
        p_sym = sympify(power)

        base_scalar = self.as_scalar
        if base_scalar is not None:
            return self._pow_trunc(
                WMWrapper(p, {(None,): PF({_vpk: base_scalar ** p_sym})}, None), p_sym)

        if p_sym.is_Integer and p_sym >= 0:
            n_int = int(p_sym)
            if n_int == 0:
                return WMWrapper(p, {(None,): PF1}, None)
            if n_int == 1:
                return self
            if self.b_left is not None:
                raise ValueError("Solution parity unbalanced")
            if set(self.dict_repr.keys()) == {(None,)}:
                v = self.dict_repr[(None,)]
                try:
                    res = WMWrapper(p, {(None,): v ** n_int}, None)
                except Exception:
                    res = WMWrapper(p, {(None,): PF(npf=Pow(v.expr(), n_int))}, None)
                return self._pow_trunc(res, p_sym)
            res = None
            base = self
            exp_n = n_int
            while exp_n > 0:
                if exp_n & 1:
                    res = base if res is None else (res @ base)
                exp_n >>= 1
                if exp_n > 0:
                    base = base @ base
            return res

        if self.b_left is None:
            # other powers are expanded around the leading part of the base, which must be
            # a scalar / number-operator expression
            lead, _ = self.leading()
            if lead is None or lead.as_scalar is None:
                raise ValueError(
                    f"Power {p_sym} requires the leading part of the base to be a scalar or "
                    f"number-operator expression.")
            order = getattr(p, '_evolved_order', 1)
            return p.expand_func_d(lambda x: x ** p_sym, order, self)
        raise ValueError("Exponentiation of single-mode weak-field operators only supports "
                         "non-negative integers.")

    def _pow_trunc(self, res: 'WMWrapper', p_sym) -> 'WMWrapper':
        """
        Accuracy of the power res = self ** p_sym computed in closed form: the relative
        accuracy is kept (leading order p_sym * lo), and res is truncated there when the base
        is inexact.
        """
        if self.is_exact or not res:
            return res
        acc = p_sym * self.lo + self.rel_order
        if not (acc.is_number and acc.is_finite):
            return res
        try:
            return res.expand(acc)
        except Exception:
            return res.with_order(acc, False)

    def diff(self, *symbols, gc=None, **kwargs) -> 'WMWrapper':
        if not symbols:
            raise ValueError("diff requires at least one variable.")
        p = self.processor
        var = symbols[0]
        rest_args = symbols[1:]

        is_wrt_g = False
        is_wrt_t = False

        if isinstance(var, WMWrapper):
            if getattr(var, '_is_g', False):
                is_wrt_g = True
                if gc is None:
                    gc = getattr(var, '_gc', None)
            elif getattr(var, '_is_t', False):
                is_wrt_t = True
            else:
                sc = var.as_scalar
                if sc is None:
                    raise ValueError(f"Cannot differentiate with respect to operator: {var}")
                var = sc
        elif isinstance(var, str):
            if var == 't':
                is_wrt_t = True
            elif var in ('g', '_g', r'\hat{g}'):
                is_wrt_g = True
            elif var == 'nlc':
                var = nlc
            else:
                var = Symbol(var)
        elif var == t_ or str(var) == '_t':
            is_wrt_t = True
        elif str(var) in ('g', '_g', r'\hat{g}'):
            is_wrt_g = True

        if is_wrt_g and gc is not None:
            order_n = 1
            if rest_args and isinstance(rest_args[0], int):
                order_n = rest_args[0]
                rest_args = rest_args[1:]
            gc_inv_n = gc ** (-order_n)
            return self.apply(
                lambda v: p.n_simp(diff(v, t_, order_n, *rest_args, **kwargs) * gc_inv_n),
                on_npf=True)
        if is_wrt_t:
            var = t_
        return self.apply(lambda v: diff(v, var, *rest_args, **kwargs), on_npf=True)

    Derivative = diff
