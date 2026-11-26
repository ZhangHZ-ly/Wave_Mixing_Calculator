import copy
from collections import defaultdict
from functools import lru_cache
from itertools import product
from math import comb, factorial
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol, Dummy
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.function import diff, _mexpand
from sympy.core.numbers import Integer, I
from sympy.functions.elementary.complexes import Abs, conjugate
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polytools import Poly
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from ..pattern_perturb import PF, PF0, PF1, pf
from ..op_pattern import BosonNum, BosonPat, FermionPat
from ..op_pattern.boson_pattern import fac
from ..pattern_perturb.pattern_form import pf_add, _vpk, bn_simp
from .multimode import MultimodeNum, MultimodeNum2, MultimodeCOP
from ..wm_variables import alpha_0, alpha_2, betas, t_, nlc, nm, pm
from .wm_processor import WMProcessor, vac_pf, _merge_dicts, _t_mul, _tdz

__all__ = ['WMWrapper']

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

# ---- coherent seeding of one mode pair (see WMWrapper._eval_seeded) ----
# A single-mode operator word is a tuple of (kind, e): kind 0 = b^e, 1 = b†^e, 2 = N_b^e.

def _pat_word(b) -> list:
    """Word of a single-mode boson pattern (BosonOp, BosonNum or BosonPat), left to right."""
    if isinstance(b, BosonNum):
        return [(1, 1), (0, 1)]
    if isinstance(b, BosonOp):
        return [(0 if b.is_annihilation else 1, 1)]
    ann, word = bool(b.args[1]), []
    for e in b.exps:
        word.append((0 if ann else 1, int(e)))
        ann = not ann
    return word

def _compact(word) -> tuple:
    """Merges adjacent factors of the same kind and drops the trivial ones."""
    out = []
    for kind, e in word:
        if not e:
            continue
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + e)
        else:
            out.append((kind, e))
    return tuple(out)

def _beta_mono(beta, m: int, n: int):
    """Coherent average conj(β)^m β^n, with the pairs written as |β|^2."""
    k = min(m, n)
    rest = beta ** (n - k) if n > m else conjugate(beta) ** (m - k)
    return Abs(beta) ** (2 * k) * rest

@lru_cache(maxsize=4096)
def _seed_avg(word: tuple, beta):
    """
    Coherent-state average <β| w |β> of a single-mode word, computed exactly by normal
    ordering (b^n b†^e = Σ_k C(n,k) C(e,k) k! b†^(e-k) b^(n-k)); β = 0 gives the vacuum.
    """
    no = {(0, 0): 1}  # normal-ordered form {(m, n): c} of Σ c b†^m b^n

    def times_a(d, e):
        return {(m, n + e): c for (m, n), c in d.items()}

    def times_c(d, e):
        out = defaultdict(int)
        for (m, n), c in d.items():
            for k in range(min(n, e) + 1):
                out[(m + e - k, n - k)] += c * comb(n, k) * comb(e, k) * factorial(k)
        return out

    for kind, e in word:
        if kind == 0:
            no = times_a(no, e)
        elif kind == 1:
            no = times_c(no, e)
        else:
            for _ in range(e):
                no = times_a(times_c(no, 1), 1)
    if beta == 0:
        return Integer(no.get((0, 0), 0))
    return Add(*(c * _beta_mono(beta, m, n) for (m, n), c in no.items() if c))

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

    def __init__(self, dc, dict_repr=None, b_left=None, order=S.Infinity, truncate=True):
        """
        A finite `order` truncates the expression at that order, unless `truncate` is False
        (for content known to carry no terms beyond it).
        """
        self.processor = _as_processor(dc)
        # absolute perturbative order up to which the expression is accurate (+oo if exact)
        self.order = sympify(order)
        self._lo = None
        self._hash = None
        if isinstance(dict_repr, PF):
            dict_repr = {(None,): dict_repr} if dict_repr else dict()
        elif dict_repr is None:
            dict_repr = dict()
        # truncation order of the content (+oo: kept as it is)
        trunc = self.order if truncate else S.Infinity
        dr = self.processor._dr_prep(dict_repr, trunc)
        if dr is None:
            new_dr = dict()
            for k, v in dict_repr.items():
                if not (isinstance(k, BosonOp) and k.name in self.processor.a[0].names):
                    raise ValueError(f'{k} is not a legal weak-field key for this instance')
                v = self.processor._dr_prep(v, trunc)
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
        o = '' if self.is_exact else f', order={self.order}'
        return f"WMWrapper(dc={self.dc}, {self.dict_repr}, b_left={self.b_left}{o})"

    def with_order(self, order, truncate=True) -> 'WMWrapper':
        """
        A shallow copy of this wrapper with the accuracy order set to `order`; when the order
        is lowered, the copy is truncated there unless `truncate` is False (for content known
        to carry no terms beyond it).
        """
        order = sympify(order)
        if truncate and order < self.order:
            return WMWrapper(self.processor, self.dict_repr, self.b_left, order)
        w = copy.copy(self)
        w.order = order
        w._hash = None
        return w

    @property
    def is_exact(self) -> bool:
        return self.order is S.Infinity

    @property
    def lo(self):
        """Leading perturbative order of the expression (S.Infinity if it vanishes)."""
        if self._lo is None:
            lo_pf = self.processor.pe.lo_order_pf
            pfs = (self.dict_repr.values() if self.b_left is None
                   else (v for d in self.dict_repr.values() for v in d.values()))
            self._lo = min((lo_pf(v) for v in pfs), default=S.Infinity)
        return self._lo

    @property
    def rel_order(self):
        """Relative accuracy: order - leading order (S.Infinity if exact)."""
        return self.order - self.lo

    def _zero_order(self, other: 'WMWrapper'):
        """Accuracy order of the vanishing product of self and other (one of them is zero)."""
        return min((a.order + (b.lo if b else S.Zero)
                    for a, b in ((self, other), (other, self)) if not a),
                   default=S.Infinity)

    def _prod_plan(self, other: 'WMWrapper'):
        """
        (target order, S.Infinity if the product is exact; left factor; right factor) of the
        product of two non-zero wrappers. The target is lo_1 + lo_2 + min(rel_1, rel_2),
        rel_i = order_i - lo_i; the factor more accurate than needed is truncated beforehand.
        """
        if self.is_exact and other.is_exact:
            return S.Infinity, self, other
        lo1, lo2 = self.lo, other.lo
        r1, r2 = self.rel_order, other.rel_order
        r = min(r1, r2)
        target = lo1 + lo2 + r
        f1 = self.expand(lo1 + r) if r1 > r else self
        f2 = other.expand(lo2 + r) if r2 > r else other
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

    def __hash__(self):
        if self._hash is None:
            dr = self.dict_repr
            items = (frozenset(dr.items()) if self.b_left is None
                     else frozenset((b, frozenset(d.items())) for b, d in dr.items()))
            self._hash = hash((self.processor, items, self.b_left, self.order))
        return self._hash

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

    def apply(self, f, on_npf=False) -> 'WMWrapper':
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
            min(self.order, order), False)

    def _expand_terms(self, order):
        """Yields (WMWrapper, order) pairs of the perturbative terms up to the given order."""
        p = self.processor
        acc = min(self.order, order)
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
            alpha = alpha_0 if p.dc else alpha_2
        sub_dict = {p.nd.name: alpha}

        def _cohr_dict(d: dict) -> dict:
            return {
                k: cp for k, v in d.items()
                if (cp := p.pe.cohr_pf(v, sub_dict, order))
            }

        acc = min(self.order, order)
        if self.b_left is None:
            return WMWrapper(p, _cohr_dict(self.dict_repr), None, acc, False)
        res = {b: cd for b, d in self.dict_repr.items() if (cd := _cohr_dict(d))}
        return WMWrapper(p, res, self.b_left if res else None, acc, False)

    def seeded_separate(self, mode: int = 1, vac_c: bool = True, observed: bool = False) -> 'WMWrapper':
        """
        Evaluates this WMWrapper with one separate mode b of the weak fields seeded in the
        coherent state |β_j> (the global symbols beta_j, j the index of the mode), coupled with
        its partner mode in another coherent state or in vacuum.

        :param mode: The subscript of the seeded mode. For parametric down-conversion and FWM,
        the signal is seeeded by default, and 0 indicates that the degenerate mode is seeded.

        :param vac_c: The coupled mode is in vacuum (True) or in its coherent state |β_k> (False);
        no effect for a degenerate seed.

        :param observed: Whethre the single-mode weak-field operators of the expression
        are the seeded modes themselves.

        After assert_mode_number(1) there is no multimode sum left: with `observed`, the weak modes
        themselves are then in these states.
        """
        p = self.processor
        from .wm_solver import fwm_processor  # deferred: wm_solver imports this module
        mode = 1 if p is fwm_processor else int(mode)
        if mode not in (0, 1):
            raise ValueError(f'The seeded mode must be 0 or 1, not {mode}')
        index = lambda n: int(str(n).rsplit('_', 1)[-1])
        names = p.a[0].names
        if p.dc and mode == 0:  # degenerate seed of amplitude beta_1
            return self._eval_seeded({names[0]: betas[1]}, observed, degenerate=True)
        # down-conversion / four-wave mixing: the signal (index 1); state transfer: index `mode`
        seeded = next(n for n in names if index(n) == mode)
        coupled = next(n for n in names if n != seeded)
        amps = {seeded: betas[index(seeded)], coupled: S.Zero if vac_c else betas[index(coupled)]}
        return self._eval_seeded(amps, observed)

    def _eval_seeded(self, amps: dict, observed: bool = False, degenerate: bool = False) -> 'WMWrapper':
        """
        Coherent average over the seeded mode pair b of the two weak fields (see
        seeded_separate): `amps` maps the weak-mode names (Symbols, as in p.a[0].names) to their amplitudes (0: vacuum);
        with `degenerate`, both weak fields share the single seeded mode, of the amplitude
        of the first name.
        """
        p = self.processor
        names = p.a[0].names
        # word slots of the seeded modes: one per weak field, or a single degenerate one
        slot = dict.fromkeys(names, 0) if degenerate else {n: j for j, n in enumerate(names)}
        n_slots = 1 if degenerate else 2
        beta = ((next(iter(amps.values()), S.Zero),) if degenerate
                else tuple(amps.get(n, S.Zero) for n in names))
        nbs = tuple(Dummy('N_b') for _ in range(n_slots))
        # annihilation flags of the two modes in each COP a[0], a[1]
        cop_flags = [(cop.annih1, cop.annih2) for cop in p.a]
        half = S.Half

        def _split(m):
            """Replacement of a multimode number by the rest + its b part (as N_b dummies)."""
            if isinstance(m, MultimodeNum):
                if m.name not in slot:
                    return m
                if degenerate:  # Σ(N_j + d) -> (N_b + d)/2
                    return m + half * (nbs[0] + m.displacement)
                return m + nbs[slot[m.name]] + m.displacement
            n1, n2 = m.names
            if degenerate:  # Σ(1 + N_1 + N_2) -> N_b + 1/2
                return m + nbs[0] + half
            if m.parity:  # Σ(N_2 - N_1)
                return m + nbs[slot[n2]] - nbs[slot[n1]]
            return m + 1 + nbs[slot[n1]] + nbs[slot[n2]]  # Σ(1 + N_1 + N_2)

        def _tail_terms(t):
            """{(k_1, ...): coefficient} of a tail as a polynomial in the N_b dummies."""
            subs = {m: _split(m) for m in t.atoms(MultimodeNum, MultimodeNum2)}
            if observed:
                for bn in t.atoms(BosonNum):
                    if bn.name in slot:
                        if degenerate:
                            raise ValueError('A degenerate seed can only be observed through '
                                             'single weak-field operators, not their products')
                        subs[bn] = nbs[slot[bn.name]]
            t = t.xreplace(subs)
            if not t.has(*nbs):
                return {(0,) * n_slots: t}
            # the number operators of the tail commute: commuting dummies for Poly
            cm = {o: Dummy() for o in t.atoms(MultimodeNum, MultimodeNum2, BosonNum)}
            back = {d: o for o, d in cm.items()}
            try:
                terms = Poly(t.xreplace(cm), *nbs).terms()
            except PolynomialError:
                raise ValueError('The seeded mode enters the expression non-polynomially')
            return {mon: c.xreplace(back) for mon, c in terms}

        def _cop_splits(k):
            """(coefficient, rest key, words of the b parts per slot) of a COP key."""
            if k == (None,):
                yield S.One, k, tuple([] for _ in range(n_slots))
                return
            i, *exps = k
            blocks = []
            for e in exps:
                blocks.append((int(i), int(e)))
                i = not i
            for js in product(*(range(e + 1) for _, e in blocks)):
                c = S.One
                words = tuple([] for _ in range(n_slots))
                for (ci, e), j in zip(blocks, js):
                    c *= comb(e, j)
                    if degenerate:  # Σ(a_1 a_2) -> b²/2
                        c *= half ** j
                    for m in (0, 1):
                        words[slot[names[m]]].append((0 if cop_flags[ci][m] else 1, j))
                yield c, _tdz(k[0], [e - j for (_, e), j in zip(blocks, js)]), words

        def _seed_dict(d: dict, key_words) -> dict:
            """
            Seeded average of a {COP key: PF} dict; key_words = (left, right) words per slot
            of the averaged single-mode key.
            """
            kl, kr = key_words
            tables, npfs = defaultdict(lambda: defaultdict(list)), defaultdict(list)
            for k, v in d.items():
                splits = list(_cop_splits(k))
                # (patterns, fermion patterns, tail, whether it is the non-pattern part)
                entries = [(bs, fs, t, False) for (bs, fs), t in v.table.items()]
                if v.npf != 0:
                    entries.append(((), (), v.npf, True))
                for bs, fs, t, is_npf in entries:
                    pat_words, rest_bs = tuple([] for _ in range(n_slots)), []
                    for b in bs:
                        if observed and getattr(b, 'name', None) in slot:
                            if degenerate:
                                raise ValueError('A degenerate seed can only be observed through '
                                                 'single weak-field operators, not their products')
                            pat_words[slot[b.name]].extend(_pat_word(b))
                        else:
                            rest_bs.append(b)
                    if is_npf and observed and any(getattr(b, 'name', None) in slot
                                                   for b in t.atoms(BosonOp, BosonPat)):
                        raise ValueError('Seeded single-mode operators in a non-pattern part')
                    poly = _tail_terms(t)
                    for c, rest_k, cop_words in splits:
                        for mon, coeff in poly.items():
                            avg = c
                            for m in range(n_slots):
                                word = _compact(kl[m] + cop_words[m] + pat_words[m]
                                                + [(2, mon[m])] + kr[m])
                                avg = avg * _seed_avg(word, beta[m])
                                if avg == 0:
                                    break
                            if avg == 0:
                                continue
                            if is_npf:
                                npfs[rest_k].append(avg * coeff)
                            else:
                                tables[rest_k][(tuple(rest_bs), fs)].append(avg * coeff)
            out = {}
            for rk in set(tables) | set(npfs):
                pv = PF({key: s for key, ts in tables[rk].items() if (s := Add(*ts)) != 0},
                        Add(*npfs[rk]))
                if pv:
                    out[rk] = pv
            return out

        empty = tuple([] for _ in range(n_slots))
        if self.b_left is None:
            return WMWrapper(p, _seed_dict(self.dict_repr, (empty, empty)), None, self.order, False)
        if not observed:
            res = {b: sd for b, d in self.dict_repr.items() if (sd := _seed_dict(d, (empty, empty)))}
            return WMWrapper(p, res, self.b_left if res else None, self.order, False)
        # the observed single-mode key is a seeded mode: it is averaged with the rest
        parts = []
        for b, d in self.dict_repr.items():
            kw = tuple([] for _ in range(n_slots))
            kw[slot[b.name]].append((0 if b.is_annihilation else 1, 1))
            parts.append(_seed_dict(d, (kw, empty) if self.b_left else (empty, kw)))
        return WMWrapper(p, _merge_dicts(parts), None, self.order, False)

    @property
    def expr(self) -> Expr:
        """Converts the wrapper representation into a SymPy Expr."""
        p = self.processor
        if self.b_left is None:
            return p._d_expr(self.dict_repr)
        if self.b_left:
            return Add(*(b * p._d_expr(d) for b, d in self.dict_repr.items()))
        return Add(*(p._d_expr(d) * b for b, d in self.dict_repr.items()))

    def assert_mode_number(self, n: int = 1, degenerate=False, mismatch=None) -> 'WMWrapper':
        """
        Takes the multimode sums over n explicit weak-mode pairs (n_m = n), returning a
        single-mode WMWrapper. For n = 1 the modes keep their names (a_1, a_2); otherwise
        they are a_1_k, a_2_k (k = 1, 2, ...), and the single-mode weak-field operators are
        those of k = 1. n = 0 leaves no mode in the sums (the single-mode operators are kept,
        e.g. for the modes seeded by seeded_separate).

        degenerate=True (dc only, n >= 1) makes one of the n modes degenerate (signal = idler
        = a_1, H = χ (a_0† a_1² + a_0 a_1†²) / 2), which is also the single-mode one:
            Σ(a_1 a_2) -> a_1²/2,  Σ(1 + N_1 + N_2) -> N_1 + 1/2,
            Σ(N_j + d) -> (N_1 + d)/2,  n_m -> 1/2,  a_2 -> a_1
        plus the sums over the other n - 1 pairs. These maps preserve every commutator the
        multimode solutions use, so they are exact for the solver outputs and `@` products;
        single-mode products (`*`) of weak-field operators must be taken after this conversion.

        mismatch: phase mismatches (Δk_1, Δk_2, ...) of the non-degenerate pairs k = 1, 2, ...,
        replacing the common Δk, which a degenerate mode keeps (as its operators keep their
        names). A term depending on Δk must involve a single mode: it contains one multimode
        sum (a COP, Σ(...) or n_m, taken as Σ_k c(Δk_k) X_k), or none (single-mode weak-field
        and drive operators, with the mismatch of the single-mode one); otherwise ValueError
        is raised. Products of terms
        of different modes must thus be taken after this conversion. With TWMSolver
        (mismatch=True) this restricts it to the solutions of orders up to 1 (get_sol(1):
        the drive up to its first correction, the weak fields at their leading order) and
        to expressions linear in their multimode sums; from order 2 on, the solutions
        couple the mismatches of different modes, which a common Δk cannot represent. The
        modes keep their own frames (rotated by exp(∓i Δk_k t / 2), see TWMSolver).
        """
        n = int(n)
        if n < 0:
            raise ValueError('The mode number must be a nonnegative integer')
        p = self.processor
        deg = bool(degenerate) and p.dc and n > 0
        if mismatch is not None:
            mismatch = tuple(sympify(x) for x in mismatch)
            if len(mismatch) != n - deg:
                raise ValueError(f'{n - deg} phase mismatches expected, got {len(mismatch)}')
            if deg:
                mismatch = (pm, *mismatch)

        def _conv(k, v, rep=False) -> PF:
            if mismatch is None:
                return p._mn_key(k, n, deg) * p._mn_pf(v, n, deg)
            return p._mn_split(k, v, n, deg, mismatch, rep)

        def _dict_mn(d: dict, rep=False) -> PF:
            return pf_add(*(_conv(k, v, rep) for k, v in d.items()))

        if self.b_left is None:
            pf_res = _dict_mn(self.dict_repr)
        elif self.b_left:
            pf_res = pf_add(*(p._mn_key(b, n, deg) * _dict_mn(d, True) for b, d in self.dict_repr.items()))
        else:
            pf_res = pf_add(*(_dict_mn(d, True) * p._mn_key(b, n, deg) for b, d in self.dict_repr.items()))
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

    def _mul(self, other: 'WMWrapper', order=S.Infinity) -> 'WMWrapper':
        """Single-mode product of non-zero wrappers, truncated at `order` (S.Infinity: exact)."""
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
        return res if order is S.Infinity else res.expand(order)

    def __rmul__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__mul__(self)

    def __matmul__(self, other) -> 'WMWrapper':
        """Multimode product, truncated at the accuracy order of the product."""
        other = WMWrapper.from_val(other, self.processor)
        if not self or not other:
            return WMWrapper(self.processor, {}, None, self._zero_order(other))
        order, f1, f2 = self._prod_plan(other)
        return f1._matmul(f2, order).with_order(order, False)  # truncated in _matmul

    def _matmul(self, other: 'WMWrapper', order=S.Infinity) -> 'WMWrapper':
        """Multimode product of non-zero wrappers, truncated at `order` (S.Infinity: exact)."""
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
            # truncation order set by the caller
            order = getattr(p, '_func_order', getattr(p, '_evolved_order', 1))
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
            elif var in ('g', r'\hat{g}'):
                is_wrt_g = True
            elif var == 'nlc':
                var = nlc
            else:
                var = Symbol(var)
        elif var == t_ or str(var) == '_t':
            is_wrt_t = True
        elif str(var) in ('g', r'\hat{g}'):
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

    def bch(self, a, order: int) -> 'WMWrapper':
        """
        Term (i t)^n / n! [h, [h, ... [h, a]]] (n = order nested commutators) of the BCH series
        of e^{i h t} a e^{-i h t}, h being this wrapper (a time-independent Hamiltonian).
        """
        order = int(order)
        if order < 0:
            raise ValueError('The BCH order must be a nonnegative integer')
        a = WMWrapper.from_val(a, self.processor)
        return _bch(self, a, a.order, order)

    def bch_sum(self, a, order: int) -> 'WMWrapper':
        """BCH series of e^{i h t} a e^{-i h t} up to the term of `order` (see bch)."""
        return self.processor.d_add(*(self.bch(a, n) for n in range(int(order) + 1)))


@lru_cache(maxsize=None)
def _bch(h: WMWrapper, a: WMWrapper, a_order, n: int) -> WMWrapper:
    if n == 0:
        return a
    prev = _bch(h, a, a_order, n - 1)
    return ((h @ prev - prev @ h) * (I * t_ / n)).apply(lambda e: bn_simp(_mexpand(e)), on_npf=True)
