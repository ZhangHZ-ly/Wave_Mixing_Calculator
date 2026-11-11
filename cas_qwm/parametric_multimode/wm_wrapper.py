from collections import defaultdict
from itertools import product
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.expr import Expr
from sympy.core.symbol import Symbol
from sympy.core.add import Add
from sympy.core.power import Pow
from sympy.core.function import diff, _mexpand
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.dagger import Dagger
from ..pattern_perturb import PF, PF0, PF1
from ..op_pattern import BosonNum
from ..op_pattern.boson_pattern import fac
from ..pattern_perturb.pattern_form import pf_add, _vpk, bn_simp
from ..pattern_perturb.perturb_expander import gperm
from .multimode import MultimodeNum, MultimodeNum2
from .wm_processor import (WMProcessor, wm_processors, vac_pf, _dr_prep,
                           _t_mul, _tdz, a_s, t_, _g, nlc, nm)

__all__ = ['WMWrapper']

def _sum_dl(dl: dict) -> dict:
    return {k: vs for k, ps in dl.items() if (vs := pf_add(*ps))}

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
    """

    def __init__(self, dc, dict_repr=None, b_left=None):
        self.processor = dc if isinstance(dc, WMProcessor) else wm_processors[bool(dc)]
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
    @property
    def dc(self) -> bool:
        return self.processor.dc

    @classmethod
    def from_val(cls, val, dc=True) -> 'WMWrapper':
        proc = dc if isinstance(dc, WMProcessor) else wm_processors[bool(dc)]
        if isinstance(val, cls):
            return val if val.dc == proc.dc else cls(proc, val.dict_repr, val.b_left)
        return proc.from_expr(val)

    def __bool__(self) -> bool:
        return bool(self.dict_repr)

    def __repr__(self) -> str:
        return f"WMWrapper(dc={self.dc}, {self.dict_repr}, b_left={self.b_left})"

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
            return WMWrapper(p, {}, None)
        return p.d_add(*swapped)

    def __eq__(self, other) -> bool:
        if not isinstance(other, WMWrapper):
            return self.__eq__(self.from_val(other))
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

    @property
    def parity(self):
        """Returns the weak-field parity (0 or 1), or None if b_left is None."""
        if self.b_left is None or not self.dict_repr:
            return None
        return self.processor.f_parity(tuple(self.dict_repr))

    def apply(self, f, self_update=False) -> 'WMWrapper':
        """
        Applies f to every leaf PF (or its scalar/number-operator tails via PF.apply)
        in dict_repr, returning a new WMWrapper.
        """
        def _apply_node(e):
            if isinstance(e, dict):
                return {k: nv for k, v in e.items() if (nv := _apply_node(v))}
            if isinstance(e, PF):
                try:
                    return e.apply(f, on_npf=True)
                except (TypeError, AttributeError):
                    pass
            return f(e)

        res = _apply_node(self.dict_repr)
        if self_update:
            self.dict_repr = res
            if not res:
                self.b_left = None
            return self
        return WMWrapper(self.processor, res, self.b_left if res else None)

    def expand(self, order, sub_g: bool = False, gc=None) -> 'WMWrapper':
        """Expands this WMWrapper to the given perturbative order."""
        res = self.apply(lambda p: self.processor.pe.expand_pf(p, order))
        if sub_g and gc is not None:
            return res.apply(lambda t: t.subs({t_: _g / gc}))
        return res

    def _combine_tcp_vac(self, k, bs_weak):
        """
        Combines weak-field BosonPat operators with a multimode exponent tuple k
        when acting on vacuum |0> on the right.
        """
        if not bs_weak:
            yield S.One, k
            return

        p = self.processor
        try:
            cop_a1, cop_exp, bs_other = p.match_cop(bs_weak)
        except ValueError:
            cop_a1, cop_exp, bs_other = None, S.Zero, bs_weak

        if cop_exp > 0 and not bs_other:
            if cop_a1:
                return
            new_k = k
            for _ in range(cop_exp):
                new_k = _t_mul(new_k, (None,), 0)
            s_comb = fac(cop_exp) / gperm(nm + cop_exp - 1, cop_exp)
            yield s_comb, new_k
            return

        if len(bs_weak) == 1:
            b = bs_weak[0]
            if b.left_base.is_annihilation:
                return
            b_op = BosonOp(b.name, b.left_base.is_annihilation)
            if k == (None,):
                return
            i, *exps = k
            for s, _, nes in p.bt_swap(b_op, i, exps, to_left=True):
                yield s, _tdz(i, nes)
            return

        yield S.One, k

    def eval_vacuum(self, vac_modes) -> 'WMWrapper':
        """
        Evaluates the partial or full expectation value of this WMWrapper
        under vacuum input assertions for the specified mode(s).
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

        def _collapse_mmn(name, displacement):
            if name in vac_sets[1]:
                return displacement * nm
            return MultimodeNum(name, displacement)

        def _collapse_mmn2(n1, n2, par):
            in1, in2 = n1 in vac_sets[1], n2 in vac_sets[1]
            if not in1 and not in2:
                return MultimodeNum2(n1, n2, par)
            if not par:
                return nm if (in1 and in2) else MultimodeNum(n2 if in1 else n1, 1)
            if in1 and in2:
                return S.Zero
            return MultimodeNum(n2) if in1 else -MultimodeNum(n1)

        _collapse_num = (lambda expr: sympify(expr).replace(MultimodeNum, _collapse_mmn)
                         .replace(MultimodeNum2, _collapse_mmn2))

        def _clean_pf(pf_obj: PF) -> PF:
            applied = pf_obj.apply(_collapse_num)
            clean_table = {k: v for k, v in applied.table.items() if v != 0}
            return PF(clean_table, applied.npf) if (clean_table or applied.npf != 0) else PF0

        tuple_cache = {}

        def _vac_tuple(k: tuple):
            if k in tuple_cache:
                return tuple_cache[k]
            if len(k) == 1:
                tuple_cache[k] = S.One
                return S.One
            if not p.dc and len(vac_sets[1]) == 2:
                tuple_cache[k] = S.Zero
                return S.Zero

            ann_idx = True if p.dc else (a_s.name not in vac_sets[1])
            i, *exps = k
            if bool(i) != ann_idx or (len(exps) & 1) or sum(exps[0::2]) != sum(exps[1::2]):
                tuple_cache[k] = S.Zero
                return S.Zero

            B = p.a[ann_idx]
            exps = list(exps)
            acc = S.One
            while exps:
                c_r = exps.pop()
                a_r = exps.pop()
                if a_r > c_r:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                step_terms = [
                    _collapse_num(factor)
                    for _, e1_op, factor in B.swap_exps(a_r, c_r, order=a_r, left_p=True)
                    if e1_op is S.One
                ]
                if not step_terms:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                step_op = _collapse_num(Add(*step_terms))
                if step_op == 0:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                acc = _collapse_num(_mexpand(step_op * acc))
                if acc == 0:
                    tuple_cache[k] = S.Zero
                    return S.Zero
                rem_c = c_r - a_r
                if rem_c > 0:
                    if not exps:
                        tuple_cache[k] = S.Zero
                        return S.Zero
                    exps[-1] += rem_c

            tuple_cache[k] = acc
            return acc

        def _vac_dict(d: dict) -> dict:
            if vac_sets[0]:
                d = {k: vp for k, pf_v in d.items() if (vp := vac_pf(pf_v, vac_sets[0]))}
                if not d:
                    return {}

            if not vac_sets[1]:
                return {k: cp for k, pf_v in d.items() if (cp := _clean_pf(pf_v))}

            pfs = []
            for k, pf_v in d.items():
                if not pf_v:
                    continue
                if pf_v.npf != 0:
                    ck = _vac_tuple(k)
                    if ck != 0 and (c_npf := _collapse_num(pf_v.npf)) != 0:
                        pfs.append(PF({}, _collapse_num(ck * c_npf)))

                for (bs, fs), tail in pf_v.table.items():
                    bs_weak = tuple(b for b in bs if b.name in weak_names)
                    bs_strong = tuple(b for b in bs if b.name not in weak_names)
                    vp = _clean_pf(PF({(bs_strong, fs): tail}, S.Zero))
                    if not vp:
                        continue
                    if not bs_weak:
                        ck = _vac_tuple(k)
                        if ck != 0:
                            pfs.append(_collapse_num(ck) * vp)
                    else:
                        for s_comb, new_k in self._combine_tcp_vac(k, bs_weak):
                            ck = _vac_tuple(new_k)
                            if ck != 0:
                                pfs.append(_collapse_num(s_comb * ck) * vp)

            if not pfs:
                return {}
            tot = pf_add(*pfs)
            return {(None,): tot} if tot else {}

        if self.b_left is None:
            return WMWrapper(p, _vac_dict(self.dict_repr), None)

        if len(vac_sets[1]) == 2:
            return WMWrapper(p, {}, None)
        if not vac_sets[1]:
            res = {b: vd for b, d in self.dict_repr.items() if (vd := _vac_dict(d))}
            return WMWrapper(p, res, self.b_left if res else None)

        ws = []
        for b, d in self.dict_repr.items():
            if not d:
                continue
            if b.name in vac_sets[1]:
                if (b.is_annihilation and not self.b_left) or (not b.is_annihilation and self.b_left):
                    continue
                swapped = p.bd_swap(b, d, to_left=not b.is_annihilation)
                b_partner = p.odag(b)
                d_partner = swapped.get(b_partner)
                if d_partner and (vd := _vac_dict(d_partner)):
                    w_part = WMWrapper(p, {b_partner: vd}, not b.is_annihilation)
                    ws.append(w_part if w_part.b_left == self.b_left else w_part.flip())
            else:
                if (vd := _vac_dict(d)):
                    ws.append(WMWrapper(p, {b: vd}, self.b_left))

        return p.d_add(*ws) if ws else WMWrapper(p, {}, None)

    def eval_coherent(self, order: int = 0) -> 'WMWrapper':
        """
        Evaluates this WMWrapper under coherent state assertion for the strong driving field.
        """
        p = self.processor
        alpha = Symbol('alpha_0') if p.dc else Symbol('alpha_2')
        sub_dict = {p.nd.name: alpha}

        def _cohr_dict(d: dict) -> dict:
            return {
                k: cp for k, v in d.items()
                if (cp := p.pe.cohr_pf(v, sub_dict, order))
            }

        if self.b_left is None:
            return WMWrapper(p, _cohr_dict(self.dict_repr), None)
        res = {b: cd for b, d in self.dict_repr.items() if (cd := _cohr_dict(d))}
        return WMWrapper(p, res, self.b_left if res else None)

    @property
    def expr(self) -> Expr:
        """Converts the wrapper representation into a SymPy Expr."""
        p = self.processor
        if self.b_left is None:
            return p._d_expr(self.dict_repr)
        if self.b_left:
            return Add(*(b * p._d_expr(d) for b, d in self.dict_repr.items()))
        return Add(*(p._d_expr(d) * b for b, d in self.dict_repr.items()))

    def to_single_mode(self) -> 'WMWrapper':
        """
        Converts multimode operator representations into single-mode boson
        operators (mode_expr) with n_m = 1, returning a single-mode WMWrapper.
        """
        p = self.processor
        def _dict_sm(d: dict) -> PF:
            return pf_add(*(
                p._smk(k) * v.apply(p._sm_expr, on_npf=True)
                for k, v in d.items()
            ))

        if self.b_left is None:
            pf_res = _dict_sm(self.dict_repr)
        elif self.b_left:
            pf_res = pf_add(*(p._smk(b) * _dict_sm(d) for b, d in self.dict_repr.items()))
        else:
            pf_res = pf_add(*(_dict_sm(d) * p._smk(b) for b, d in self.dict_repr.items()))
        return WMWrapper(p, pf_res.apply(bn_simp), None)

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
            return WMWrapper(p, _adj_dict(self.dict_repr), None)
        res = {Dagger(b): ad for b, d in self.dict_repr.items() if (ad := _adj_dict(d))}
        return WMWrapper(p, res, (not self.b_left) if res else None)

    def __add__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        return self.processor.d_add(self, other)

    def __radd__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__add__(self)

    def __neg__(self) -> 'WMWrapper':
        return self.apply(lambda p: -p)

    def __sub__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        return self.processor.d_add(self, -other)

    def __rsub__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__sub__(self)

    def __mul__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        p = self.processor
        if not self or not other:
            return WMWrapper(p, {}, None)
        if self.b_left is None or other.b_left is None:
            return self.__matmul__(other)
        if self.parity == other.parity:
            raise ValueError('Solution parity unbalanced')
        res_l = defaultdict(list)
        if other.b_left:
            e1 = (self.flip() if self.b_left else self).dict_repr
            for (b1, d1), (b2, d2) in product(e1.items(), other.dict_repr.items()):
                if b1 == Dagger(b2):
                    bn = BosonNum(b1.name) + int(b1.is_annihilation)
                    p.d_mul(res_l, d1, d2, bn)
                elif b1.compare(b2) == 1:
                    p.d_mul(res_l, d1, d2, b2.is_annihilation, True)
                else:
                    p.d_mul(res_l, d1, d2, b1.is_annihilation, True)
        else:
            if self.b_left:
                for (b1, d1), (b2, d2) in product(self.dict_repr.items(),
                                                other.dict_repr.items()):
                    mid_dl = defaultdict(list)
                    p.d_mul(mid_dl, d1, d2)
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
        return WMWrapper(p, _sum_dl(res_l), None)

    def __rmul__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__mul__(self)

    def __matmul__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        e1, e2 = self.dict_repr, other.dict_repr
        p = self.processor
        if not e1 or not e2:
            return WMWrapper(p, {}, None)

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
                    p.d_mul(res_l, d1, d2, mn)
                elif b1.compare(b2) == 1:
                    p.d_mul(res_l, d1, d2, b2.is_annihilation)
                else:
                    p.d_mul(res_l, d1, d2, b1.is_annihilation)
            return WMWrapper(p, _sum_dl(res_l), None)

        if is_f1:
            if self.b_left:
                for b, d in e1.items():
                    res_dl = defaultdict(list)
                    p.d_mul(res_dl, d, e2)
                    if (res_b := _sum_dl(res_dl)):
                        result[b] = res_b
                return WMWrapper(p, result, True)
            res_dl = defaultdict(lambda: defaultdict(list))
            for b, d1 in e1.items():
                for nb, p2_prime in p.bd_swap(b, e2, to_left=False).items():
                    p.d_mul(res_dl[nb], d1, p2_prime)
            result = {nb: rb for nb, dl in res_dl.items() if (rb := _sum_dl(dl))}
            return WMWrapper(p, result, False)

        if is_f2:
            if not other.b_left:
                for b, d in e2.items():
                    res_dl = defaultdict(list)
                    p.d_mul(res_dl, e1, d)
                    if (res_b := _sum_dl(res_dl)):
                        result[b] = res_b
                return WMWrapper(p, result, False)
            res_dl = defaultdict(lambda: defaultdict(list))
            for b, d2 in e2.items():
                for nb, p1_prime in p.bd_swap(b, e1, to_left=True).items():
                    p.d_mul(res_dl[nb], p1_prime, d2)
            result = {nb: rb for nb, dl in res_dl.items() if (rb := _sum_dl(dl))}
            return WMWrapper(p, result, True)

        res_dl = defaultdict(list)
        p.d_mul(res_dl, e1, e2)
        return WMWrapper(p, _sum_dl(res_dl), None)

    def __rmatmul__(self, other) -> 'WMWrapper':
        return WMWrapper.from_val(other, self.processor).__matmul__(self)

    def __truediv__(self, other) -> 'WMWrapper':
        other = WMWrapper.from_val(other, self.processor)
        denom = other.as_scalar
        if denom is not None:
            inv_w = WMWrapper(self.processor, {(None,): PF({_vpk: denom ** S.NegativeOne})}, None)
            return self.__matmul__(inv_w)
        raise ValueError("Division is only supported by scalar or number-operator expressions.")

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
            return WMWrapper(p, {(None,): PF({_vpk: base_scalar ** p_sym})}, None)

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
                    return WMWrapper(p, {(None,): v ** n_int}, None)
                except Exception:
                    return WMWrapper(p, {(None,): PF(npf=Pow(v.expr(), n_int))}, None)
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
            order = getattr(p, '_evolved_order', 1)
            return p.expand_func_d(lambda x: x ** p_sym, order, self)
        raise ValueError("Exponentiation for this operator type only supports non-negative integers or expanded multimode dicts.")

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
                lambda v: p.n_simp(diff(v, t_, order_n, *rest_args, **kwargs) * gc_inv_n)
            )
        if is_wrt_t:
            var = t_
        return self.apply(lambda v: diff(v, var, *rest_args, **kwargs))

    Derivative = diff
