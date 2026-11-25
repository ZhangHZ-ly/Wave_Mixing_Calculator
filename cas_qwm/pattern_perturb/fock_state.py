from __future__ import annotations
from collections.abc import Sequence
from sympy.core.sympify import sympify
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.add import Add
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.combinatorial.factorials import ff, rf
from sympy.physics.quantum.boson import BosonOp
from sympy.physics.quantum.fermion import FermionOp
from ..op_pattern.boson_pattern import BosonNum
from .pattern_form import PF

__all__ = ['FockState']


def _nat_n(n, fermion=False):
    n = sympify(n)
    if n.is_integer is False or n.is_negative or (fermion and (n - 1).is_positive):
        raise ValueError(f'{n} cannot be taken as the occupation number of a '
                         f'{"fermion" if fermion else "boson"} mode')
    return n


def _mode_name(m):
    if isinstance(m, str):
        return Symbol(m)
    name = getattr(m, 'name', None)
    return sympify(m) if name is None else name


def _is_base(x):
    return isinstance(x, Sequence) and not any(isinstance(e, Sequence) for e in x)


def _factors(p):
    if isinstance(p, (BosonOp, FermionOp)):
        return ((p.is_annihilation, S.One),)
    ia = bool(p.left_base.is_annihilation)
    return tuple((ia ^ bool(i & 1), e) for i, e in enumerate(p.exps))[::-1]


class FockState:
    """
    Superposition of Fock basis states, stored as {(n_b1, ..., n_f1, ...): coefficient}
    with the modes sorted by name; fermion bases are (c_1^†)^{n_1} (c_2^†)^{n_2} ... |0>.

    :param bs, fs: Boson / fermion modes (operators or names).
    :param states: (basis, coefficient) pairs or {basis: coefficient} dicts; shorter bases
    are padded with zeros.
    :param ket: Ket, or bra (with conjugated coefficients).
    :param strict: Check the occupation numbers (symbolic ones pass unless known invalid).
    """
    __slots__ = ('ket', 'boson_modes', 'fermion_modes', 'state_dict')

    def __init__(self, bs, fs, *states, ket=True, strict=True):
        self.ket = bool(ket)
        self.boson_modes = tuple(sorted({_mode_name(b) for b in bs}, key=Symbol.sort_key))
        self.fermion_modes = tuple(sorted({_mode_name(f) for f in fs}, key=Symbol.sort_key))
        self.state_dict = {}
        for state in states:
            self._rec_fock_key(self.state_dict, state, strict=strict)

    @classmethod
    def _new(cls, src: FockState, state_dict: dict, ket=None) -> FockState:
        obj = object.__new__(cls)
        obj.ket = src.ket if ket is None else ket
        obj.boson_modes, obj.fermion_modes = src.boson_modes, src.fermion_modes
        obj.state_dict = state_dict
        return obj

    def _make_fock_base(self, seq: Sequence, strict=True):
        seq, l = tuple(seq), self.nm
        if len(seq) > l:
            raise ValueError(f'The basis {seq} has more occupation numbers than the {l} modes')
        if strict:
            nb = len(self.boson_modes)
            seq = tuple(_nat_n(n, i >= nb) for i, n in enumerate(seq))
        else:
            seq = tuple(sympify(n) for n in seq)
        return seq + (S.Zero,) * (l - len(seq))

    def _rec_fock_key(self, sd: dict, state, strict=True):
        if hasattr(state, 'items'):
            items = state.items()
        elif isinstance(state, Sequence) and len(state) == 2 and _is_base(state[0]):
            items = (state,)
        else:
            raise ValueError('Invalid input format, required (basis, coefficient) or {basis: coefficient}')
        for k_raw, v in items:
            k = self._make_fock_base(k_raw, strict)
            sd[k] = sd.get(k, S.Zero) + sympify(v)

    def update(self, *states, strict=True):
        """Adds (basis, coefficient) pairs / dicts, or a single basis and coefficient, to the state."""
        if len(states) == 2 and _is_base(states[0]) and not isinstance(states[1], Sequence) \
                and not hasattr(states[1], 'items'):
            states = ((states[0], states[1]),)
        for state in states:
            self._rec_fock_key(self.state_dict, state, strict=strict)

    def __repr__(self):
        side = 'ket' if self.ket else 'bra'
        return f'FockState({self.boson_modes}, {self.fermion_modes}, {self.state_dict}, {side})'

    def __bool__(self):
        return any(v != 0 for v in self.state_dict.values())

    @property
    def nm(self):
        return len(self.boson_modes) + len(self.fermion_modes)

    @property
    def modes(self):
        return self.boson_modes + self.fermion_modes

    @property
    def amplitude(self):
        """Norm of the state."""
        return sqrt(Add(*(v.conjugate() * v for v in self.state_dict.values())).expand().factor())

    def normalize(self, renew=False):
        """Normalized state: a new one, or this one updated with renew=True."""
        amp = self.amplitude
        new_states = {k: v / amp for k, v in self.state_dict.items()}
        if renew:
            self.state_dict = new_states
            return self
        return self._new(self, new_states)

    def adjoint(self):
        return self._new(self, {k: v.conjugate() for k, v in self.state_dict.items()}, not self.ket)

    def _apply_term(self, bs, t, fs, k):
        occ = dict(zip(self.modes, k))
        coef = S.One
        for p in reversed(fs):
            j = self.fermion_modes.index(p.name)
            n, n_ops = occ[p.name], S.Zero
            for ia, e in _factors(p):
                if not e.is_Integer:
                    raise ValueError(f'Cannot apply the fermion pattern {p} with symbolic exponents')
                if e > 1:
                    return None
                coef *= n if ia else 1 - n
                n += -1 if ia else 1
                n_ops += 1
            if coef == 0:
                return None
            coef *= S.NegativeOne ** (n_ops * Add(*(occ[m] for m in self.fermion_modes[:j])))
            occ[p.name] = n
        t = t.xreplace({BosonNum(b): occ[b] for b in self.boson_modes})
        if not t.is_commutative:
            raise ValueError(f'The operator tail {t} is not a function of number operators')
        coef *= t
        for p in reversed(bs):
            n = occ[p.name]
            for ia, e in _factors(p):
                coef *= sqrt(ff(n, e)) if ia else sqrt(rf(n + 1, e))
                n += -e if ia else e
            if coef == 0:
                return None
            occ[p.name] = n
        return tuple(occ[m] for m in self.modes), coef

    def operate(self, operator):
        """
        Operate the operator (Expr / PF) on the Fock state.
        The state must include all the modes in the operator,
        and the operator cannot include any noncommutative Symbol variable.
        A bra is operated from the right: <ψ|O = (O^† |ψ>)^†.
        """
        op = PF.from_val(operator)
        if op.rem != 0 or op.npf != 0:
            raise ValueError('The operator must be fully in pattern form')
        if not self.ket:
            return self.adjoint().operate(op.adjoint()).adjoint()
        b_modes, f_modes, nc = op.pattern_modes
        if nc:
            raise ValueError(f'The operator includes noncommutative symbols {set(nc)}')
        missing = (b_modes - set(self.boson_modes)) | (f_modes - set(self.fermion_modes))
        if missing:
            raise ValueError(f'The state does not include the modes {set(missing)}')
        new = {}
        for k, c in self.state_dict.items():
            for (bs, fs), t in op.table.items():
                if (res := self._apply_term(bs, t, fs, k)) is not None:
                    nk, f = res
                    new[nk] = new.get(nk, S.Zero) + c * f
        return self._new(self, {k: v for k, v in new.items() if v != 0})

    def inner_prod(self, other):
        """
        Calculates the inner product between two Fock states.
        The two states must include the same modes.

        The other state will be flipped with adjoint() if they are both bras or kets.
        """
        if (self.boson_modes, self.fermion_modes) != (other.boson_modes, other.fermion_modes):
            raise ValueError('The two states must include the same modes')
        if self.ket == other.ket:
            other = other.adjoint()
        bra, ket = (other, self) if self.ket else (self, other)
        return Add(*(v * ket.state_dict[k] for k, v in bra.state_dict.items() if k in ket.state_dict))
