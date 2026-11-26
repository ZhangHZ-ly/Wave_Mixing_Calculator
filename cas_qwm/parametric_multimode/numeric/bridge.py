"""Bridge between the symbolic framework (WMWrapper, FockState) and the numerical evolution."""
import numpy as np
from sympy import Dummy, Poly, PolynomialError, Symbol, lambdify, sympify
from sympy.physics.quantum.boson import BosonOp
from ...op_pattern import BosonNum
from ...pattern_perturb import FockState
from ...wm_variables import nm
from ..multimode import MultimodeNum, MultimodeNum2
from ..wm_wrapper import WMWrapper
from .operators import Op, identity, diagonal, K_plus, K_minus, K_0, N_0, N_1, N_2

__all__ = ['Layout', 'observable', 'initial_state', 'to_fock']


class Layout:
    """
    Correspondence between the modes of a (down-conversion or FWM) processor and the chains of
    an Evolution. The multimode sums Σ run over all the chains: a collective chain of n modes
    adds n to n_m, an explicit chain adds its own term (like the separate mode of
    seeded_separate, the rest keeps n_m; a degenerate mode enters through the degenerate map of
    assert_mode_number). The single-mode operators a_1, a_2 of the processor act on the
    explicit chain `single` (default: the first one); other explicit chains are named by
    `names` {chain: (signal, idler) or name} (default: a_1_k, a_2_k, k = 1, 2, ...).
    """

    def __init__(self, processor, chains, single=None, names=None):
        from ..wm_solver import fwm_processor  # deferred: wm_solver imports this package's parents
        if not processor.dc:
            raise ValueError('Only down-conversion and FWM processors have a numerical evolution')
        self.processor, self.chains = processor, tuple(chains)
        self.process = 'fwm' if processor is fwm_processor else 'twm'
        self.drive = processor.nd.name
        n1, n2 = processor.a[0].names
        explicit = [i for i, c in enumerate(self.chains) if c.explicit]
        self.single = explicit[0] if single is None and explicit else single
        names = dict(names or {})
        k = 0
        for i in explicit:
            if i not in names:
                if i == self.single:
                    names[i] = (n1, n2)
                else:
                    k += 1
                    names[i] = (Symbol(f'{n1}_{k}'), Symbol(f'{n2}_{k}'))
            nm_i = names[i] if isinstance(names[i], tuple) else (names[i],)
            names[i] = tuple(sympify(x) for x in nm_i)[:1 if self.chains[i].kind == 'degenerate' else 2]
        self.names = names
        self.modes = {x: (i, f) for i, ns in names.items() for f, x in enumerate(ns)}
        if self.single is not None and self.chains[self.single].kind == 'degenerate':
            self.modes[n2] = (self.single, 0)
        self.n_m = sum(c.n for c in self.chains if not c.explicit)

    def factor(self, name, annihilation):
        """Op factor of the single-mode operator of `name`."""
        if name == self.drive:
            return ('a0' if annihilation else 'a0d', None)
        if name not in self.modes:
            raise ValueError(f'Mode {name} is not in the layout')
        c, f = self.modes[name]
        return (f'a{f + 1}' + ('' if annihilation else 'd'), c)


def _pattern(layout, b):
    if isinstance(b, BosonOp):
        return [layout.factor(b.name, b.is_annihilation)]
    ann = bool(b.left_base.is_annihilation)
    return [layout.factor(b.name, ann ^ bool(i & 1)) for i, e in enumerate(b.exps) for _ in range(int(e))]


def _atom(layout, x):
    n1, n2 = layout.processor.a[0].names
    if x == nm:
        return lambda grid, d, L: layout.n_m
    if isinstance(x, BosonNum):
        if x.name == layout.drive:
            return lambda grid, d, L: np.maximum(grid.drive(L), 0)
        c, f = layout.modes[x.name]
        return lambda grid, d, L: grid.numbers(d, c, f)
    if isinstance(x, MultimodeNum2):
        return lambda grid, d, L: 2 * (grid.states + grid.k(d)).sum(1)
    if isinstance(x, MultimodeNum) and x.name in (n1, n2):
        f, disp = int(x.name == n2), float(x.displacement)

        def mm(grid, d, L):
            res = 0
            for c, ch in enumerate(layout.chains):
                w = 0.5 if ch.kind == 'degenerate' else 1
                res = res + w * (grid.numbers(d, c, f) + disp * ch.n)
            return res
        return mm
    raise ValueError(f'{x} cannot be evaluated numerically')


def _number_op(layout, x):
    n1, n2 = layout.processor.a[0].names
    if x == nm:
        return layout.n_m * identity
    if isinstance(x, BosonNum):
        if x.name == layout.drive:
            return N_0
        c, f = layout.modes[x.name]
        return (N_2 if f else N_1)(c)
    if isinstance(x, MultimodeNum2):
        return sum(2 * K_0(c) for c in range(len(layout.chains)))
    if isinstance(x, MultimodeNum) and x.name in (n1, n2):
        f, disp = int(x.name == n2), float(x.displacement)
        return sum((0.5 if ch.kind == 'degenerate' else 1) * ((N_2 if f else N_1)(c) + disp * ch.n)
                   for c, ch in enumerate(layout.chains))
    raise ValueError(f'{x} cannot be evaluated numerically')


def _tail(layout, t, subs):
    t = sympify(t).xreplace(subs)
    atoms = [x for x in t.atoms(BosonNum, MultimodeNum, MultimodeNum2)] + ([nm] if t.has(nm) else [])
    ds = [Dummy() for _ in atoms]
    t = t.xreplace(dict(zip(atoms, ds)))
    if t.free_symbols - set(ds) or not t.is_commutative:
        raise ValueError(f'Unresolved symbols or operators in {t} (pass their values in subs)')
    if not atoms:
        return complex(t)
    try:
        poly = Poly(t, *ds)
    except PolynomialError:
        poly = None
    if poly is not None and all(c.is_number for c in poly.coeffs()):
        ops = [_number_op(layout, x) for x in atoms]
        res = 0
        for exps, c in poly.terms():
            term = complex(c) * identity
            for o, e in zip(ops, exps):
                term = term * o ** e
            res = res + term
        return res
    f, fs = lambdify(ds, t, 'numpy'), [_atom(layout, x) for x in atoms]
    return diagonal(lambda grid, d, L: np.broadcast_to(f(*(g(grid, d, L) for g in fs)), (grid.dim,)))


def _pf(layout, v, subs):
    res = 0
    for (bs, fs), t in v.table.items():
        if fs:
            raise ValueError('Fermion patterns have no numerical evolution here')
        ops = Op([(1, [x for b in bs for x in _pattern(layout, b)])])
        res = res + ops * _tail(layout, t, subs)
    if v.npf != 0:
        res = res + _tail(layout, v.npf, subs)
    return res


def _key(layout, k):
    if isinstance(k, BosonOp):
        return Op([(1, [layout.factor(k.name, k.is_annihilation)])])
    res, (i, *exps) = identity, k
    for e in exps:
        A = sum((K_minus(c) if layout.processor.a[i].annih1 else K_plus(c)) for c in range(len(layout.chains)))
        res = res * A ** int(e)
        i = not i
    return res


def observable(w, layout: Layout, subs=None) -> Op:
    """
    Op of a symbolic operator at t = 0 (WMWrapper, or what WMWrapper.from_val takes) in multimode
    notation, on the chains of `layout`; subs gives the values of its other symbols.
    """
    if not isinstance(w, WMWrapper):
        w = WMWrapper.from_val(w, layout.processor)
    subs = {sympify(k): sympify(v) for k, v in (subs or {}).items()}
    dict_op = lambda d: sum(_key(layout, k) * _pf(layout, v, subs) for k, v in d.items())
    if w.b_left is None:
        return dict_op(w.dict_repr) + 0 * identity
    if w.b_left:
        return sum(_key(layout, b) * dict_op(d) for b, d in w.dict_repr.items())
    return sum(dict_op(d) * _key(layout, b) for b, d in w.dict_repr.items())


def initial_state(layout: Layout, fock: FockState = None, seeds=None) -> dict:
    """
    Keyword arguments `initial` and `seeds` of Evolution from a FockState of the drive and
    explicit modes (default: their vacuum) and coherent seeds {mode: beta} of explicit modes.
    """
    explicit = [i for i, c in enumerate(layout.chains) if c.explicit]
    comps = {}
    if fock is None:
        comps[(0, tuple((0, 0) if layout.chains[i].kind == 'pair' else 0 for i in explicit))] = 1
    else:
        if not fock.ket or fock.fermion_modes:
            raise ValueError('A ket of boson modes is expected')
        for basis, amp in fock.state_dict.items():
            occ, n0 = {i: [0, 0] for i in explicit}, 0
            for name, n in zip(fock.boson_modes, basis):
                if name == layout.drive:
                    n0 = int(n)
                elif name in layout.modes:
                    c, f = layout.modes[name]
                    occ[c][f] += int(n)
                else:
                    raise ValueError(f'Mode {name} of the FockState is not in the layout')
            key = tuple(tuple(occ[i]) if layout.chains[i].kind == 'pair' else occ[i][0] for i in explicit)
            comps[(n0, key)] = comps.get((n0, key), 0) + complex(amp)
    return {'initial': [(n0, key, amp) for (n0, key), amp in comps.items()],
            'seeds': {layout.modes[sympify(name)]: complex(beta) for name, beta in (seeds or {}).items()}}


def to_fock(evolution, layout: Layout, N=0, time=0, tol=1e-14) -> FockState:
    """Evolved state as a FockState of the drive and the explicit modes (no collective chain), with
    the drive in |N> added to the input's drive numbers; time: index into the times."""
    if any(not c.explicit for c in layout.chains):
        raise ValueError('Collective chains have no FockState (use distribution)')
    if evolution._quad:
        raise ValueError('Seeds averaged by quadrature have no FockState (seed_nodes="poisson")')
    blocks = evolution.blocks(N)
    grid = evolution.fock_grid
    names = [layout.drive] + [x for i in sorted(layout.names) for x in layout.names[i]]
    order = sorted(range(len(names)), key=lambda j: Symbol.sort_key(names[j]))
    states = {}
    for (d, L), v in blocks.items():
        v = v if np.ndim(v) == 1 else v[time]
        for idx in np.nonzero(abs(v) > tol)[0]:
            occ = [int(round(L - grid.r * grid.m_sum[idx]))]
            for c in sorted(layout.names):
                n1 = int(grid.numbers(d, c, 0)[idx])
                occ += [n1] if layout.chains[c].kind == 'degenerate' else [n1, int(grid.numbers(d, c, 1)[idx])]
            states[tuple(occ[j] for j in order)] = v[idx]
    return FockState(names, [], states)
