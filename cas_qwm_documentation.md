# `cas_qwm` — Python API Reference

**`cas_qwm`** (*Computer Algebra System for Quantum Wave Mixing*) is a SymPy-based library for symbolic calculations with canonical operators in quantum mechanics and quantum optics. It is the engine behind the web calculator described in the [main README](readme.md).

It provides three main capabilities:

1. **Operator patterns** — products of boson / fermion operators of the same mode are collected into a single object, and boson patterns are rewritten in terms of number operators as far as possible.
2. **Pattern form and perturbative expansion** — any expression is reordered into a unique *pattern form*, enabling equality tests and fast algebra in the non-commutative operator algebra; expressions can be Taylor-expanded up to a given perturbative order, where the order of every symbol / operator is user-defined.
3. **Multimode wave mixing** — Heisenberg-picture solutions of three- and four-wave mixing, solved order by order in the pump depletion, together with an algebra for the resulting multimode operator expressions.

All examples on this page were run with SymPy 1.14 from the repository root.

---

## Contents

- [Installation and import](#installation-and-import)
- [Architecture](#architecture)
- [Printing conventions](#printing-conventions)
- [Layer 1 — `cas_qwm.op_pattern`](#layer-1--cas_qwmop_pattern)
  - [`BosonPat`](#bosonpat) · [`BosonNum`](#bosonnum) · [`FermionPat`](#fermionpat) · [`redo_mul`](#redo_mul) · [`NumShift` / `apply_num_shift`](#numshift--apply_num_shift) · [`UidCommutator`](#uidcommutator) · [Utilities](#utilities)
- [Layer 2 — `cas_qwm.pattern_perturb`](#layer-2--cas_qwmpattern_perturb)
  - [`pf`](#pf--pattern_form) · [`PF`](#pf-class) · [`PE`](#pe--perturbative-expander) · [`perturb_rec`](#perturb_rec)
- [Layer 3 — `cas_qwm.parametric_multimode`](#layer-3--cas_qwmparametric_multimode)
  - [Shared symbols](#shared-symbols) · [Multimode operators](#multimode-operators) · [`TWMSolver` / `FWMSolver`](#twmsolver--fwmsolver) · [`WMWrapper`](#wmwrapper) · [`WMProcessor`](#wmprocessor) · [`dsolve_pm1` / `dsolve_pm2` / `p2s`](#differential-equation-helpers)
- [Two strong inputs — `cas_qwm.twm_dual_stimulated`](#two-strong-inputs--cas_qwmtwm_dual_stimulated)
  - [Method](#method) · [`TWMDualSolver`](#twmdualsolver) · [`exact_fock_apply`](#exact_fock_apply) · [Accuracy](#accuracy) · [Jacobi elliptic functions](#jacobi-elliptic-functions--cas_qwmjacobi) · [`dsolve_lame`](#elliptic-polynomial-algebra--cas_qwmjacobidsolve_lame) · [`elliptic_algebra`](#elliptic-pf-coefficients--cas_qwmjacobielliptic_algebra)
- [Exported names](#exported-names)

---

## Installation and import

`cas_qwm` only depends on SymPy:

```bash
pip install sympy==1.14.0
```

Run Python from the repository root (or add it to `PYTHONPATH`) and import from the package:

```python
import cas_qwm
from cas_qwm import BosonPat, BosonNum, FermionPat, redo_mul, pf, PF, PE, TWMSolver
```

The package uses lazy loading ([PEP 562](https://peps.python.org/pep-0562/)): a submodule is only imported when one of its names is first accessed. Importing `cas_qwm.op_pattern` patches SymPy so that powers of `BosonOp` / `FermionOp` become `BosonPat` / `FermionPat` objects, and so that `Dagger` works on sums / integrals with limits.

## Architecture

The package is organised in three layers; each layer only depends on the ones above it.

| Layer | Module | Content |
|---|---|---|
| 1 | `cas_qwm.op_pattern` | Single-mode operator patterns (`BosonPat`, `FermionPat`), number operators (`BosonNum`), number shifts and commutators. |
| 2 | `cas_qwm.pattern_perturb` | Pattern-form reordering (`pf`), the `PF` algebra, perturbative expansion (`PE`). |
| 3 | `cas_qwm.parametric_multimode` | Multimode operators, wave-mixing processors (`WMProcessor`), solvers (`TWMSolver`, `FWMSolver`), the `WMWrapper` algebra, ODE helpers (`dsolve_pm1`, `dsolve_pm2`, `p2s`). |
| – | `cas_qwm.jacobi`, `cas_qwm.twm_dual_stimulated` | The `jacobi` package holds all the elliptic algebra: Jacobi elliptic functions (`jacobi.py`), the elliptic-polynomial ring with its closed-form Lamé solver (`dsolve_lame.py`), and the λ-series of that ring as `PF` coefficients with their `PE` coefficient algebra (`elliptic_algebra.py`). `twm_dual_stimulated` applies it to single-mode three-wave mixing with two strong inputs (`TWMDualSolver`). |

## Printing conventions

| Printed form | Meaning |
|---|---|
| `BP(a; 2, 2)` | Boson pattern $a^2 (a^\dagger)^2$; the exponents alternate between $a$ and $a^\dagger$, starting from the operator shown. |
| `BP(D(a); 2)` | Pattern starting with a creation operator: $(a^\dagger)^2$. `D(·)` stands for `Dagger(·)`. |
| `FP(c; 1, n, 1)` | Fermion pattern $c\,(c^\dagger)^n c$. |
| `N_(a)` | Number operator $\hat N_a = a^\dagger a$ (`BosonNum`). |
| `Σ(...)` | Sum over all mode pairs of a multimode process (`MultimodeCOP`, `MultimodeNum`, `MultimodeNum2`). |
| `PatternForm({...}, npf)` | A `PF` object: table of pattern-form terms and the part `npf` that has no pattern form. |
| `_t`, `χ`, `Δk`, `n_m` | Time, nonlinear coupling, phase mismatch, number of mode pairs. |

---

## Layer 1 — `cas_qwm.op_pattern`

### `BosonPat`

```python
BosonPat(name, is_annihilation, *exps, merge_ends=None)
```

Collects a product of operators of the same boson mode into one *pattern*.

- With 1 or 2 arguments, creates a plain `BosonOp(name, is_annihilation)`.
- With 3 or more arguments, the first operator is `BosonOp(name)` (annihilation) or its adjoint (if `is_annihilation` is false), and the remaining arguments are the exponents of the factors, which alternate between annihilation and creation. Exponents may be symbolic.
- With `merge_ends` set, the first argument must be an existing `BosonPat`, which is extended by further exponents.

Main members:

| Member | Description |
|---|---|
| `name`, `exps` | Mode name and exponent tuple. |
| `left_base`, `right_base` | Left-most / right-most factor base. |
| `parity_diff` | Number of creation minus number of annihilation operators. |
| `num_repr(left_p=True)` | Returns `(canonical_power, f(N))` such that the pattern equals `canonical_power * f(N)` (`left_p=True`) or `f(N) * canonical_power`. |
| `to_normal_order(via_dict=False)` | Normal-ordered form; `via_dict=True` uses a faster method valid only for known non-negative integer exponents. |
| `rewrite(BosonOp)` / `rewrite(BosonNum)` | Explicit product of operators / number-operator form. |

```python
>>> from cas_qwm import BosonPat as BP, BosonNum
>>> from sympy.physics.quantum.boson import BosonOp
>>> bp = BP('a', True, 2, 2)
>>> bp
BP(a; 2, 2)
>>> bp.num_repr()
(1, (1 + N_(a))*(2 + N_(a)))
>>> bp.rewrite(BosonOp)
a**2*Dagger(a)**2
>>> bp.rewrite(BosonNum)
(1 + N_(a))*(2 + N_(a))
>>> bp.to_normal_order()          # _k is a dummy summation index
Sum(FallingFactorial(2, _k)*binomial(2, _k)*BP(D(a); 2 - _k, 2 - _k), (_k, 0, 2))
```

### `BosonNum`

```python
BosonNum(mode)
```

Hermitian number operator $\hat N = a^\dagger a$ of the given mode (printed `N_(a)`). It is a subclass of `BosonPat` and supports differentiation with respect to it, which is used by the perturbative expansion.

### `FermionPat`

```python
FermionPat(name, is_annihilation, *exps, eval_exp=True, merge_ends=None)
```

Fermionic counterpart of `BosonPat`: represents $\cdots c^x (c^\dagger)^y c^z \cdots$. Because $c^2 = 0$, any product of operators of one fermion mode equals one of $0, 1, c, c^\dagger, c^\dagger c, c c^\dagger$; the constructor collapses the pattern to this form whenever all exponents are known (`eval_exp=True`). Patterns with unknown exponents are kept symbolic.

```python
>>> from cas_qwm import FermionPat as FP
>>> FP('c', True, 1, 1, 1)     # c c† c = c
c
>>> FP('c', True, 2)           # c² = 0
0
```

### `redo_mul`

```python
redo_mul(expr, prune_zero=True, num_repr=False)
```

Direct SymPy multiplication of operators does not merge factors of the same mode. `redo_mul` redoes all multiplications in an expression with the pattern rules; `num_repr=True` additionally rewrites $a^\dagger a$ as `N_(a)` (or $a a^\dagger$ as $1 \pm$ `N_(a)`).

```python
>>> from cas_qwm import redo_mul
>>> from sympy.physics.quantum.boson import BosonOp
>>> from sympy.physics.quantum.dagger import Dagger
>>> a = BosonOp('a')
>>> a**2 * Dagger(a)**2
BP(a; 2)*BP(D(a); 2)
>>> redo_mul(a**2 * Dagger(a)**2)
BP(a; 2, 2)
```

### `NumShift` / `apply_num_shift`

```python
NumShift(expr_num, *shift)
apply_num_shift(expr)
```

`NumShift` is an unevaluated object representing the reordering $f(\hat N)\, a^d \to a^d\, g(\hat N)$, i.e. the shift of number operators in `expr_num` by the exponents in `shift` (dicts `{op: d}` or pairs `(op, d)`). Its `doit()` performs the shift; `apply_num_shift(expr)` evaluates all `NumShift` objects in an expression. Keeping the shift unevaluated allows caching during pattern-form processing.

### `UidCommutator`

```python
UidCommutator(A, B)
```

*Unidentified* commutator between fermion patterns whose exponents are not fully known: depending on the parities of the exponents, it is either a commutator or an anticommutator. `doit()` evaluates it when possible.

### Utilities

| Function | Description |
|---|---|
| `extract_modes(expr)` | Returns a 5-tuple of sets: boson modes, fermion modes, non-commutative symbols, commutative symbols, and integration / summation variables. |
| `check_num(expr)` | Whether `expr` is expressed purely in number operators (and commutative quantities). |
| `in_boson_num(expr, mode, prune=False)` | Returns `(check_num(expr), commutes_with_N_mode)`. |
| `mul_pattern(e1, e2)` | Pattern-aware product of two factors. |

```python
>>> from cas_qwm import extract_modes
>>> from sympy import symbols
>>> x = symbols('x', real=True)
>>> a, b = symbols('a b', cls=BosonOp)
>>> extract_modes(x*a*Dagger(b))
({a, b}, set(), set(), {x}, set())
```

---

## Layer 2 — `cas_qwm.pattern_perturb`

### `pf` — pattern form

```python
pf(expr, as_dict=False, rem=False, ignore=(), _s=True)   # alias of pattern_form
```

Reorders a quantum expression into its **pattern form**: a sum of terms, each a product of a commutative coefficient, powers of canonical operators (at most one per mode, always on the left), and a function of number operators. The pattern form of equal expressions is identical, which makes it possible to establish equalities in the non-commutative operator algebra.

| Parameter | Description |
|---|---|
| `as_dict` | Output type: `False` → SymPy `Expr`; `None` → `PF` object; `True` → raw `dict`. |
| `rem` | If true, does not assume independent modes: also returns the commutator **remainder** generated while reordering, as a pair `(pattern_form, remainder)`. |
| `ignore` | Non-commutative types to be treated as if they were number operators. |
| `_s` | Whether to simplify the expression before reordering. |

```python
>>> from cas_qwm import pf
>>> from sympy import symbols, sin, cos
>>> a, b = symbols('a b', cls=BosonOp)
>>> pf(a**2*(cos(Dagger(a)*a) + Dagger(b)*sin(b*Dagger(b)))*Dagger(a)*b**2)
a*b*(-1 + N_(b))*(1 + N_(a))*sin(-1 + N_(b)) + a*BP(b; 2)*(1 + N_(a))*cos(1 + N_(a))

>>> pf(Dagger(b)*sin(b*Dagger(b))*Dagger(a), rem=True)
(Dagger(a)*Dagger(b)*sin(1 + N_(b)), -Dagger(b)*[Dagger(a),sin(1 + N_(b))] - [Dagger(a),Dagger(b)]*sin(1 + N_(b)))
```

Related functions: `is_pattern_form(*exprs)` checks whether expressions are already in pattern form; `bn_simp(expr)` simplifies expressions of number operators.

### `PF` class

```python
PF(table=None, npf=0, cache=None, rem=0)
```

Table representation of a pattern-formed expression. `table` maps keys `(boson_patterns, fermion_patterns)` — tuples of canonical-operator powers — to number-operator coefficients; `npf` holds the part of the expression that has no pattern form. Algebra on `PF` objects stays in pattern form and is usually much faster than working with `sympy.Expr`.

> Creating a `PF` directly from a dict does not validate the input; prefer `pf(expr, as_dict=None)` or `PF.from_val(expr)`.

| Member | Description |
|---|---|
| `PF.from_val(val)` | Converts an `Expr`, a `PF` or a `(PF, remainder)` pair to `PF`. |
| `+`, `-`, `*` (= `@`), `/` by scalars, `**` (non-negative integer) | Pattern-form algebra (modes assumed independent). |
| `expr(with_npf=True)` | Convert back to a SymPy expression. |
| `adjoint()` | Hermitian adjoint. |
| `apply(f, on_npf=False)` | Apply `f` to every coefficient (and to `npf` if requested). |
| `diff(*symbols)` | Derivative with respect to commutative symbols. |
| `table`, `npf`, `rem` | Raw data. |
| `PF0`, `PF1` | The zero and identity pattern forms. |
| `pf_add(*pfs)` | Sum of several `PF` objects. |

```python
>>> from cas_qwm import PF, pf
>>> pf(a**2*(cos(Dagger(a)*a) + Dagger(b)*sin(b*Dagger(b)))*Dagger(a)*b**2, as_dict=None)
PatternForm({((a, b), ()): (-1 + N_(b))*(1 + N_(a))*sin(-1 + N_(b)), ((a, BP(b; 2)), ()): (1 + N_(a))*cos(1 + N_(a))}, 0)

>>> x = symbols('x', real=True)
>>> p = PF.from_val(x*a + Dagger(a))
>>> (p**2).expr()
x**2*BP(a; 2) + x*(1 + N_(a)) + x*N_(a) + BP(D(a); 2)
>>> p.adjoint().expr()
x*Dagger(a) + a

>>> pf(sin(a), as_dict=None)       # no pattern form: everything goes to npf
PatternForm({}, sin(a))
```

`BSC` and `FSC` are the internal *sort contexts* used to merge sorted boson / fermion pattern sequences; they are exported for advanced use.

### `PE` — perturbative expander

```python
PE(assumptions)
```

Expands expressions perturbatively, given the order of each symbol or operator in the `assumptions` dict:

| Order | Meaning |
|---|---|
| `0` | $\sim 1$ (default for anything not listed) |
| `1` | $\ll 1$, adds one perturbative order |
| `-1` | $\gg 1$, removes one perturbative order |
| `oo` | exactly zero |

Non-integer orders (e.g. `-1/2`) are allowed. Functions are expanded as multivariate Taylor series, taking operator ordering into account.

| Method | Description |
|---|---|
| `expand_expr(expr, n)` | Generator of `(term, order)` pairs up to order `n`. |
| `leading_order(expr)` | `(leading_term, order)`. |
| `expand_pf(pf, n, with_order=False)` | Expansion of a `PF` object up to order `n`. |
| `lo_pf(pf)` | Leading-order part of a `PF` and its order. |
| `lo_order_pf(pf)` | Leading order of a `PF` only (`oo` if it vanishes), without expanding it. |
| `cohr_avr(expr, cohr_dict, n)` / `cohr_pf(pf, cohr_dict, n)` | Expectation value under coherent states `{mode: alpha}`, expanded up to order `n`. |
| `update(orders, clear=False)` | Add / change orders of symbols or modes (an order of 0 removes the entry); `clear=True` first drops all previous orders. Registered types are kept. |
| `assumptions` | Plain `dict` of orders; string keys (mode names) are sympified to `Symbol`s. |
| `register(type, category, ...)` | Teach the expander the order rule of a new expression type (`'c'` constant, `'s'` symbol, `'p'` pattern). |
| `register_algebra(alg, cancel=False)` | Hand the expressions recognized by a coefficient algebra over to it (see below). |
| `handle_order_limits(handle, n_args=1)` | Analyse the order constraints of a function handle. |
| `copy(cache=False)` | Copy of the expander. |

```python
>>> from cas_qwm import PE
>>> x, y = symbols('x y')
>>> pe = PE({x: 1})
>>> tuple(pe.expand_expr(sin(x + y), 2))
((sin(y), 0), (x*cos(y), 1), (-x**2*sin(y)/2, 2))
>>> pe.leading_order(x**2*sin(y) + x**3)
(x**2*sin(y), 2)
```

**Coefficient algebras.** Some coefficients are better expanded in their own canonical representation than as SymPy expressions. A registered algebra provides `match(expr) -> bool`, `leading_order(expr) -> (term, order)` and `expand(expr, n) -> iterable of (term, order)`. `leading_order`, `expand_expr` and therefore `lo_pf` / `expand_pf` hand every matching expression to it. The orders stay entirely in the expansion step: `NumShift` and the `PF` products shift the number operators exactly, and the algebra expands the shifted coefficients when `PE` truncates them. `EllipticAlgebra` (`cas_qwm.jacobi.elliptic_algebra`) is such an algebra; `TWMDualSolver` uses it for its elliptic coefficients.

### `perturb_rec`

```python
perturb_rec(_comm=None, _is_comm=None, _is_num=None, _add=Add, _mul=None, _pow=None)
```

Decorator factory unifying multivariable perturbative recursion and Taylor expansion for non-commutative arguments. It implements a branch-and-bound search over perturbation orders, builds the symmetrised operator permutations of each Taylor term, and caches derivatives. The keyword arguments customise commutation tests (`_comm`, `_is_comm`), scalar detection (`_is_num`) and the arithmetic used to assemble terms (`_add`, `_mul`, `_pow`), so that the same machinery works for `Expr`, `PF` and `WMWrapper` objects. It is used internally by `PE` and `WMProcessor.expand_func_d`.

---

## Layer 3 — `cas_qwm.parametric_multimode`

This layer applies the framework to multimode parametric wave mixing. Mode 0 is the pump (high frequency); modes 1 and 2 are signal and idler. Every weak field is multimode, with all mode pairs sharing the same nonlinear coupling $\chi$ and phase mismatch $\Delta k$. In three-wave mixing, the Heisenberg equations are

$$\dot a_0 = -i\chi\, a_1 a_2\, e^{-i\Delta k t},\qquad
\dot a_1 = -i\chi\, a_0 a_2^\dagger\, e^{i\Delta k t},\qquad
\dot a_2 = -i\chi\, a_1^\dagger a_0\, e^{i\Delta k t}.$$

In down-conversion (`dc=True`) the strong driving field is $a_0$; in frequency conversion / state transfer (`dc=False`) it is $a_2$.

### Shared symbols

| Name | Value |
|---|---|
| `a_p`, `a_s`, `a_i` | `BosonOp('a_0')`, `BosonOp('a_1')`, `BosonOp('a_2')` |
| `t_` | time, `Dummy('t', real=True)` |
| `nlc` | nonlinear coupling $\chi$ (real, non-negative) |
| `pm` | phase mismatch $\Delta k$ (real) |
| `nm` | number of mode pairs $n_m$ |
| `alpha_0`, `alpha_2` | coherent amplitudes of the strong driving field (down-conversion and four-wave mixing: $a_0$; state transfer: $a_2$) |
| `beta_0`, `beta_1`, `beta_2`, `betas` | coherent amplitudes of a seeded weak mode, `betas[j]` = `beta_j` |
| `gain`, `g_hat` | scalar parametric gain $g$ (real; also the integration variable of the g-domain solutions) and the gain operator $\hat g$ (non-commutative) used by the gain display |

### Multimode operators

```python
MultimodeCOP(name1, name2, parity=0, annih1=1)
MultimodeNum(name, displacement=0)
MultimodeNum2(name1, name2, parity=0)
```

| Class | Represents |
|---|---|
| `MultimodeCOP` | Multimode canonical operator $\sum_\text{modes} a_1^{(\dagger)} a_2^{(\dagger)}$; `parity` says whether the two operators differ in being annihilation / creation, `annih1` whether the left one is an annihilation operator. `swap_exps(exp1, exp2, ...)` yields the reordering of $[C^{e_1}, (C^\dagger)^{e_2}]$. |
| `MultimodeNum` | $\sum_\text{modes} (\hat N + d)$ with a shift `displacement` $d$ per mode. |
| `MultimodeNum2` | The commutator of a `MultimodeCOP` with its adjoint, e.g. $\sum_\text{modes}(1 + \hat N_1 + \hat N_2)$ for `parity=0`. |

```python
>>> from cas_qwm import MultimodeCOP, MultimodeNum, MultimodeNum2
>>> MultimodeCOP('a_1', 'a_2')
Σ(a_1*a_2)
>>> MultimodeCOP('a_1', 'a_2', 1)
Σ(a_1*D(a_2))
>>> MultimodeNum('a_1')
Σ(N_(a_1))
>>> MultimodeNum2('a_1', 'a_2')
Σ(1 + N_(a_1) + N_(a_2))
```

### `TWMSolver` / `FWMSolver`

```python
TWMSolver(dc=True, mismatch=False)
FWMSolver()
```

Perturbative solvers of the Heisenberg equations. A solver is a read-only sequence: `solver.get_sol(order)` returns the 3-tuple of `WMWrapper` objects `(a_0(t), a_1(t), a_2(t))` accurate up to `order`, each labeled with its actual accuracy order (`solver.sol_orders(order)`: in TWM the weak fields are corrected at even orders and the driving field at odd orders only, so a solution is accurate up to the order before its next correction), and `solver.sol_expr(mode, order)` returns the corresponding SymPy expression (including the phase factors of a mismatched process).

- **`TWMSolver`** — three-wave mixing driven by one strong field (pump for `dc=True`, idler for `dc=False`). The strong field is of order $-1$, the weak fields of order 0, and $\chi$ of order 1. Orders $\ge -1$ are solved recursively on demand and cached, so the first call of a high order can be slow. `mismatch=True` keeps a symbolic $\Delta k$.
- **`FWMSolver`** — four-wave mixing with a degenerate monochromatic pump, including self- and cross-phase modulation. The pump is of order $-1$, the weak fields of order 0, and $\chi$ of order 2, so the parametric gain $\chi a_0^2$ and the phase modulation $\chi \hat N_0$ are of order 0 and all orders are integers. Three solutions are computed at construction: the undepleted pump alone, with the weak fields, and with the first pump-depletion correction. In each one, every mode carries the highest order it applies to. The pump's first depletion correction is at order 1, so it applies up to order 0 in the first two solutions and up to order 2 in the third. The weak fields' first correction (order 2) is not computed, so they apply up to order 1 ($-1$ in the first solution, where they vanish). The order of a solution is the lowest of its modes, so the three solutions have orders $-1$, $0$ and $1$ (`FWMSolver.orders`). `get_sol(order)` returns the highest solution whose order does not exceed `order`, so any order above 1 gives the order-1 solution; orders below $-1$ raise `ValueError`. Products of them, and hence Tab 3 results, may thus be accurate to a lower order than the preset (e.g. 1 for $\hat N_0$ at preset order 2).

Common attributes: `processor` (the underlying `WMProcessor`), `dc`, `nd` (number operator of the driving field), `pe` (the `PE` used), `gc` (gain coefficient, $g = g_c\,t$).

```python
>>> from cas_qwm import TWMSolver, FWMSolver
>>> s = TWMSolver(dc=True)
>>> s.sol_expr(1, 0)        # signal at order 0 (undepleted pump)
cosh(_t*χ*sqrt(N_(a_0)))*a_1 - I*a_0*N_(a_0)**(-1/2)*sinh(_t*χ*sqrt(N_(a_0)))*Dagger(a_2)

>>> s.sol_expr(0, 1)        # pump up to the first depletion correction
a_0*(1 + Σ(1 + N_(a_1) + N_(a_2))*N_(a_0)**(-1)/4 - Σ(1 + N_(a_1) + N_(a_2))*N_(a_0)**(-1)*cosh(2*_t*χ*sqrt(N_(a_0)))/4) + Σ(D(a_1)*D(a_2))*BP(a_0; 2)*(-_t*I*χ*N_(a_0)**(-1)/2 + I*N_(a_0)**(-3/2)*sinh(2*_t*χ*sqrt(N_(a_0)))/4) + Σ(a_1*a_2)*(-_t*I*χ/2 - I*N_(a_0)**(-1/2)*sinh(2*_t*χ*sqrt(N_(a_0)))/4)

>>> FWMSolver().sol_expr(1, 0)
-2*I*χ*BP(a_0; 2)*(Δk*(-Δk + 4*χ*N_(a_0)))**(-1/2)*sinh(_t*sqrt(Δk*(-Δk + 4*χ*N_(a_0)))/2)*Dagger(a_2) + (I*(-Δk + 2*χ*N_(a_0))*(Δk*(-Δk + 4*χ*N_(a_0)))**(-1/2)*sinh(_t*sqrt(Δk*(-Δk + 4*χ*N_(a_0)))/2) + cosh(_t*sqrt(Δk*(-Δk + 4*χ*N_(a_0)))/2))*a_1
```

Module-level instances: `twm_processors` (`(WMProcessor(False), WMProcessor(True))`, indexed by `dc`) and `fwm_processor` (`WMProcessor(True, nlc_order=2)`).

### `WMWrapper`

```python
WMWrapper(processor, dict_repr=None, b_left=None, order=S.Infinity, truncate=True)
WMWrapper.from_val(val, processor)
```

Algebraic container for multimode wave-mixing expressions; the solver outputs are `WMWrapper` objects. Internally, an expression is stored as a dict of multimode exponent tuples to `PF` objects, optionally with one unpaired weak-field operator on the left (`b_left=True`) or right (`b_left=False`).

Every wrapper carries an accuracy `order`: the absolute perturbative order up to which it is accurate, `+oo` (`S.Infinity`, the default) for an exact expression, which is accurate to all orders. A product (`*` or `@`) of factors with leading orders $l_1, l_2$ and orders $o_1, o_2$ is accurate to $l_1 + l_2 + \min(o_1 - l_1,\, o_2 - l_2)$ (an exact factor counting as $+\infty$ relative accuracy); the factor more accurate than needed is truncated beforehand, and every partial product is truncated at that order inside `WMProcessor.d_mul`. Sums take the lowest order of their terms (the more accurate terms are truncated there), `expand(n)` and `eval_coherent(n)` lower it to `n`, `expand_func_d` keeps the relative accuracy of its arguments, and the other transformations keep it.

A finite `order` given to the constructor truncates the content at that order (`truncate=False` skips this for content known to have no terms beyond it), and `with_order(order)` truncates when it lowers the order. The solver outputs are therefore cut at their accuracy labels, and a saved variable bound with a lower recorded order loses the terms it cannot support.

| Member | Description |
|---|---|
| `@` | **Multimode product**: paired weak-field operators are summed over all modes, e.g. `dag(a_1) @ a_1` → $\sum_\text{modes}\hat N_1$. |
| `*` | **Single-mode product**: paired weak-field operators belong to the same mode, e.g. $a_1^\dagger a_1 \to \hat N_1$. |
| `+`, `-`, `/` by scalars, `**` | Usual algebra. |
| `adjoint()` | Hermitian adjoint. |
| `expr` | Conversion to a SymPy expression. |
| `order`, `lo`, `rel_order`, `is_exact` | Accuracy order (`+oo` if exact), leading order (`+oo` if zero), their difference, and whether the order is `+oo`. |
| `with_order(order, truncate=True)` | Copy with another accuracy order, truncated there when the order is lowered. |
| `expand(order, with_order=False)` | Truncate / expand to a perturbative order; with `with_order=True`, yields `(term, order)` pairs. |
| `eval_vacuum(vac_modes)` | (Partial) expectation value with the given weak modes (`'a_1'`, `'a_2'`, …) in vacuum. |
| `eval_coherent(order=0)` | Expectation value with the driving field in a coherent state $|\alpha\rangle$ (`alpha_0` or `alpha_2`). |
| `seeded_separate(mode=1, vac_c=True, observed=False)` | (Partial) expectation value with one separate weak mode $b$ seeded in the coherent state $|\beta_j\rangle$ (the global `beta_j`, $j$ the mode index), coupled with its partner in vacuum (`vac_c`) or in its own coherent state. `mode`: down-conversion `1` seeds the signal (coupled with the idler) and `0` a degenerate mode (signal = idler, coupled with itself; `vac_c` has no effect); state transfer: the index of the seeded mode (`1` signal, `0` pump); four-wave mixing: no effect (signal). The seeded modes are of order 0, so the average is exact. Every multimode operator is split into its $b$ part and the rest $\Sigma'$, which keeps the symbols and $n_m$ of the full sum. COP powers are expanded binomially, $(B + \Sigma')^e = \sum_j \binom{e}{j} B^j \Sigma'^{\,e-j}$, and the tails become polynomials in $\hat N_b$. A degenerate $b$ part follows the degenerate map of `to_single_mode`. The $b$ operators of each term are then normal-ordered ($b^n b^{\dagger e} = \sum_k \binom{n}{k}\binom{e}{k} k!\, b^{\dagger e-k} b^{n-k}$) and averaged, giving polynomials in $\beta$ and $\bar\beta$. With `observed=True`, the single-mode weak-field operators (key and single-mode patterns) are the seeded modes and are averaged too; a degenerate seed can only be observed through the key. After `to_single_mode()` there is no sum left, so the weak modes themselves are seeded. |
| `to_single_mode(degenerate=False)` | Replace multimode operators by single-mode ones ($n_m = 1$). With `degenerate=True` (dc only), map onto degenerate down-conversion (signal = idler = $a_1$, $H = \chi(a_0^\dagger a_1^2 + a_0 a_1^{\dagger 2})/2$) via $\sum a_1 a_2 \to a_1^2/2$, $\sum(1 + \hat N_1 + \hat N_2) \to \hat N_1 + 1/2$, $\sum(\hat N_j + d) \to (\hat N_1 + d)/2$, $n_m \to 1/2$, $a_2 \to a_1$. This map preserves all commutators the multimode solutions use, so it is exact for the solver outputs and for `@` products (a multimode sum $\sum \hat N_j$ becomes $\hat N_1/2$). Take single-mode products (`*`) of weak-field operators after the conversion. |
| `diff(*symbols, gc=None)` | Derivative (w.r.t. time or gain). |
| `apply(f, on_npf=False)` | Apply `f` to every number-operator coefficient. |
| `flip()`, `as_scalar`, `parity`, `dc` | Internal representation helpers. |

```python
>>> s = TWMSolver(dc=True)
>>> a0, a1, a2 = s.get_sol(0)
>>> n1 = a1.adjoint() @ a1                     # total signal photon number at time t
>>> n1.eval_vacuum(['a_1', 'a_2']).expr        # signal and idler in vacuum
n_m*N_(a_0)*sinh(_t*χ*sqrt(N_(a_0)))*N_(a_0)**(-1)*sinh(_t*χ*sqrt(N_(a_0)))

>>> a0, a1, a2 = s.get_sol(1)
>>> n0 = (a0.adjoint() @ a0).eval_vacuum(['a_1', 'a_2'])
>>> n0.eval_coherent(1).expr                   # pump photon number, coherent pump, order 1
-n_m*cosh(2*_t*χ*Abs(alpha_0))/2 + n_m/2 + Abs(alpha_0)**2
```

The last result is the pump depletion $\langle \hat N_0(t)\rangle = |\alpha_0|^2 - n_m \sinh^2(\chi t |\alpha_0|)$.

### `WMProcessor`

```python
WMProcessor(dc=True, nlc_order=1)
```

Base processor shared by the solvers and wrappers of one process: it stores the mode structure, the perturbative expander (`pe`), and the low-level dict algebra. The strong driving field is of order $-1$ and the weak fields of order 0; `nlc_order` is the order of $\chi$ (1 for three-wave mixing, 2 for four-wave mixing, so that the parametric gain is of order 0).

| Member | Description |
|---|---|
| `init_ops` | Input-operator wrappers at $t = 0$: `'a_0'`, `'a_1'`, `'a_2'` (and aliases `'a_p'`, `'a_s'`, `'a_i'`), the multimode canonical operator `'A'` and the multimode number operator `'N_t'`. |
| `expand_func_d(func, order, *ds, with_order=False)` | Perturbative expansion of `func(*ds)` with `WMWrapper` arguments (without arguments, of the constant `func()` by `PE.expand_expr`), accurate to $\min(\text{order},\ l_f + \min_j(o_j - l_j))$ with $l_f$ the leading order of `func` and $o_j, l_j$ the accuracy and leading orders of the inexact arguments. |
| `p2s(expr, xs=(t_,), simp=None)` | Product-to-sum transformation of hyperbolic (dc) or trigonometric (fc) functions of time, grouping terms by time dependence. |
| `d_add(*ws)` | Sum of several wrappers, accurate to the lowest order of its terms (vanishing terms included). |
| `d_mul(d0, d1, d2, mid=None, sc=False, order=S.Infinity)` | Accumulates the products of two `{tuple: PF}` dicts (with an optional middle operator `mid`) into `d0`; with a finite `order`, pairs whose leading orders already exceed it are skipped and every product is truncated there. |
| `bd_swap`, `bp_swap`, `bt_swap`, `tp_swap`, `tcp_swap` | Commutation of single-mode operators, `PF` objects and multimode exponent tuples. |

```python
>>> from sympy import sqrt
>>> from cas_qwm import WMWrapper, BosonNum
>>> p = s.processor
>>> w = WMWrapper.from_val(BosonNum('a_0'), p)
>>> p.expand_func_d(sqrt, 1, w + 1).expr       # sqrt(N_0 + 1) up to order 1
N_(a_0)**(-1/2)/2 + sqrt(N_(a_0))
```

The module-level function `vac_pf(p, vac_mode)` evaluates the vacuum projection of a `PF` for one or several modes:

```python
>>> from cas_qwm import vac_pf, pf, BosonNum
>>> vac_pf(pf((BosonNum('a_1') + 1)**2, as_dict=None), 'a_1').expr()
1
```

### Differential-equation helpers

```python
p2s(expr, x, h=True, subs=None)
dsolve_pm1(rhs, x, h=True, f0=0, subs=None)
dsolve_pm2(rhs, x, h=True, f0=0, df0=0, subs=None)
```

Closed-form solvers for the linear ODEs that appear in the order-by-order solution, with sources of the form $\sum x^{\text{deg}}\, F(n_k x)$, $F \in \{\cos, \sin, \cosh, \sinh\}$. The flag `h` selects hyperbolic (`True`) or trigonometric (`False`) functions.

| Function | Solves |
|---|---|
| `p2s` | Product-to-sum rewriting of the (hyperbolic) trigonometric functions of `x`. |
| `dsolve_pm1` | $f'(x) = \text{rhs}$, $f(0) = f_0$. |
| `dsolve_pm2` | $f''(x) \mp f(x) = \text{rhs}$ (`−` for `h=True`, `+` for `h=False`), $f(0) = f_0$, $f'(0) = f'_0$. |

```python
>>> from cas_qwm import p2s, dsolve_pm1, dsolve_pm2
>>> from sympy import symbols, sinh, cosh
>>> x = symbols('x')
>>> p2s(sinh(x)*cosh(x), x)
sinh(2*x)/2
>>> dsolve_pm1(x*cosh(2*x), x)
x*sinh(2*x)/2 - cosh(2*x)/4 + 1/4
>>> dsolve_pm2(sinh(2*x), x, True, 0, 0)       # f'' - f = sinh(2x), f(0) = f'(0) = 0
-2*sinh(x)/3 + sinh(2*x)/3
```

---

## Two strong inputs — `cas_qwm.twm_dual_stimulated`

Single-mode three-wave mixing $H = \hbar\chi(a_1 a_2 a_0^\dagger + a_1^\dagger a_2^\dagger a_0)$ in which **two inputs are strong** and one is weak: sum-frequency generation from a strong signal and idler (weak pump, `weak='p'`), or difference-frequency generation from a strong pump and one strong lower-frequency field (`weak='s'` or `weak='i'`). The leading order is the classical Jacobi elliptic solution; the solver computes the quantum corrections recursively, to any order. The derivation, the check of the first-order formulas of the original notes and the corrections to them are in [docs/dual_stimulated_twm.md](docs/dual_stimulated_twm.md).

### Method

1. **Conservation laws.** $M_s = \hat N_1 + \hat N_0$, $M_i = \hat N_2 + \hat N_0$ and $H' = H/\hbar\chi$ are conserved. Using $\hat N_0(t) = M_s - a_1^\dagger(t) a_1(t)$ etc., every field obeys a **closed** second-order Heisenberg equation:

$$\ddot a_1 = \chi^2\big[(2\hat N_1 + \hat N_0 - \hat N_2)\,a_1 - 2\,a_1^\dagger a_1 a_1\big](t),\qquad
\ddot a_2 = \chi^2\big[(2\hat N_2 + \hat N_0 - \hat N_1)\,a_2 - 2\,a_2^\dagger a_2 a_2\big](t),$$
$$\ddot a_0 = \chi^2\big[-(\hat N_1 + \hat N_2 + 2\hat N_0 + 1)\,a_0 + 2\,a_0^\dagger a_0 a_0\big](t),$$

   where the linear coefficients are conserved and stand on the left of the field.
2. **Forward recursion.** The strong fields are of order $-1$, the weak one of order 0, and $\chi$ of order 1. The leading order (order $-1$) is the classical solution; order $k$ solves the variational equation with a source given by the order-$k$ residual of the lower orders. Operator products are evaluated with `PF`: the coefficients are `EllipticTail` functions of the number operators, which `NumShift` shifts exactly, so the commutator (ordering) corrections enter automatically. `PE` truncates the products by orders, with `EllipticAlgebra` registered as a coefficient algebra (both in `cas_qwm.jacobi.elliptic_algebra`). The coupling term $g^\dagger a_0 a_0$ of the variational equation is the antilinear involution $J[X] = T^2X^\dagger/N_T$ ($T$ the leading key, $N_T = T^\dagger T$ at leading order), which a `PF` product evaluates: it maps the key $K$ to $T^2K^\dagger$, conjugates the coefficient and applies the exact weak-number shift. $P = g + J[g]$ and $Q = g - J[g]$ then obey uncoupled equations, solved once per pair $\{K, J(K)\}$ at a representative key: $g_K = (P + Q)_K/2$, $g_{J(K)} = J[(P - Q)_K/2]$. These are **Lamé equations of degree 1 and 2**, $y'' = (n(n+1)m\,\mathrm{sn}^2 - h)y$. Only six of them occur, and their fundamental solutions are tabulated in closed form (one is a Jacobi function or a derivative of one; the other contains $u$ and $E$):

| $(n, h)$ | $y_1$ | $y_2$ |
|---|---|---|
| $(1, 1)$ | $\mathrm{cn}$ | $[\mathrm{cn}((1-m)u - E) + \mathrm{sn}\,\mathrm{dn}]/(1-m)$ |
| $(1, m)$ | $\mathrm{dn}$ | $(E\,\mathrm{dn} - m\,\mathrm{sn}\,\mathrm{cn})/(1-m)$ |
| $(1, 1+m)$ | $\mathrm{cn}\,\mathrm{dn} - \mathrm{sn}(u - E)$ | $\mathrm{sn}$ |
| $(2, 1+m)$ | $\mathrm{cn}\,\mathrm{dn}$ | $[(1-m)u\,\mathrm{cn}\,\mathrm{dn} - (1+m)E\,\mathrm{cn}\,\mathrm{dn} + (1+m^2)\mathrm{sn} - m(1+m)\mathrm{sn}^3]/(1-m)^2$ |
| $(2, 1+4m)$ | $\mathrm{cn} - u\,\mathrm{sn}\,\mathrm{dn} + [(2m-1)E\,\mathrm{sn}\,\mathrm{dn} + m(1-2m)\mathrm{cn}\,\mathrm{sn}^2]/(m-1)$ | $\mathrm{sn}\,\mathrm{dn}$ |
| $(2, 4+m)$ | $\mathrm{dn} - 2u\,\mathrm{sn}\,\mathrm{cn} + [(m-2)E\,\mathrm{sn}\,\mathrm{cn} + (2-m)\mathrm{dn}\,\mathrm{sn}^2]/(m-1)$ | $\mathrm{sn}\,\mathrm{cn}$ |

   They are normalized to unit Wronskian, $y_1(0) = y_2'(0) = 1$, $y_1'(0) = y_2(0) = 0$. The particular solution $y_2\int_0^u y_1 S - y_1\int_0^u y_2 S$ is integrated in closed form by recursion, as in `dsolve_pm` (module `cas_qwm.jacobi.dsolve_lame`). The integrands are polynomials in $(u, E)$ times $\mathrm{sn}^{2k}$ or $\mathrm{sn}^{2k+1}\mathrm{cn}\,\mathrm{dn}$. These are reduced by the recurrence of $\int\mathrm{sn}^{2k}$ and by integration by parts. Every coefficient lies in the ring $\mathbb Q(m)[n, u, E(u), \mathrm{sn}, \mathrm{cn}, \mathrm{dn}]$, where $E$ is Jacobi's epsilon function. No numerical step is involved.
3. **Variables.** The weak number $n = \hat N_w$ and two conserved strong quantities $L_X, L_Y$ are used:

| `weak` | $L_X$ | $L_Y$ | parameter $m$ | strong fields at leading order |
|---|---|---|---|---|
| `'p'` | $\hat N_1 + \hat N_0$ | $\hat N_2 + \hat N_0$ | $L_X/L_Y$ | $a_1 \to \mathrm{cn}\,a_1$, $a_2 \to \mathrm{dn}\,a_2$, $a_0 \to -i\lambda\,\mathrm{sn}\,a_1a_2$ |
| `'s'` | $\hat N_0 + \hat N_1$ | $\hat N_2 - \hat N_1$ | $-L_X/L_Y$ | $a_0 \to \mathrm{cn}\,a_0$, $a_2 \to \mathrm{dn}\,a_2$, $a_1 \to -i\lambda\,\mathrm{sn}\,a_0a_2^\dagger$ |
| `'i'` | $\hat N_0 + \hat N_2$ | $\hat N_1 - \hat N_2$ | $-L_X/L_Y$ | $a_0 \to \mathrm{cn}\,a_0$, $a_1 \to \mathrm{dn}\,a_1$, $a_2 \to -i\lambda\,\mathrm{sn}\,a_0a_1^\dagger$ |

   with $u = \chi t\sqrt{L_Y}$, $\lambda = L_Y^{-1/2}$ and the Jacobi functions of $(u|m)$. A negative parameter is the imaginary-modulus form of the $\mathrm{sd}, \mathrm{cd}, \mathrm{nd}$ functions of the notes. A shifted coefficient is expanded by the registered algebra: exactly in $n$, and as a Taylor series in $\lambda^2$ on $(u, m, \lambda)$, using the closed-form $m$-derivatives of the Jacobi functions.
4. **No phase mismatch.** A mismatch $\Delta k \sim \chi\sqrt L$ changes the leading orbit (Armstrong et al., Phys. Rev. 127, 1918, Secs. V–VI): the modulus is set by the roots of the mismatched cubic, and the phase is an elliptic integral of the third kind, which lies outside the ring above. It is therefore not included.

The solution is returned in **pattern form**: a sum of canonical-operator keys with number-operator coefficients on the right. This is the representation used by `PF`. For $a_1$ with a weak pump, the keys are $(a_0^\dagger a_1 a_2)^r a_1$ and $(a_0 a_1^\dagger a_2^\dagger)^{r-1} a_2^\dagger a_0$, i.e. the harmonic decomposition of the notes' standard form $U a_1 + V a_2^\dagger a_0$.

### `TWMDualSolver`

```python
TWMDualSolver(weak='p')
```

| Member | Description |
|---|---|
| `get_sol(target, order)` | List of the `PF` terms of orders $-1, \dots,$ `order` of mode `target` (`0`/`'p'`, `1`/`'s'`, `2`/`'i'`); their coefficients are `EllipticTail`s of `alg`. Computed recursively and cached. |
| `sol_pf(target, order)` | The solution accurate up to `order` as one `PF`. It can be multiplied with other `PF`s and truncated with `pe.expand_pf`. |
| `sol_expr(target, order, explicit=True)` | SymPy expression accurate up to `order`. `explicit=True`: in number operators and $t$ (Jacobi functions of operator arguments); `explicit=False`: with the symbols `u`, `m`, `λ`, `n`, `sn`, `cn`, `dn`, `E`. |
| `apply(target, order, fock, chi_t)` | Numerical action on $|n_0, n_1, n_2\rangle$ at $\chi t$ = `chi_t`; returns `{(n_0, n_1, n_2): amplitude}`. |
| `symbols()` | Expressions of $L_X, L_Y, m, n, u$ in number operators. |
| `residual(target, A, cap)` | Residual of the closed equation (in $u$ units) of a `PF` `A`, expanded up to order `cap`. |
| `alg`, `pe` | The `EllipticAlgebra` of the coefficients, and the `PE` instance (strong modes of order $-1$, `alg` registered) that counts and truncates the orders. |

```python
>>> P = s.sol_pf('s', 1)
>>> n_s = s.pe.expand_pf(P.adjoint() * P, 0)   # N_s(t) up to order 0, as a PF
```

```python
>>> from cas_qwm import TWMDualSolver
>>> s = TWMDualSolver('p')
>>> s.sol_expr('s', 0, explicit=False)     # signal, accurate up to the first correction
cn*a_1 + λ**3*(-I*E*cn/(2*m**2 - 2*m) - I*cn*u/(2*m) + I*dn*sn/(2*m - 2))*Dagger(a_0)*BP(a_1; 2)*a_2 + λ*(-I*E*cn/(2*m - 2) - I*cn*u/2 + I*dn*sn*(2 - m)/(2*m - 2))*a_0*Dagger(a_2)
```

Solving order 2 takes about 2 s per field, and order 3 about 10 s.

### `exact_fock_apply`

```python
exact_fock_apply(target, fock, chi_t)
```

Exact single-mode reference, $a_\text{target}(t)|n_0, n_1, n_2\rangle$ as `{(n_0, n_1, n_2): amplitude}`. In a conserved block $(M_s, M_i)$, $H'$ is a tridiagonal matrix in $n_0$. The state is propagated there by a Chebyshev expansion of $e^{\mp i\chi H' t}$ (Jacobi–Anger series, Bessel coefficients from Miller's recurrence). It is pure Python in double precision and needs no NumPy; $N = 3200$ takes a few seconds.

```python
>>> from cas_qwm import exact_fock_apply
>>> ex = exact_fock_apply('s', (1, 400, 800), 2.0 / 801**0.5)
>>> ap = s.apply('s', 2, (1, 400, 800), 2.0 / 801**0.5)
```

### Accuracy

Each order gains a factor $L^{-1/2}$. Typical errors $\|(a^\text{exact} - a^\text{pert})|n\rangle\|$ for `weak='p'`, target $a_1$, $u = 3$, $n_0 = 2$, $(\hat N_1, \hat N_2) = (N/2, N)$:

| order | $N = 200$ | $N = 800$ | $N = 3200$ |
|---|---|---|---|
| $-1$ (classical) | 2.1 | 2.2 | 2.2 |
| 0 | 1.0e-1 | 3.8e-2 | 1.7e-2 |
| 1 | 8.1e-2 | 2.1e-2 | 5.3e-3 |
| 2 | 5.6e-3 | 5.8e-4 | 6.8e-5 |

All nine (weak mode, target) combinations were checked up to order 2 (weak pump up to order 3), at times beyond the first zero of $\mathrm{cn}$.

### Jacobi elliptic functions — `cas_qwm.jacobi`

`jacobi_sn(u, m)`, `jacobi_cn(u, m)`, `jacobi_dn(u, m)` and Jacobi's epsilon function `jacobi_epsilon(u, m)` $= E(u|m) = \int_0^u \mathrm{dn}^2(v|m)\,dv$. All use the parameter convention $m = k^2$, as in mpmath and DLMF §22. They evaluate numerically through `evalf` / `N` for any real or complex arguments, including $m < 0$ and $m > 1$. The derivatives with respect to both arguments are in closed form, so the ring generated by $u, E, \mathrm{sn}, \mathrm{cn}, \mathrm{dn}$ over $\mathbb Q(m)$ is closed under $\partial_u$ and $\partial_m$:

$$\partial_m \mathrm{sn} = \mathrm{cn}\,\mathrm{dn}\,X + \frac{\mathrm{sn}\,\mathrm{cn}^2}{2(1-m)},\quad
\partial_m \mathrm{cn} = -\mathrm{sn}\,\Phi,\quad
\partial_m \mathrm{dn} = -\frac{\mathrm{sn}^2\mathrm{dn}}{2(1-m)} - m\,\mathrm{sn}\,\mathrm{cn}\,X,\quad
\partial_m E = \frac{E - u}{2m} + \mathrm{dn}\,\Phi,$$

with $X = \dfrac{(1-m)u - E}{2m(1-m)}$ and $\Phi = \partial_m \mathrm{am} = \dfrac{u\,\mathrm{dn}}{2m} - \dfrac{E\,\mathrm{dn}}{2m(1-m)} + \dfrac{\mathrm{sn}\,\mathrm{cn}}{2(1-m)}$.

```python
>>> from sympy import Rational, N
>>> from cas_qwm import jacobi_sn, jacobi_epsilon
>>> N(jacobi_sn(Rational(23, 10), Rational(37, 100)))
0.905451503535721
>>> N(jacobi_epsilon(Rational(23, 10), -Rational(1, 2)))
2.94782630151328
```

### Elliptic-polynomial algebra — `cas_qwm.jacobi.dsolve_lame`

The ring algebra behind `TWMDualSolver`, independent of the operator layer. Its elements are SymPy `PolyElement`s of the ring `R` $= \mathbb Q(m)[n, u, E, s, c, d, j]$ ($s, c, d = \mathrm{sn}, \mathrm{cn}, \mathrm{dn}(u|m)$, $j^2 = -1$), kept reduced modulo $c^2 = 1 - s^2$, $d^2 = 1 - m s^2$.

| Name | Description |
|---|---|
| `m`, `R`, `M` | The parameter symbol, the ring (generators `R.gens` = $n, u, E, s, c, d, j$; `R.one`, `R.zero`) and $m$ as a coefficient. |
| `red(f)`, `mul(f, g)` | Reduction of $c^2, d^2, j^2$; reduced product. |
| `diff_u(f)`, `diff_m(f)`, `diff_um(f, a, b)` | $\partial_u$, $\partial_m$ (closed form, see above), and the cached $\partial_u^a\partial_m^b$. |
| `conj(f)`, `at0(f)`, `key(f)` | Complex conjugate ($j \to -j$), value at $u = 0$, hashable key. |
| `antideriv(f)`, `integrate0(f)` | Antiderivative in $u$ and $\int_0^u$, by the recurrence of $\int\mathrm{sn}^{2k}$ and integration by parts; `ObstructionError` if it leaves the ring (e.g. $\int E\,du$). |
| `LAME`, `lame(V)` | Closed-form fundamental solutions of the six Lamé equations above, keyed by $(\nu, h)$; lookup by the potential $V = \nu(\nu+1)m s^2 - h$. |
| `solve_lame(V, S, y0, dy0)` | Solution of $y'' - V y = S$, $y(0) = y_0$, $y'(0) = y'_0$, by variation of constants. |
| `evaluate(f, m, values)` | Numerical value at the parameter `m` and the generator values $(n, u, E, s, c, d, j)$ (e.g. mpmath numbers). |

```python
>>> from cas_qwm.jacobi.dsolve_lame import R, M, solve_lame
>>> n, u, E, s, c, d, j = R.gens
>>> y = solve_lame(2*M*s**2 - 1, c, 0, 0)       # y'' = (2m sn² - 1) y + cn, y(0) = y'(0) = 0
>>> y.as_expr()
E**2*c/(2*m**2 - 2*m) + E*c*u/m - E*d*s/(m**2 - m) + c*s**2/(2*m - 2) + c*u**2*(m - 1)/(2*m) - d*s*u/m
```

### Elliptic PF coefficients — `cas_qwm.jacobi.elliptic_algebra`

Functions of three number operators $(N_1, N_2, N_3)$ in the variables of the ring: $L_X = \sum_k x_k N_k$, $L_Y = \sum_k y_k N_k$, $n = \sum_k z_k N_k$, with $m = \varepsilon L_X/L_Y$, $\lambda = L_Y^{-1/2}$ and $u = \chi t\sqrt{L_Y}$. Such a function is an **LSeries** `{p: f_p}` $= \sum_p \lambda^p f_p(u, m, n)$ with $f_p$ in the ring of `dsolve_lame`.

```python
EllipticAlgebra(nops, lx, ly, ln, eps=1)
```

| Member | Description |
|---|---|
| `N` | The number operators `nops` as LSeries. |
| `tail(f)` | The `PF` coefficient of an LSeries: `EllipticTail(id, N_1, N_2, N_3)`, a SymPy function of the number operators. `NumShift` (and therefore every `PF` product) shifts its arguments exactly, e.g. to `F3(N_0, -1 + N_1, N_2)`; its adjoint conjugates the series. |
| `match`, `leading_order`, `expand` | The `PE` coefficient-algebra interface (see `PE`): polynomials in `nops` and the tails of this algebra are expanded here. A shifted tail is expanded exactly in $n$ and as a Taylor series in $\lambda^2$ on $(u, m, \lambda)$, using the closed-form derivatives of the Jacobi functions. |
| `to_ls(expr, maxp)` | LSeries of such a polynomial, truncated at $\lambda^{\text{maxp}}$. |
| `shift(f, dN, maxp)` | $f(N + \Delta N)$, truncated at $\lambda^{\text{maxp}}$. |
| `symbols(chi_t=None)` | $L_X, L_Y, m, n$ (and $u$) in the number operators. |
| `to_expr(f, chi_t=None)` | SymPy expression of an LSeries: in number operators and Jacobi functions if `chi_t` is given, otherwise in the symbols `λ`, `m`, `n`, `u`, `sn`, `cn`, `dn`, `E`. |
| `evaluator(nvals, chi_t)` | Function giving the numerical value of an LSeries at given numbers of quanta. |

The module also provides the LSeries operations `ls_add`, `ls_mul` (truncated), `ls_scale` and `ls_map`.

```python
>>> from cas_qwm import PE, PF, a_s
>>> from cas_qwm.op_pattern import BosonNum
>>> from cas_qwm.jacobi.dsolve_lame import R
>>> from cas_qwm.jacobi.elliptic_algebra import EllipticAlgebra
>>> n, u, E, s, c, d, j = R.gens
>>> nops = [BosonNum(x) for x in ('a_0', 'a_1', 'a_2')]
>>> alg = EllipticAlgebra(nops, (1, 1, 0), (1, 0, 1), (1, 0, 0))   # weak pump
>>> pe = PE({'a_1': -1, 'a_2': -1}); pe.register_algebra(alg)
>>> A = PF({k: alg.tail({0: c}) for k in PF.from_val(a_s).table})  # a_1 cn(u|m)
>>> (A * A).table                                                  # exact shift (ids vary)
{((BP(a_1; 2),), ()): F0(N_(a_0), -1 + N_(a_1), N_(a_2))*F0(N_(a_0), N_(a_1), N_(a_2))}
>>> (t,) = pe.expand_pf(A * A, 0).table.values()                    # expanded up to order 0
>>> alg.to_expr(alg.to_ls(t, 2))
-sn**2 + λ**2*(E*cn*dn*sn/(2*m**2 - 2*m) + cn*dn*sn*u/(2*m) + sn**4/(2*m - 2) - sn**2/(2*m - 2)) + 1
```

---

## Exported names

Every name below can be imported directly from `cas_qwm`.

| Module | Names |
|---|---|
| `op_pattern` | `BosonNum`, `BosonPat`, `NumShift`, `apply_num_shift`, `FermionPat`, `UidCommutator`, `extract_modes`, `check_num`, `in_boson_num`, `mul_pattern`, `redo_mul` |
| `pattern_perturb` | `BSC`, `FSC`, `PF`, `pf`, `pf_add`, `PF0`, `PF1`, `PE`, `perturb_rec` |
| `parametric_multimode` | `a_p`, `a_s`, `a_i`, `t_`, `nlc`, `pm`, `nm`, `gain`, `g_hat`, `p2s`, `dsolve_pm1`, `dsolve_pm2`, `MultimodeCOP`, `MultimodeNum`, `MultimodeNum2`, `WMProcessor`, `twm_processors`, `fwm_processor`, `vac_pf`, `WMWrapper`, `WMSolver`, `TWMSolver`, `FWMSolver` |
| `jacobi` | `jacobi_sn`, `jacobi_cn`, `jacobi_dn`, `jacobi_epsilon` |
| `twm_dual_stimulated` | `TWMDualSolver`, `exact_fock_apply` |

Further functions (e.g. `pattern_form`, `is_pattern_form`, `bn_simp` in `cas_qwm.pattern_perturb.pattern_form`) can be imported from their submodules.
