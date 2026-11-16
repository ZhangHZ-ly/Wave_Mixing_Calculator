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
| `_g` | dummy parametric gain $g$ |
| `alpha_0`, `alpha_2` | coherent amplitudes of the strong driving field (down-conversion and four-wave mixing: $a_0$; state transfer: $a_2$) |
| `beta_0`, `beta_1`, `beta_2`, `betas` | coherent amplitudes of a seeded weak mode, `betas[j]` = `beta_j` |
| `gain`, `g_hat` | scalar parametric gain $g$ (real) and the gain operator $\hat g$ (non-commutative) used by the gain display |

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

## Exported names

Every name below can be imported directly from `cas_qwm`.

| Module | Names |
|---|---|
| `op_pattern` | `BosonNum`, `BosonPat`, `NumShift`, `apply_num_shift`, `FermionPat`, `UidCommutator`, `extract_modes`, `check_num`, `in_boson_num`, `mul_pattern`, `redo_mul` |
| `pattern_perturb` | `BSC`, `FSC`, `PF`, `pf`, `pf_add`, `PF0`, `PF1`, `PE`, `perturb_rec` |
| `parametric_multimode` | `a_p`, `a_s`, `a_i`, `t_`, `nlc`, `pm`, `nm`, `_g`, `p2s`, `dsolve_pm1`, `dsolve_pm2`, `MultimodeCOP`, `MultimodeNum`, `MultimodeNum2`, `WMProcessor`, `twm_processors`, `fwm_processor`, `vac_pf`, `WMWrapper`, `WMSolver`, `TWMSolver`, `FWMSolver` |

Further functions (e.g. `pattern_form`, `is_pattern_form`, `bn_simp` in `cas_qwm.pattern_perturb.pattern_form`, and the Jacobi elliptic functions in `cas_qwm.jacobi`) can be imported from their submodules.
