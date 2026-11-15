# Quantum Operator Algebra Calculator

A browser-based calculator for symbolic quantum-optical operator algebra, built on top of **`cas_qwm`** (*Computer Algebra System for Quantum Wave Mixing*), a SymPy-based library developed in this repository.

The calculator lets you:

- reorder products of bosonic / fermionic operators of a single mode (pattern form, number-operator form, normal order);
- process arbitrary multi-mode quantum expressions into a unique **pattern form** (canonical operators on the left, functions of number operators on the right), optionally keeping the commutator remainder between non-independent modes;
- compute the **Heisenberg-picture evolution** of operator expressions in multimode parametric **wave-mixing** processes (three-wave down-conversion, three-wave frequency conversion / state transfer, four-wave mixing) as a perturbative series in the pump depletion, with or without phase mismatch, and evaluate them under input-state assertions (coherent pump, vacuum signal / idler).

All results are rendered with KaTeX and can be copied as LaTeX.

> **Python API:** the full programming interface of the `cas_qwm` library is documented separately in **[cas_qwm_documentation.md](cas_qwm_documentation.md)**.

---

## Contents

1. [Project structure](#project-structure)
2. [Installation](#installation)
3. [Running the calculator](#running-the-calculator)
4. [Web interface guide](#web-interface-guide)
   - [Expression syntax (shared by all tabs)](#expression-syntax-shared-by-all-tabs)
   - [Tab 1 — Single Mode](#tab-1--single-mode)
   - [Tab 2 — Quantum Expression](#tab-2--quantum-expression)
   - [Tab 3 — WM Evolution](#tab-3--wm-evolution)
   - [The variable tables](#the-variable-tables)
5. [HTTP API (for scripting)](#http-api-for-scripting)
6. [Troubleshooting](#troubleshooting)

---

## Project structure

```
Calc/
├── cas_qwm/                 # Core symbolic library (see cas_qwm_documentation.md)
│   ├── op_pattern/          #   Layer 1: operator patterns (BosonPat, FermionPat, BosonNum, ...)
│   ├── pattern_perturb/     #   Layer 2: pattern form (PF, pf) and perturbative expansion (PE)
│   ├── parametric_multimode/#   Layer 3: multimode wave-mixing processors and solvers
│   ├── dsolve_pm.py         #   ODE solvers for polynomial × (hyperbolic) trigonometric sources
│   └── wm_variables.py      #   Shared symbols: a_0, a_1, a_2, t, χ
├── calculator/              # Backend glue between the web UI and cas_qwm (parsing, presets, assertions)
├── static/                  # Front-end assets (CSS, JavaScript; optional local KaTeX)
├── index.html               # Single-page web interface
├── main.py                  # FastAPI server
├── setup_offline.py         # Downloads KaTeX (and optionally pip wheels) for offline use
└── requirements.txt         # Pinned Python dependencies
```

---

## Installation

### Requirements

- **Python ≥ 3.10** (developed and tested with Python 3.12)
- A modern web browser (Chrome, Edge, Firefox, Safari)

The Python dependencies are SymPy (`sympy==1.14.0`, with `mpmath`) for the symbolic engine, and FastAPI + Uvicorn for the web server. Exact versions are pinned in [requirements.txt](requirements.txt).

### 1. Get the code

```bash
git clone https://github.com/ZhangHZ-ly/Wave_Mixing_Calculator.git
cd Wave_Mixing_Calculator
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

If you only want to use the `cas_qwm` library from Python (without the web interface), `pip install sympy==1.14.0` is sufficient; then import it from the repository root (or add the repository root to `PYTHONPATH`).

### 4. (Optional) Offline setup

By default the page loads KaTeX (the math renderer) from the jsDelivr CDN. To run the calculator on a machine without internet access, download KaTeX into `static/katex/` once while you are online:

```bash
python setup_offline.py
```

To also prepare the Python packages for an air-gapped machine, download the wheels on a connected machine:

```bash
python setup_offline.py --wheels
```

then copy the whole folder (including `wheels/`) to the target machine and install from it:

```bash
pip install --no-index --find-links=./wheels -r requirements.txt
```

The server automatically uses the local `static/katex/` copy when it exists and falls back to the CDN otherwise.

---

## Running the calculator

From the repository root:

```bash
python main.py
```

and open **<http://127.0.0.1:8000>** in your browser.

`python main.py` starts Uvicorn with auto-reload (convenient while editing the code). For a non-reloading server, or to choose another host / port:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

> The first calculation of a given wave-mixing preset (and each higher retained order) solves the perturbative equations symbolically and may take from several seconds to minutes. Solutions are cached in the server process, so subsequent calculations with the same preset are fast. Restarting the server clears this cache.

---

## Web interface guide

The page has three tabs, from the most elementary to the most specialised:

| Tab | Purpose |
|---|---|
| **Single Mode** | Reorder a product of operators of one boson / fermion mode. |
| **Quantum Expression** | Bring any multi-mode expression to pattern form; build up named expressions. |
| **WM Evolution** | Heisenberg evolution of expressions in wave-mixing processes; input-state assertions. |

Every result area offers **Copy LaTeX**. Hovering over a result or table row shows additional details.

### Expression syntax (shared by all tabs)

Expressions are typed as plain text and parsed by SymPy with implicit multiplication enabled.

| Input | Meaning |
|---|---|
| `a b`, `2x`, `x a` | Implicit multiplication (a space or juxtaposition multiplies). |
| `*`, `/`, `+`, `-` | Usual arithmetic. |
| `**` or `^` | Power, e.g. `a**2` or `a^2`. |
| `dag(a)` (also `Dagger(a)`, `dagger(a)`) | Hermitian adjoint $a^\dagger$. Works on whole expressions, e.g. `dag(x a + a^2)`. |
| `N_(a)` (also `BosonNum(a)`) | Number operator $\hat N_a = a^\dagger a$ of a boson mode. |
| `Pat(a, e1, e2, ...)` (also `pattern`) | Operator pattern $a^{e_1}(a^\dagger)^{e_2}a^{e_3}\cdots$. If the first argument is `dag(a)`, the pattern starts with $a^\dagger$. Exponents may be symbols. |
| `BP(a, is_annihilation, e1, ...)`, `FP(c, is_annihilation, e1, ...)` | Explicit boson / fermion pattern (`BosonPat` / `FermionPat`), e.g. `BP(a, True, 2, 2)` $=a^2(a^\dagger)^2$. |
| `exp`, `sin`, `cos`, `sinh`, `cosh`, `tanh`, `sqrt` | Elementary functions; they may take operator arguments, e.g. `cos(N_(a))`. |
| `diff(expr, x)` | Derivative with respect to a commutative symbol. |
| `i` or `I`, `pi`, `e` | Imaginary unit, $\pi$, Euler's number. |

Identifiers you define must be valid Python names (letters, digits, underscores; not starting with a digit) and must not clash with the reserved names above (`i`, `e`, `pi`, `exp`, `dag`, `N_`, `Pat`, `BP`, `FP`, `MMN`, …). Names containing digits or underscores, such as `a1` or `b_1`, are never split into several symbols.

### Tab 1 — Single Mode

Reorders a single-mode operator product $\;c_1^{e_1}\, c_2^{e_2}\, c_3^{e_3}\cdots$ where the operators alternate between annihilation and creation.

1. **Particle Type** — *Boson* or *Fermion*.
2. **Operator Name** — e.g. `a`, `c`, `b_1`.
3. **First Operator** — whether the left-most factor is an *Annihilation* or a *Creation* operator; the following factors alternate.
4. **Exponent Sequence** — e.g. `1, 2, 1` means $a\,(a^\dagger)^2 a$. Exponents may be symbolic (e.g. `1, n_1, 2`); any character other than letters, digits and underscores acts as a separator.
5. Choose an action:
   - **Original Expression** — show the product as entered;
   - **Rewrite to PatForm** — the pattern form: a power of $a$ or $a^\dagger$ on the left times a function of $\hat N$ (for bosons), or the collapsed form among $0, 1, c, c^\dagger, c^\dagger c, c c^\dagger$ (for fermions);
   - **Normal Order** — all creation operators to the left.

   The result panel also states whether the result is Hermitian.
6. **Import to Tab 2 as PF** saves the result as a named variable in the Tab 2 table, so it can be reused in further expressions.

*Example:* boson `a`, first operator *Annihilation*, exponents `1, 2, 1` → PatForm $(1+\hat N)\hat N$, normal order $2a^\dagger a + (a^\dagger)^2 a^2$.

### Tab 2 — Quantum Expression

Brings an arbitrary expression of commutative symbols and boson / fermion operators to **pattern form**: a sum of terms, each being a product of canonical-operator powers (one per mode, on the left) and a function of number operators (on the right). The pattern form is unique, so two expressions are equal if and only if their pattern forms coincide.

1. Make sure every symbol and operator you use is defined in the **Defined Expressions** table below the input (by default: boson `a`, fermion `c`, real symbol `x`). Add new ones with **Add / Create ▼** → *New Commutative Symbol*, *New Boson Operator / Mode*, *New Fermion Operator / Mode*.
2. Type the expression in **Expression Input**, e.g. `(x*a + dag(a))**2`, `Pat(a, 1, 2) + Pat(c, 1, 1)`, or an expression using previously saved variables such as `H`.
3. Click
   - **Calculate** — assumes that different modes are mutually independent (all commute);
   - **Calculate with Remainder** — keeps track of the commutators generated between different modes while reordering, and shows them in a separate **remainder** panel (e.g. terms like $[a, b^\dagger]$). Use this when your modes are not independent. The remainder itself cannot be saved as a variable.
   
   Tick **Expand output** (next to the input label) to multiply out products and integer powers in the number-operator coefficients of the result and of the remainder. Terms that cancel then vanish, e.g. $1-(1+\hat N)+\hat N \to 0$ and $(1+\hat N)(2+\hat N) \to 2+3\hat N+\hat N^2$. Function arguments such as $\cos(1+\hat N)$ are left as they are. The option is remembered by the browser.
4. **Save to Table as PF** stores the result (in expanded form if *Expand output* was ticked) under a name (e.g. `H`), which can then be used in later inputs: `dag(H) H - H dag(H)`, `H^3`, ….

*Example:* with real `x`, `(x*a + dag(a))**2` → $x^2 a^2 + x(1+\hat N) + x\hat N + (a^\dagger)^2$.

### Tab 3 — WM Evolution

Computes the time evolution $O(t)$ of operator expressions in a multimode parametric wave-mixing process, expressed in terms of the input ($t=0$) operators, as a perturbative series in the depletion of the strong driving field.

#### Physical model and notation

Three field groups take part: mode 0 (pump, high frequency) and modes 1, 2 (signal and idler). Each weak field is in general **multimode**: the evolution couples every signal mode to its partner idler mode, and multimode sums are shown as $\sum_\text{modes}(\cdots)$. The symbol $n_m$ denotes the number of mode pairs.

| Preset | Strong (driving) field | Hamiltonian structure |
|---|---|---|
| **Down-conversion** (TWM) | pump $a_0$ | $a_0 \to a_1 + a_2$ |
| **Frequency Conversion / State Transfer** (TWM) | idler $a_2$ | $a_0 \leftrightarrow a_1$, driven by $a_2$ |
| **Four-Wave Mixing** | pump $a_0$ (degenerate) | $2a_0 \to a_1 + a_2$, including self- and cross-phase modulation |

In the three-wave-mixing case the Heisenberg equations are
$\dot a_0 = -i\chi\, a_1 a_2\, e^{-i\Delta k t}$,
$\dot a_1 = -i\chi\, a_0 a_2^\dagger\, e^{i\Delta k t}$,
$\dot a_2 = -i\chi\, a_1^\dagger a_0\, e^{i\Delta k t}$.

#### Step 1 — Process presets

Configure these **before** evaluating an expression:

- **Mixing Process Type** — *Down-conversion*, *Frequency Conversion / State Transfer*, or *Four-Wave Mixing*.
- **Phase-Matching Condition** — *Phase Matched* ($\Delta k = 0$; for FWM: *SPM Compensated*, $\Delta k = 2\chi \hat N_0$) or *Arbitrary Phase Mismatch* (keeps $\Delta k$ as a symbol).
- **Retained Order** — the perturbative order kept in the result. The strong field counts as order $-1$ (TWM) or $-1/2$ (FWM), the weak fields as order 0, and $\chi$ as order 1. Order 0 is the undepleted-pump (parametric) approximation; order 1 includes the first pump-depletion correction. Four-wave mixing currently supports orders up to 1.

#### Step 2 — Enter the expression

In addition to the general syntax, the following objects are predefined in this tab:

| Name | Meaning |
|---|---|
| `a_p`, `a_s`, `a_i` | **Evolved** pump, signal, idler operators at time $t$ (displayed as $a_0, a_1, a_2$). |
| `a_0`, `a_1`, `a_2` | **Input** operators at $t = 0$. |
| `A` | Multimode canonical operator at $t=0$: $\sum_\text{modes} a_1 a_2$ (down-conversion) or $\sum_\text{modes} a_1 a_2^\dagger$ (state transfer). |
| `N_t` | Multimode number operator at $t=0$: $\sum_\text{modes}(1 + \hat N_1 + \hat N_2)$ (down-conversion) or its state-transfer counterpart. |
| `MMN(a_s)` | Multimode number operator $\sum_\text{modes} \hat N_1$; `MMN(a_s, d)` adds a shift $d$ per mode. |
| `MMN2(d)` | `N_t` shifted by $d\,n_m$. |
| `nlc`, `pm` | Nonlinear coupling $\chi$ and phase mismatch $\Delta k$. |
| `t`, `g` | Time and parametric gain. |
| `n_m` | Number of mode pairs. |

**Multiplication rule.** Implicit multiplication (a space, e.g. `dag(a_s) a_s`) is the **multimode product**: paired weak-field operators are summed over all modes, giving e.g. $\sum_\text{modes}\hat N_1(t)$. An explicit `*` (e.g. `dag(a_s)*a_s`) is the **single-mode product**, i.e. the observable of one signal mode. Ticking **Single-mode processing** replaces all multimode operators with single-mode ones ($n_m = 1$).

Examples:

- `dag(a_p) a_p - N_(a_p)` — pump depletion $\hat N_0(t) - \hat N_0(0)$ (`N_(·)` of an evolved operator is the number operator of the corresponding input mode);
- `dag(a_s) a_s` — total signal photon number at time $t$;
- `dag(a_s)*a_s` — photon number of a single signal mode;
- `a_s a_i` — two-mode correlation $\sum_\text{modes} a_1(t)a_2(t)$.

**Cascaded system.** Tick *Cascaded system: evolve for* to let the calculated expression evolve through a second, identical wave-mixing stage for a further time. The time box defaults to `t` (the same duration as the first stage); you can replace it with the name of a new or existing custom commutative symbol, e.g. `t_c`. The box can be edited at any time, but it only takes effect while the checkbox is ticked. An empty box means `t`.

#### Step 3 — Calculate and read the result

Click **Calculate Evolution**. The result is shown as a function of time $t$; **Display in terms of gain** rewrites it in terms of the parametric gain operator $\hat g$ (e.g. $\hat g = \chi t\sqrt{\hat N_0}$ for phase-matched down-conversion), which usually gives a much more compact form. Click again to switch back to time.

**Save to Table** stores the result (in its time or gain form) as a named variable that can be reused in later Tab 3 expressions — for example, save `dag(a_s) a_s` as `Ns` and then compute `Ns^2 - Ns`.

Before calculating, the page checks your expression against the presets and warns you when

- a referenced variable was computed with a **lower retained order** than the current preset (*Force Continue* proceeds anyway, at the risk of missing expansion terms), or
- a referenced variable was computed under a **different mixing process** or **single-mode setting** (results would be mathematically inconsistent).

#### Step 4 — Input State Assertions

Switch to the **Input State Assertions** sub-tab to evaluate the expression for specific input states at $t=0$. Tick any combination of

- *Pump input is in coherent state* $|\alpha_0\rangle$ (state transfer: *Idler*, $|\alpha_2\rangle$),
- *Signal input is in vacuum state* $|0\rangle_1$,
- *Idler input is in vacuum state* $|0\rangle_2$ (state transfer: *Pump*, $|0\rangle_0$),

and click **Calculate under Assertions**. With all three ticked, the result is the expectation value; with only some ticked, it is a partial expectation value that remains an operator in the other modes. The gain display uses the scalar gain $g$ (e.g. $g = \chi t|\alpha_0|$).

*Example:* down-conversion, phase matched, order 0, input `dag(a_s) a_s`, all three assertions → $n_m \sinh^2(\chi t|\alpha_0|) = n_m\sinh^2 g$.

Results evaluated under assertions are expectation values and cannot be saved or used in further operator algebra.

### The variable tables

Tabs 2 and 3 each have a **Defined Expressions** table holding the symbols, operators and saved expressions available to the input:

- **Click** a row to edit it (identifier, LaTeX display text, notes, mathematical assumptions such as *Real*, *Integer*, *Non-negative*, *Positive*). In Tab 3, the detail view also shows each item's **leading order** and the **accuracy (order)** it was computed with; the leading order of custom symbols and boson modes is editable.
- **Right-click** a row for *Edit*, *Insert into Input* (inserts the identifier at the cursor) and *Delete*.
- **Add / Create ▼** adds new symbols, modes, or saves the current answer.
- **Reset Defaults** restores the default table of the tab (removing your own entries); in Tab 3, **Restore Base Solutions** only re-adds or un-renames the evolved operators `a_p`, `a_s`, `a_i`, and asks you to rename or delete any variable that currently occupies one of these names.

The **Display Text (LaTeX)** of an item only changes how it is rendered (e.g. identifier `omega`, display `\omega_0`).

Table contents are stored in your browser's local storage, so they survive page reloads and server restarts. Clearing the browser's site data resets them.

---

## HTTP API (for scripting)

The web page talks to the FastAPI backend through JSON endpoints, which can also be called directly. Interactive documentation of all request / response schemas is generated automatically at **<http://127.0.0.1:8000/docs>** while the server is running.

| Endpoint | Purpose |
|---|---|
| `POST /api/single_mode` | Tab 1 calculation. |
| `POST /api/pattern_form` | Tab 2 calculation (`use_rem` for the remainder mode). |
| `POST /api/solve` | Raw perturbative solution of one field (`var_index` 0/1/2) at a given order. |
| `POST /api/combination` | Tab 3 evolution of an expression. |
| `POST /api/combination/assert` | Tab 3 evaluation under input-state assertions. |
| `POST /api/combination/cascade` | Tab 3 cascaded-system evolution. |
| `POST /api/variable/save` | Save the latest result as a named variable. |
| `POST /api/combination/reset` | Reset the custom perturbative orders of the wave-mixing processors. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/combination \
     -H "Content-Type: application/json" \
     -d '{"text": "dag(a_s) a_s", "process": "twm_dc", "mismatch": false, "order": 0, "use_gain": true}'
```

For anything beyond the web workflow, use the `cas_qwm` library directly — see **[cas_qwm_documentation.md](cas_qwm_documentation.md)**.

---

## Troubleshooting

| Symptom | Remedy |
|---|---|
| Formulas are shown as raw LaTeX | KaTeX could not be loaded. Check the internet connection, or run `python setup_offline.py` once. |
| A calculation seems to hang | Higher retained orders (≥ 2) and phase-mismatched presets take considerably longer the first time; wait for the solver, or lower the order. |
| *"… is not defined"* / parsing error | Every symbol and operator must be defined in the tab's table. Check for typos and for reserved names. |
| Order or condition warning dialog | A referenced saved variable was computed with a lower order or under a different preset. Recompute it under the current preset, or choose *Force Continue*. |
| Port 8000 is in use | Start the server on another port: `uvicorn main:app --port 8001`. |
