**This is a SymPy-based library to process symbolic expressions with canonical operators in quantum mechanics/optics. It includes two main preprocessing operations on the given quantum expression. The first operation is collection and reordering the oprators in the expression and combine the creation and annihilation operators into number operators as much as possible; the second operation is Talor expasion for functions in the expression up to the given order, where the order of each single symbol/operator can be self defined.** 

## Representation of expressions — `op_patterns`
The product of bosonic/fermionic operators in the same mode is rewritten into `BosonPat`/`FermionPat` class (see `boson_pattern.BosonPat` & `ferimon_pattern.FermionPat`). The `num_repr` method combines the creation and annihilation bonsonic operators into number operators (`boson_pattern.BosonNum`) as much as possible; while this is unnecessary for fermionic operators, because any multiplication of fermionic operators (denote $c$ for annihilation) in the same mode must equals one of $0$, $1$, $c$, $c^\dagger$, $c^\dagger c$ and $c c^\dagger$ — the `__new__` method of the class will automatically collapse the expression into one of these forms if all the exponents of the operator are known.

An example of the relavant classes and methods:

`>>> from op_patterns import BosonPat as BP`

`>>> bp = BP('a', True, 2, 2)`

`>>> bp`

`BP(a; 2, 2)`

`>>> bp.num_repr()`

`(1, (1 + N_(a))*(2 + N_(a)))`

`>>> from sympy.physics.quantum.boson import BosonOp`

`>>> bp.rewrite(BosonOp)`

`a**2*Dagger(a)**2`

`from op_patterns import BosonNum`

`>>> bp.rewrite(BosonNum)`

`(1 + N_(a))*(2 + N_(a))`

`>>> from op_patterns import FermionPat as FP`

`>>> FP('c', True, 1, 1, 1)`

`c`

Note that by inputing directly multiplications of operators, the output won't combine all the factors in the same mode to form a unified pattern item; the multiplication of operatorsfrom the same mode can be taken into account by applying the `redo_mul` function (see `tools.redo_mul`):

`>>> import op_patterns`

`>>> from sympy.physics.quantum.boson import BosonOp`

`>>> from sympy.physics.quantum.dagger import Dagger`

`>>> a = BosonOp('a')`

`>>> a**2 * Dagger(a)**2`

`BP(a; 2)*BP(D(a); 2)`

`>>> from op_patterns import redo_mul`

`>>> redo_mul(a**2 * Dagger(a)**2)`

`BP(a; 2, 2)`

## Reordering and expanding the quantum expression — `perturb_eval`
The function `pf` (see `pattern_form.pattern_form`) reorders the quantum expression into the so-called "pattern form", which is unique for equal expressions. This hence allows to establish commutative equations in the noncommutative algebra of quantum operators.

In the reordering process, the expression is saved in a `dict` structure to seperate the pure `BosonPat`/`FermionPat` factors and the `BosonNum` (in keys) and commutative factors (in values). The "res" parameter provides an option to calculate the generated commutators along the reordering procedure if the quantum modes are not independent with each other.

`>>> from perturb_eval import pf`

`>>> from sympy import symbols, cos, sin`

`>>> from sympy.physics.quantum.boson import BosonOp`

`>>> from sympy.physics.quantum.dagger import Dagger`

`>>> a, b = symbols('a b', cls=BosonOp)`

`>>> pf(a**2*(cos(Dagger(a)*a) + Dagger(b)*sin(b*Dagger(b)))*Dagger(a)*b**2)`

`a*b*(1 + N_(a))*(1 + N_(b))*sin(-1 + N_(b)) + a*BP(b; 2)*(1 + N_(a))*cos(1 + N_(a))`

`>>> pf(Dagger(b)*sin(b*Dagger(b))*Dagger(a), res=True)`

`(Dagger(a)*Dagger(b)*sin(1 + N_(b)), -Dagger(b)*[Dagger(a),sin(1 + N_(b))] - [Dagger(a),Dagger(b)]*sin(1 + N_(b)))`

The class `PF` (see `sortcontext.PFTableProcessor`) includes methods for basic algebraic operations such as addition, multiplication and integer power. It is usually more efficient to use this class instead of `sympy.Expr` if the calculation mainly treats pattern-formed expressions. To create `PF` objects, use `as_dict=None` option in `pf` function, or input directly the `dict` to the class. But the direct creation of `PF` object also accepts illegal inputs.

`>>> from perturb_eval import pf, PF`

`>>> from op_patterns import BonsonNum`

`>>> from sympy import symbols, cos, sin`

`>>> from sympy.physics.quantum.boson import BosonOp`

`>>> from sympy.physics.quantum.dagger import Dagger`

`>>> a, b = symbols('a b', cls=BosonOp)`

`>>> pf(a**2*(cos(Dagger(a)*a) + Dagger(b)*sin(b*Dagger(b)))*Dagger(a)*b**2, as_dict=None)`

`PatternForm({((a, b), ()): (-1 + N_(b))*(1 + N_(a))*sin(-1 + N_(b)), ((a, BP(b; 2)), ()): (1 + N_(a))*cos(1 + N_(a))}, 0)`

`>>> pf1 = PF({((a**2,), ()): cos(BosonNum('a'))})`

`>>> pf2 = PF({((Dagger(a),), ()): sin(BosonNum('a'))})`

`>>> pf1.mul_ind(pf2)`

`PatternForm({((a,), ()): (1 + N_(a))*cos(1 + N_(a))*sin(N_(a))}, 0)`

The in the second entrance of `PF` indicates the part that cannot be represented into pattern form.

`>>> from perturb_eval import pf`

`>>> from sympy.physics.quantum.boson import BosonOp`

`>>> from sympy.functions.elementary.triganometric import sin`

`>>> pf(sin(BosonOp('a')), as_dict=None)`

`PatternForm({}, sin(a))`

`PE` class (see `pertrub_expander.PerturbExpander`) can expand the expression up to the desired order, which can turn a difference equation into differential equation, so as to solve the analytical solution for the quantum process.

`>>> from perturb_eval import PE`

`>>> from sympy import symbols, sin`

`>>> x, y = symbols('x y')`

`>>> pe = PE({x: 1})`

`>>> tuple(pe.expand_expr(sin(x + y), 2))`

`((sin(y), 0), (x*cos(y), 1), (-x**2*sin(y)/2, 2))`
