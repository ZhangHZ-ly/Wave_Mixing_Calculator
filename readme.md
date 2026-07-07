**This is a SymPy-based library to process symbolic expressions with canonical operators in quantum mechanics/optics. It includes two main preprocessing operations on the given quantum expression. The first operation is collection and reordering the oprators in the expression and combine the creation and annihilation operators into number operators as much as possible; the second operation is Talor expasion for functions in the expression up to the given order, where the order of each single symbol/operator can be self defined.** 

## Representation of expressions — `op_patterns`
The product of bosonic/fermionic operators in the same mode is rewritten into `BosonPat`/`FermionPat` class (see `boson_pattern.BosonPat` & `ferimon_pattern.FermionPat`). The `num_repr` method combines the creation and annihilation bonsonic operators into number operators (`boson_pattern.BosonNum`) as much as possible; while this is unnecessary for fermionic operators, because any multiplication of fermionic operators (denote $c$ for annihilation) in the same mode must equals one of $0$, $1$, $c$, $c^\dagger$, $c^\dagger c$ and $c c^\dagger$ — the `__new__` method of the class will automatically collapse the expression into one of these forms if all the exponents of the operator are known.\\
An example of the relavant classes and methods:\\
`>>> from op_patterns import BosonPat as BP`\\
`>>> bp = BP('a', True, 2, 2)`\\
`>>> bp`\\
`BP(a; 2, 2)`\\
`>>> bp.num_repr()`\\
`(1, (1 + N_(a))*(2 + N_(a)))`\\
`>>> from sympy.physics.quantum.boson import BosonOp`\\
`>>> bp.rewrite(BosonOp)`\\
`a**2*Dagger(a)**2`\\
`from op_patterns import BosonNum`\\
`>>> bp.rewrite(BosonNum)`\\
`(1 + N_(a))*(2 + N_(a))`\\
`>>> from op_patterns import FermionPat as FP`\\
`>>> FP('c', True, 1, 1, 1)`\\
`c`\\
Note that by inputing directly multiplications of operators, the output won't combine all the factors in the same mode to form a unified pattern item; the multiplication of operatorsfrom the same mode can be taken into account by applying the `redo_mul` function (see `tools.redo_mul`):\\
`>>> import op_patterns`\\
`>>> from sympy.physics.quantum.boson import BosonOp`\\
`>>> from sympy.physics.quantum.dagger import Dagger`\\
`>>> a = BosonOp('a')`\\
`>>> a**2 * Dagger(a)**2`\\
`BP(a; 2)*BP(D(a); 2)`\\
`>>> from op_patterns import redo_mul`\\
`>>> redo_mul(a**2 * Dagger(a)**2)`\\
`BP(a; 2, 2)`

## Reordering and expanding the quantum expression — `perturb_eval`

The function `pf` (see `pattern_form.pattern_form`) reorders the 
