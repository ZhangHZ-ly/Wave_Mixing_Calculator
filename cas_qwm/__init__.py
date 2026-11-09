"""
cas_qwm: Computer Algebra System for Quantum Wave Mixing.

Layered architecture (in ascending order of abstraction, descending order of priority):
1. op_pattern (highest priority / base layer):
   Elementary operator pattern representations (Boson, Fermion, commutators).
2. pattern_form:
   PatternForm (PF) normal ordered polynomial representations and perturbation expansion (PE).
3. parametric_multimode:
   Multimode processors, solvers (TWM, FWM), differential equations (dsolve_pm), and algebraic wrappers.

Lazy loading is implemented via PEP 562 (__getattr__, __dir__) and an import hook.
Submodules and their exported symbols are loaded on demand upon first access,
strictly following the priority order: op_pattern > pattern_form > parametric_multimode.
"""

from importlib.abc import MetaPathFinder
import importlib
from importlib.util import find_spec
import sys
from typing import Any, Dict, List, Optional, Set

# Submodules in priority order (op_pattern is the base layer with highest priority)
_PRIORITY_MODULES: List[str] = [
    'cas_qwm.op_pattern',
    'cas_qwm.pattern_form',
    'cas_qwm.parametric_multimode',
]

_SUBMODULES: Dict[str, str] = {
    'op_pattern': 'cas_qwm.op_pattern',
    'op_patterns': 'cas_qwm.op_pattern',
    'pattern_form': 'cas_qwm.pattern_form',
    'parametric_multimode': 'cas_qwm.parametric_multimode',
}

_MODULE_EXPORTS: Dict[str, List[str]] = {
    'cas_qwm.op_pattern': [
        'BosonNum',
        'BosonPat',
        'NumShift',
        'apply_num_shift',
        'FermionPat',
        'UidCommutator',
        'extract_modes',
        'check_num',
        'in_boson_num',
        'mul_pattern',
        'redo_mul',
    ],
    'cas_qwm.pattern_form': [
        'BSC',
        'FSC',
        'PF',
        'pf',
        'PE',
        'PF0',
        'PF1',
    ],
    'cas_qwm.parametric_multimode': [
        'p2s',
        'dsolve_pm1',
        'dsolve_pm2',
        'MultimodeCOP',
        'MultimodeNum',
        'MultimodeNum2',
        'mmn_shift',
        'nd_simp',
        'PMProcessor',
        'wm_processors',
        'a_p',
        'a_s',
        'a_i',
        't_',
        '_g',
        '_ps',
        'nlc',
        'nm',
        'vac_pf',
        'PMWrapper',
        'pm',
        'PMSolver',
        'TWMSolver',
        'FWMSolver',
    ],
}

# Construct static symbol mapping in priority order.
# Reversed iteration ensures higher-priority layers overwrite lower-priority ones.
_SYMBOL_MAP: Dict[str, str] = {}
for _mod_name in reversed(_PRIORITY_MODULES):
    for _sym in _MODULE_EXPORTS[_mod_name]:
        _SYMBOL_MAP[_sym] = _mod_name

# Construct __all__ containing submodules and symbols in priority order
__all__: List[str] = [
    'op_pattern',
    'op_patterns',
    'pattern_form',
    'parametric_multimode',
]
_seen: Set[str] = set(__all__)
for _mod_name in _PRIORITY_MODULES:
    for _sym in _MODULE_EXPORTS[_mod_name]:
        if _sym not in _seen:
            _seen.add(_sym)
            __all__.append(_sym)


class _OpPatternsAliasFinder(MetaPathFinder):
    """Import hook alias: cas_qwm.op_patterns -> cas_qwm.op_pattern"""
    def find_spec(self, fullname: str, path: Optional[Any], target: Optional[Any] = None) -> Any:
        if fullname == 'cas_qwm.op_patterns':
            return find_spec('cas_qwm.op_pattern')
        return None


if not any(isinstance(f, _OpPatternsAliasFinder) for f in sys.meta_path):
    sys.meta_path.insert(0, _OpPatternsAliasFinder())


def __getattr__(name: str) -> Any:
    """
    Lazy attribute and submodule loader adhering to PEP 562.
    Resolves attributes on demand following the priority order:
    cas_qwm.op_pattern > cas_qwm.pattern_form > cas_qwm.parametric_multimode.
    """
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # 1. Submodule access
    if name in _SUBMODULES:
        target_mod = _SUBMODULES[name]
        mod = importlib.import_module(target_mod)
        globals()[name] = mod
        if name in ('op_pattern', 'op_patterns'):
            globals()['op_pattern'] = mod
            globals()['op_patterns'] = mod
            sys.modules.setdefault('cas_qwm.op_patterns', mod)
            sys.modules.setdefault('cas_qwm.op_pattern', mod)
        return mod

    # 2. Known mapped symbols
    if name in _SYMBOL_MAP:
        mod_name = _SYMBOL_MAP[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, name)
        globals()[name] = val
        return val

    # 3. Dynamic search across layers in priority order (op_pattern -> pattern_form -> parametric_multimode)
    for mod_name in _PRIORITY_MODULES:
        if mod_name in sys.modules:
            mod = sys.modules[mod_name]
            if hasattr(mod, name):
                val = getattr(mod, name)
                globals()[name] = val
                return val
        else:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, name):
                    val = getattr(mod, name)
                    globals()[name] = val
                    return val
            except ImportError:
                continue

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> List[str]:
    """
    Returns available attributes without forcing premature imports of unaccessed submodules.
    """
    attrs = set(globals().keys()) | set(__all__)
    for mod_name in _PRIORITY_MODULES:
        if mod_name in sys.modules:
            attrs.update(getattr(sys.modules[mod_name], '__all__', []))
    return sorted(attrs)

