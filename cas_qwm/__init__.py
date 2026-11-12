"""
cas_qwm: Computer Algebra System for Quantum Wave Mixing.

Layered architecture (in ascending order of abstraction, descending order of priority):
1. op_pattern (highest priority / base layer):
   Elementary operator pattern representations (Boson, Fermion, commutators).
2. pattern_perturb:
   PatternForm (PF) normal ordered polynomial representations and perturbation expansion (PE).
3. parametric_multimode:
   Multimode processors, solvers (TWM, FWM), differential equations (dsolve_pm), and algebraic wrappers.

Lazy loading is implemented via PEP 562 (__getattr__, __dir__).
Submodules and their exported symbols are loaded on demand upon first access,
strictly following the priority order: op_pattern > pattern_perturb > parametric_multimode.
"""

import importlib
import sys
from typing import TYPE_CHECKING, Any, Dict, List, Set

if TYPE_CHECKING:
    from . import op_pattern
    from . import pattern_perturb
    from . import parametric_multimode

    from .op_pattern import (
        BosonNum,
        BosonPat,
        NumShift,
        apply_num_shift,
        FermionPat,
        UidCommutator,
        extract_modes,
        check_num,
        in_boson_num,
        mul_pattern,
        redo_mul,
    )
    from .pattern_perturb import (
        BSC,
        FSC,
        PF,
        pf,
        pf_add,
        PF0,
        PF1,
        PE,
        perturb_rec,
    )
    from .parametric_multimode import (
        a_p,
        a_s,
        a_i,
        t_,
        nlc,
        p2s,
        dsolve_pm1,
        dsolve_pm2,
        MultimodeCOP,
        MultimodeNum,
        MultimodeNum2,
        WMProcessor,
        wm_processors,
        _g,
        nm,
        vac_pf,
        WMWrapper,
        pm,
        WMSolver,
        TWMSolver,
        FWMSolver,
    )

# Submodules in priority order (op_pattern is the base layer with highest priority)
_PRIORITY_MODULES: List[str] = [
    'cas_qwm.op_pattern',
    'cas_qwm.pattern_perturb',
    'cas_qwm.parametric_multimode',
]

_SUBMODULES: Dict[str, str] = {
    'op_pattern': 'cas_qwm.op_pattern',
    'pattern_perturb': 'cas_qwm.pattern_perturb',
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
    'cas_qwm.pattern_perturb': [
        'BSC',
        'FSC',
        'PF',
        'pf',
        'pf_add',
        'PF0',
        'PF1',
        'PE',
        'perturb_rec',
    ],
    'cas_qwm.parametric_multimode': [
        'a_p',
        'a_s',
        'a_i',
        't_',
        'nlc',
        'p2s',
        'dsolve_pm1',
        'dsolve_pm2',
        'MultimodeCOP',
        'MultimodeNum',
        'MultimodeNum2',
        'WMProcessor',
        'wm_processors',
        '_g',
        'nm',
        'vac_pf',
        'WMWrapper',
        'pm',
        'WMSolver',
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
    'pattern_perturb',
    'parametric_multimode',
]
_seen: Set[str] = set(__all__)
for _mod_name in _PRIORITY_MODULES:
    for _sym in _MODULE_EXPORTS[_mod_name]:
        if _sym not in _seen:
            _seen.add(_sym)
            __all__.append(_sym)


def __getattr__(name: str) -> Any:
    """
    Lazy attribute and submodule loader adhering to PEP 562.
    Resolves attributes on demand following the priority order:
    cas_qwm.op_pattern > cas_qwm.pattern_perturb > cas_qwm.parametric_multimode.
    """
    if name.startswith('__') and name.endswith('__'):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    # 1. Submodule access
    if name in _SUBMODULES:
        mod = importlib.import_module(_SUBMODULES[name])
        globals()[name] = mod
        return mod

    # 2. Known mapped symbols
    if name in _SYMBOL_MAP:
        mod_name = _SYMBOL_MAP[name]
        mod = importlib.import_module(mod_name)
        val = getattr(mod, name)
        globals()[name] = val
        return val

    # 3. Dynamic search across layers in priority order (op_pattern -> pattern_perturb -> parametric_multimode)
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
