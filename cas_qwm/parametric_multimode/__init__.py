from importlib import import_module
from ..wm_variables import *
from ..dsolve_pm import *
from .multimode import *
from .wm_processor import *
from .wm_wrapper import *
from .wm_solver import *
from ..wm_variables import __all__ as _wv_all
from ..dsolve_pm import __all__ as _dp_all
from .multimode import __all__ as _mm_all
from .wm_processor import __all__ as _wp_all
from .wm_wrapper import __all__ as _ww_all
from .wm_solver import __all__ as _ws_all

__all__ = _wv_all + _dp_all + _mm_all + _wp_all + _ww_all + _ws_all

_NUMERIC = ['Chain', 'collective', 'pair', 'degenerate', 'Evolution', 'Layout', 'observable', 'initial_state',
            'to_fock']
__all__ += _NUMERIC


def __getattr__(name):
    if name == 'numeric' or name in _NUMERIC:
        numeric = import_module(__name__ + '.numeric')
        return numeric if name == 'numeric' else getattr(numeric, name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
