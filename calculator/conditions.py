from __future__ import annotations
from typing import Optional


def normalize_process(process: Optional[str] = None, dc: Optional[bool] = None) -> str:
    """
    Normalize mixing process identifier to one of:
    - 'twm_dc': Three-Wave Mixing Down-conversion
    - 'twm_fc': Three-Wave Mixing Frequency Conversion / State Transfer
    - 'fwm': Four-Wave Mixing
    """
    if isinstance(process, str):
        p = process.strip().lower()
        if p in ('twm_dc', 'dc', 'true'):
            return 'twm_dc'
        if p in ('twm_fc', 'fc', 'false'):
            return 'twm_fc'
        if p == 'fwm':
            return 'fwm'
    if dc is not None:
        return 'twm_dc' if bool(dc) else 'twm_fc'
    return 'twm_dc'


def process_is_dc(process: str) -> bool:
    """Returns True when pump mode a_0 is the strong driving field ('twm_dc' or 'fwm')."""
    return normalize_process(process) != 'twm_fc'


def is_degenerate(process: str, single_mode: bool, degenerate: bool) -> bool:
    """
    Effective degenerate single-mode processing (signal = idler): only with single-mode
    processing of three-wave down-conversion.
    """
    return bool(single_mode) and bool(degenerate) and normalize_process(process) == 'twm_dc'
