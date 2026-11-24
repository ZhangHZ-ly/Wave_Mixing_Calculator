from __future__ import annotations
import re
from typing import Any, Dict, Iterable, List, Optional


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


PROCESS_LABELS = {'twm_dc': 'Down-conversion', 'twm_fc': 'State Transfer', 'fwm': 'Four-Wave Mixing'}
_SUB = '₀₁₂'


def _refers(text: str, name: Optional[str]) -> bool:
    return bool(name) and re.search(rf'\b{re.escape(name)}\b', text) is not None


def referenced_expressions(text: str, named_expressions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {ne['id']: ne for ne in named_expressions}
    stack = [ne['id'] for ne in named_expressions if _refers(text, ne.get('name'))]
    found: Dict[str, Dict[str, Any]] = {}
    while stack:
        def_id = stack.pop()
        if def_id in found or def_id not in by_id:
            continue
        found[def_id] = by_id[def_id]
        stack.extend((by_id[def_id].get('refs') or {}).values())
    return list(found.values())


def _label(ne: Dict[str, Any]) -> str:
    return ne.get('name') or ne.get('source_input') or ne['id']


def condition_conflicts(text: str, named_expressions: List[Dict[str, Any]], process: Optional[str],
                        single_mode: bool, degenerate: bool) -> List[Dict[str, str]]:
    """
    Saved expressions referenced (also through other saved expressions) by `text` that were
    created under presets incompatible with the current ones: another mixing process (if they
    contain default operators), single-mode processing in a multimode calculation, or the other
    single-mode degeneracy. Returns [{name, reason}].
    """
    process = normalize_process(process)
    degenerate = is_degenerate(process, single_mode, degenerate)
    label = lambda p: PROCESS_LABELS.get(p, p)
    deg_label = lambda d: 'Degenerate' if d else 'Non-degenerate'
    out = []
    for ne in referenced_expressions(text, named_expressions):
        p = ne.get('process')
        if ne.get('has_default_ops') and p and p != process:
            out.append({'name': _label(ne), 'reason': f'created under {label(p)}, incompatible with '
                                                      f'current preset ({label(process)})'})
        if ne.get('single_mode'):
            if not single_mode:
                out.append({'name': _label(ne), 'reason': 'created under Single Mode, incompatible '
                                                          'with current Multimode preset'})
            elif bool(ne.get('degenerate')) != degenerate:
                out.append({'name': _label(ne), 'reason': (
                    f'created under {deg_label(ne.get("degenerate"))} Single Mode, incompatible with '
                    f'current {deg_label(degenerate)} Single Mode preset')})
    return out


def assertion_conds(cfg: Dict[str, Any], process: str, single_mode: bool) -> List[str]:
    """Input states of an assertion configuration, e.g. ['|α₀⟩', '|0⟩₁']."""
    dc = process_is_dc(process)
    seeded = cfg.get('seed_mode') is not None
    weak_vac = not (seeded and single_mode)
    conds = []
    if cfg.get('coherent_drive'):
        conds.append('|α₀⟩' if dc else '|α₂⟩')
    if cfg.get('vacuum_signal') and weak_vac:
        conds.append('|0⟩₁')
    if cfg.get('vacuum_other_weak') and weak_vac:
        conds.append('|0⟩₂' if dc else '|0⟩₀')
    if seeded:
        obs = ' (observed)' if cfg.get('seed_observed') else ''
        mode = int(cfg['seed_mode'])
        if process == 'twm_dc' and mode == 0:
            conds.append(f'seed |β₁⟩ (degenerate){obs}')
        else:
            j = mode if process == 'twm_fc' else 1
            k = 1 - j if process == 'twm_fc' else 2
            other = f'|0⟩{_SUB[k]}' if cfg.get('seed_vac_c', True) else f'|β{_SUB[k]}⟩'
            conds.append(f'seed |β{_SUB[j]}⟩ + {other}{obs}')
    return conds


def _assertion_key(cfg: Dict[str, Any], single_mode: bool) -> Optional[tuple]:
    weak_vac = not (cfg.get('seed_mode') is not None and single_mode)
    key = (bool(cfg.get('coherent_drive')), weak_vac and bool(cfg.get('vacuum_signal')),
           weak_vac and bool(cfg.get('vacuum_other_weak')))
    return key if any(key) else None


def assertion_conflicts(text: str, named_expressions: List[Dict[str, Any]],
                        operator_names: Iterable[str], process: Optional[str]) -> List[Dict[str, str]]:
    """
    Asserted expressions entered in `text` together with content under no assertions (operators,
    saved expressions of operators) or with asserted expressions of another configuration.
    Seeding is not compared: expressions asserted only by seeding count as unasserted ones.
    Returns [{name, reason}] of all the mixed items, or [] if there is no such mixing.
    """
    process = normalize_process(process)
    named = [ne for ne in named_expressions if _refers(text, ne.get('name'))]
    groups: Dict[Any, List[Dict[str, str]]] = {}
    for ne in named:
        cfg = ne.get('assertion')
        if cfg:
            sm = bool(ne.get('single_mode'))
            key = _assertion_key(cfg, sm)
            reason = (f"asserted expression [{', '.join(assertion_conds(cfg, ne.get('process') or process, sm))}]"
                      if key else 'seeded expression without other assertions')
            groups.setdefault(key, []).append({'name': ne['name'], 'reason': reason})
        elif ne.get('has_default_ops'):
            groups.setdefault(None, []).append(
                {'name': ne['name'], 'reason': 'saved expression without assertions'})
    named_names = {ne['name'] for ne in named}
    for name in dict.fromkeys(('a_0', 'a_1', 'a_2', 'N_', 'MMN', 'MMN2', *operator_names)):
        if name not in named_names and _refers(text, name):
            groups.setdefault(None, []).append({'name': name, 'reason': 'operator without assertions'})
    if len(groups) > 1 and any(k is not None for k in groups):
        return [e for entries in groups.values() for e in entries]
    return []
