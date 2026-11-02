from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional


PROCESS_LABELS: Dict[str, str] = {
    'twm_dc': 'Down-conversion (TWM)',
    'twm_fc': 'State Transfer (TWM)',
    'fwm': 'Four-Wave Mixing (FWM)',
}


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


class LogicRelation(str, Enum):
    """
    Propositional logic relation between expression state (E) and calculation preset (P).
    - XNOR: E == P (同或: absolute independence / mutually exclusive states, e.g. process)
    - IMPLIES: E -> P (蕴含 / 包含: expression state implies preset state, e.g. single_mode)
    """
    XNOR = "xnor"
    IMPLIES = "implies"


class ConditionRule:
    """
    Encapsulates compatibility logic, evaluation, and projection/conversion
    between an expression condition flag and an environment preset flag.
    """
    def __init__(
        self,
        key: str,
        name: str,
        relation: LogicRelation,
        label_true: str = "",
        label_false: str = "",
        value_labels: Optional[Dict[Any, str]] = None,
        convert_method: Optional[str] = None,
        conflict_msg: str = "",
    ):
        self.key = key
        self.name = name
        self.relation = relation
        self.label_true = label_true
        self.label_false = label_false
        self.value_labels = value_labels or {}
        self.convert_method = convert_method
        self.conflict_msg = conflict_msg

    def evaluate(self, preset_val: Any, expr_val: Any) -> bool:
        """
        Evaluate propositional logic:
        - XNOR: expr_val == preset_val
        - IMPLIES: (not expr_val) or preset_val  (i.e. expr_val -> preset_val)
        """
        if expr_val is None:
            return True
        if self.relation == LogicRelation.XNOR:
            if isinstance(preset_val, str) or isinstance(expr_val, str):
                return str(preset_val) == str(expr_val)
            return bool(preset_val) == bool(expr_val)
        elif self.relation == LogicRelation.IMPLIES:
            return (not bool(expr_val)) or bool(preset_val)
        return True

    def needs_conversion(self, preset_val: Any, expr_val: Any) -> bool:
        """
        Returns True if the expression condition is compatible with the preset
        via a projection/conversion step (e.g. Multimode expression used in Single Mode calculation).
        """
        if expr_val is None:
            return bool(preset_val) and bool(self.convert_method)
        if self.relation == LogicRelation.IMPLIES:
            return (not bool(expr_val)) and bool(preset_val)
        return False

    def convert(self, obj: Any, solver: Any = None, dc: bool = True) -> Any:
        """Apply conversion method if registered on solver or obj."""
        if self.convert_method:
            if solver is not None and hasattr(solver, self.convert_method):
                return getattr(solver, self.convert_method)(obj)
            if hasattr(obj, self.convert_method):
                return getattr(obj, self.convert_method)()
        return obj


CONDITION_REGISTRY: Dict[str, ConditionRule] = {
    'process': ConditionRule(
        key='process',
        name='Mixing Process Type',
        relation=LogicRelation.XNOR,
        value_labels=PROCESS_LABELS,
        conflict_msg='Variables containing default operators have different algebraic definitions, driving-mode orders, and evolution laws across different wave-mixing processes.'
    ),
    'single_mode': ConditionRule(
        key='single_mode',
        name='Single-mode Processing',
        relation=LogicRelation.IMPLIES,
        label_true='Single Mode',
        label_false='Multimode',
        convert_method='to_single_mode',
        conflict_msg='Expressions created under Single Mode cannot be used in Multimode calculations because multimode summation and mode degeneracy details were collapsed.'
    ),
}


def register_condition_rule(rule: ConditionRule):
    """Register a new conditional rule for extensible compatibility checking."""
    CONDITION_REGISTRY[rule.key] = rule


def check_expression_compatibility(
    named_expressions: List[Dict[str, Any]],
    presets: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Check all referenced named expressions against current calculation presets.
    Returns a list of conflict diagnostics.
    """
    conflicts = []
    norm_presets = dict(presets)
    if 'process' not in norm_presets and 'dc' in norm_presets:
        norm_presets['process'] = normalize_process(None, norm_presets['dc'])

    for item in named_expressions:
        for key, rule in CONDITION_REGISTRY.items():
            if key not in norm_presets:
                continue
            preset_val = norm_presets[key]
            expr_val = item.get(key)
            if key == 'process':
                has_default = item.get('has_default_ops', item.get('hasDefaultOps', True))
                if not has_default:
                    continue
                if expr_val is None and item.get('dc') is not None:
                    expr_val = normalize_process(None, item.get('dc'))
            if not rule.evaluate(preset_val, expr_val):
                conflicts.append({
                    'key': key,
                    'name': rule.name,
                    'item_name': item.get('name', ''),
                    'preset_val': preset_val,
                    'expr_val': expr_val,
                    'conflict_msg': rule.conflict_msg,
                })
    return conflicts
