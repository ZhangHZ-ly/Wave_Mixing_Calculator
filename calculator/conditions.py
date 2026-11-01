from __future__ import annotations
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class LogicRelation(str, Enum):
    """
    Propositional logic relation between expression state (E) and calculation preset (P).
    - XNOR: E == P (同或: absolute independence / mutually exclusive states, e.g. dc)
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
        label_true: str,
        label_false: str,
        convert_method: Optional[str] = None,
        conflict_msg: str = "",
    ):
        self.key = key
        self.name = name
        self.relation = relation
        self.label_true = label_true
        self.label_false = label_false
        self.convert_method = convert_method
        self.conflict_msg = conflict_msg

    def evaluate(self, preset_val: bool, expr_val: Optional[bool]) -> bool:
        """
        Evaluate propositional logic:
        - XNOR: expr_val == preset_val
        - IMPLIES: (not expr_val) or preset_val  (i.e. expr_val -> preset_val)
        """
        if expr_val is None:
            return True
        p = bool(preset_val)
        e = bool(expr_val)
        if self.relation == LogicRelation.XNOR:
            return p == e
        elif self.relation == LogicRelation.IMPLIES:
            return (not e) or p
        return True

    def needs_conversion(self, preset_val: bool, expr_val: Optional[bool]) -> bool:
        """
        Returns True if the expression condition is compatible with the preset
        via a projection/conversion step (e.g. Multimode expression used in Single Mode calculation).
        """
        if expr_val is None:
            return bool(preset_val) and bool(self.convert_method)
        p = bool(preset_val)
        e = bool(expr_val)
        if self.relation == LogicRelation.IMPLIES:
            return (not e) and p
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
    'dc': ConditionRule(
        key='dc',
        name='Mixing Process (dc)',
        relation=LogicRelation.XNOR,
        label_true='Down-conversion (dc=True)',
        label_false='State Transfer (dc=False)',
        conflict_msg='Variables containing default operators have different algebraic definitions and subspaces across down-conversion and state transfer.'
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
    presets: Dict[str, bool]
) -> List[Dict[str, Any]]:
    """
    Check all referenced named expressions against current calculation presets.
    Returns a list of conflict diagnostics.
    """
    conflicts = []
    for item in named_expressions:
        for key, rule in CONDITION_REGISTRY.items():
            if key not in presets:
                continue
            preset_val = presets[key]
            expr_val = item.get(key)
            if key == 'dc':
                has_default = item.get('has_default_ops', item.get('hasDefaultOps', True))
                if not has_default:
                    continue
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
