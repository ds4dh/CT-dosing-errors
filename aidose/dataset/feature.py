# aidose/dataset/feature.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Type


@dataclass(frozen=True)
class Feature:
    """
    A typed feature wrapper.

    - `value`: the feature's value (may be None to represent "missing")
    - `declared_type`: a Python type (e.g., `str`, `int`, `bool`, or a **concrete Enum subclass**)

    Rules:
      - If `value is None`, the feature is accepted (missing) for any declared_type.
      - If `declared_type` is `bool`, only actual bool values are accepted (not ints).
      - If `declared_type` is an Enum subclass, an instance of that **same** Enum subclass is required.
      - `declared_type` MUST NOT be the Enum base class itself.
      - Otherwise, `isinstance(value, declared_type)` must be True.

    One-hot:
      - Only valid for Enum-typed features (declared_type must be a concrete Enum subclass).
      - If `value is None`, all one-hot entries have value None (type=bool).
    """
    value: Any
    declared_type: Type[Any]

    def __post_init__(self) -> None:
        # Ensure declared_type is a real type object
        if not isinstance(self.declared_type, type):
            raise TypeError("declared_type must be a type object (e.g., str, int, bool, or a concrete Enum subclass).")

        # Never allow the Enum base as a declared type
        if self.declared_type is Enum:
            raise TypeError("declared_type must be a concrete Enum subclass, not Enum.")

        # Allow None (missing) for any declared_type
        if self.value is None:
            return

        # Strict bool: exclude ints (True/False only)
        if self.declared_type is bool:
            if type(self.value) is not bool:
                raise TypeError(f"Feature expects value of type bool, got {type(self.value).__name__}")
            return

        # Enum handling: declared_type must be a concrete Enum subclass
        if issubclass(self.declared_type, Enum):
            if not isinstance(self.value, self.declared_type):
                raise TypeError(f"Feature expects Enum of type {self.declared_type}, got {type(self.value)}")
            return

        # General case
        if not isinstance(self.value, self.declared_type):
            raise TypeError(
                f"Feature expects value of type {self.declared_type.__name__}, got {type(self.value).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Return a { 'value': ..., 'type': <type object> } representation."""
        return {"value": self.value, "type": self.declared_type}

    def add_to_row(self, row: Dict[str, Any], name: str) -> None:
        """Insert this feature under `name` into the given row dict."""
        row[name] = self.to_dict()

    def to_one_hot_entries(self, name: str) -> Dict[str, Dict[str, Any]]:
        """
        For Enum-typed features, produce a one-hot mapping:
          { f"{name}_{MEMBER.name}": {"value": bool|None, "type": bool}, ... }

        - declared_type must be a concrete Enum subclass (never Enum base).
        - If value is None, all entries have value None.

        Raises:
            TypeError if declared_type is not a concrete Enum subclass.
        """
        if not (isinstance(self.declared_type, type) and issubclass(self.declared_type, Enum)):
            raise TypeError("to_one_hot_entries requires declared_type to be a concrete Enum subclass.")

        enum_cls = self.declared_type
        entries: Dict[str, Dict[str, Any]] = {}

        if self.value is None:
            for member in enum_cls:
                entries[f"{name}_{member.name}"] = {"value": None, "type": bool}
            return entries

        # value is not None: we already validated it's an instance of declared_type
        for member in enum_cls:
            entries[f"{name}_{member.name}"] = {"value": (member is self.value), "type": bool}
        return entries

    def add_to_row_as_one_hot(self, row: Dict[str, Any], name: str) -> None:
        """Update the row with one-hot entries derived from this Enum feature."""
        row.update(self.to_one_hot_entries(name))