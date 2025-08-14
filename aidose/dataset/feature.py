# aidose/dataset/feature.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Type, List


@dataclass(frozen=True)
class Feature:
    """
    A typed feature wrapper with strict validation and Enum helpers.

    Fields:
      - value: the feature's value (may be None for "missing").
      - declared_type: a Python type (e.g., str, int, bool, or a concrete Enum subclass).

    Validation rules:
      - declared_type must be a type object (never a string), and NOT the Enum base class.
      - If value is None -> allowed for any declared_type.
      - If declared_type is bool -> value must be an actual bool (not 0/1).
      - If declared_type is a concrete Enum subclass:
          value may be:
            * a single Enum member of that subclass, or
            * a list of Enum members of that subclass.
      - Otherwise -> isinstance(value, declared_type) must be True.

    Enum helpers:
      - to_one_hot_entries(name): for a single Enum value; if value is None -> all None; if value is a list -> TypeError.
      - to_multi_hot_entries(name): for a list (or single) Enum value; counts per member; None -> all None.
    """
    value: Any
    declared_type: Type[Any]

    def __post_init__(self) -> None:
        # Ensure declared_type is a type
        if not isinstance(self.declared_type, type):
            raise TypeError("declared_type must be a type object (e.g., str, int, bool, or a concrete Enum subclass).")

        # Disallow the Enum base class as declared type
        if self.declared_type is Enum:
            raise TypeError("declared_type must be a concrete Enum subclass, not Enum.")

        # Missing is always allowed
        if self.value is None:
            return

        # Strict bool: exclude ints
        if self.declared_type is bool:
            if type(self.value) is not bool:
                raise TypeError(f"Feature expects value of type bool, got {type(self.value).__name__}")
            return

        # Enum family: allow single Enum OR list[Enum]
        if issubclass(self.declared_type, Enum):
            # Single Enum
            if isinstance(self.value, self.declared_type):
                return
            # List of Enum members
            if isinstance(self.value, list):
                for i, elem in enumerate(self.value):
                    if not isinstance(elem, self.declared_type):
                        raise TypeError(
                            f"Element at index {i} is not {self.declared_type.__name__}: got {type(elem).__name__}"
                        )
                return
            # Otherwise, not acceptable
            raise TypeError(
                f"Feature expects {self.declared_type.__name__} or list[{self.declared_type.__name__}], "
                f"got {type(self.value).__name__}"
            )

        # General case
        if not isinstance(self.value, self.declared_type):
            raise TypeError(
                f"Feature expects value of type {self.declared_type.__name__}, got {type(self.value).__name__}"
            )

    # ---------- basic row helpers ----------

    def to_dict(self) -> Dict[str, Any]:
        return {"value": self.value, "type": self.declared_type}

    def add_to_row(self, row: Dict[str, Any], name: str) -> None:
        row[name] = self.to_dict()

    # ---------- one-hot for single Enum ----------

    def to_one_hot_entries(self, name: str) -> Dict[str, Dict[str, Any]]:
        """
        Produce one-hot entries for a **single** Enum value.
        If value is None -> all entries have value=None.
        If value is a list -> raise TypeError (use to_multi_hot_entries instead).
        """
        if not (isinstance(self.declared_type, type) and issubclass(self.declared_type, Enum)):
            raise TypeError("to_one_hot_entries requires declared_type to be a concrete Enum subclass.")

        enum_cls = self.declared_type
        entries: Dict[str, Dict[str, Any]] = {}

        # list -> not allowed here (use multi-hot)
        if isinstance(self.value, list):
            raise TypeError("to_one_hot_entries is only valid for single Enum value, not a list. Use to_multi_hot_entries.")

        if self.value is None:
            for member in enum_cls:
                entries[f"{name}_{member.name}"] = {"value": None, "type": bool}
            return entries

        # Single Enum already validated
        for member in enum_cls:
            entries[f"{name}_{member.name}"] = {"value": (member is self.value), "type": bool}
        return entries

    def add_to_row_as_one_hot(self, row: Dict[str, Any], name: str) -> None:
        row.update(self.to_one_hot_entries(name))

    # ---------- multi-hot for Enum or list[Enum] ----------

    def to_multi_hot_entries(self, name: str) -> Dict[str, Dict[str, Any]]:
        """
        Produce multi-hot (counts) for Enum or list[Enum] values:
          { f"{name}_{MEMBER.name}": {"value": int|None, "type": int}, ... }

        - If value is None -> all counts None.
        - If value is a single Enum -> the matched member gets count 1, others 0.
        - If value is a list of Enum -> counts reflect occurrences (duplicates allowed).
        """
        if not (isinstance(self.declared_type, type) and issubclass(self.declared_type, Enum)):
            raise TypeError("to_multi_hot_entries requires declared_type to be a concrete Enum subclass.")

        enum_cls = self.declared_type
        entries: Dict[str, Dict[str, Any]] = {}

        if self.value is None:
            for member in enum_cls:
                entries[f"{name}_{member.name}"] = {"value": None, "type": int}
            return entries

        # Single Enum
        if isinstance(self.value, self.declared_type):
            for member in enum_cls:
                entries[f"{name}_{member.name}"] = {"value": 1 if member is self.value else 0, "type": int}
            return entries

        # List of Enum
        if isinstance(self.value, list):
            counts: Dict[Enum, int] = {}
            for elem in self.value:
                # already validated type
                counts[elem] = counts.get(elem, 0) + 1
            for member in enum_cls:
                entries[f"{name}_{member.name}"] = {"value": counts.get(member, 0), "type": int}
            return entries

        # Should not be reachable due to __post_init__ validation
        raise TypeError("Invalid enum value state for multi-hot.")

    def add_to_row_as_multi_hot(self, row: Dict[str, Any], name: str) -> None:
        row.update(self.to_multi_hot_entries(name))