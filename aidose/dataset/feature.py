from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Type


@dataclass(frozen=True)
class Feature:
    """
    A typed feature with a name.

    Fields:
      - name: feature name (arbitrary for scalar features; ignored by encoders)
      - value: the feature's value (may be None for "missing")
      - declared_type: a Python type (e.g., str, int, bool, or a concrete Enum subclass)

    Validation rules:
      - declared_type must be a type object (never a string), and NOT the Enum base class.
      - If value is None: accepted for any declared_type.
      - If declared_type is bool: only a real bool is accepted (not 0/1).
      - If declared_type is a concrete Enum subclass:
          value may be:
            * a single Enum member of that subclass, or
            * a list of Enum members of that subclass (for multi-hot use).
      - Otherwise: isinstance(value, declared_type) must be True.

    Encoding helpers:
      - as_one_hot(): for a **single** Enum value -> List[Feature] of bools; None -> all None
                      names are "<EnumClass>.<MEMBER>"
      - as_multi_hot(): for a single or list[Enum] -> List[Feature] of ints; None -> all None
                        names are "<EnumClass>.<MEMBER>"
    """
    name: str
    value: Any
    declared_type: Type[Any]

    # -------------------- validation --------------------

    def __post_init__(self) -> None:
        # declared_type must be a real type
        if not isinstance(self.declared_type, type):
            raise TypeError("declared_type must be a type object (e.g., str, int, bool, or a concrete Enum subclass).")

        # disallow Enum base
        if self.declared_type is Enum:
            raise TypeError("declared_type must be a concrete Enum subclass, not Enum.")

        # missing is always allowed
        if self.value is None:
            return

        # strict bool (no ints)
        if self.declared_type is bool:
            if type(self.value) is not bool:
                raise TypeError(f"Feature '{self.name}' expects value of type bool, got {type(self.value).__name__}")
            return

        # Enum handling
        if issubclass(self.declared_type, Enum):
            # single Enum member
            if isinstance(self.value, self.declared_type):
                return

            # list of Enum / None — allow all-None; allow all Enum; reject mixed
            if isinstance(self.value, list):
                if len(self.value) == 0:
                    # empty list treated as "all None"
                    return
                if all(elem is None for elem in self.value):
                    return
                if all(isinstance(elem, self.declared_type) for elem in self.value):
                    return
                raise TypeError(
                    f"Feature '{self.name}' list must contain all {self.declared_type.__name__} "
                    f"or all None, got mixed types."
                )

            # anything else is invalid for Enum declared_type
            raise TypeError(
                f"Feature '{self.name}' expects {self.declared_type.__name__} or list[{self.declared_type.__name__}], "
                f"got {type(self.value).__name__}"
            )

        # General case
        if not isinstance(self.value, self.declared_type):
            raise TypeError(
                f"Feature '{self.name}' expects value of type {self.declared_type.__name__}, "
                f"got {type(self.value).__name__}"
            )

    # -------------------- basic exports --------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a full dict including the feature name."""
        return {"name": self.name, "value": self.value, "type": self.declared_type}

    # -------------------- encodings --------------------

    def as_one_hot(self) -> List[Feature]:
        """
        For **single** Enum value: produce one-hot features of type bool.
        Names are "<EnumClass>.<MEMBER>".

        If value is None -> all features have value=None (type=bool).
        If value is a list -> TypeError (use as_multi_hot instead).
        """
        if not (isinstance(self.declared_type, type) and issubclass(self.declared_type, Enum)):
            raise TypeError("as_one_hot requires declared_type to be a concrete Enum subclass.")

        enum_cls = self.declared_type

        if isinstance(self.value, list):
            raise TypeError("as_one_hot is only valid for a single Enum value. Use as_multi_hot for lists.")

        feats: List[Feature] = []
        if self.value is None:
            for member in enum_cls:
                feats.append(Feature(f"{self.name}.{member.name}", None, bool))
            return feats

        # single Enum (validated in __post_init__)
        for member in enum_cls:
            feats.append(Feature(f"{self.name}.{member.name}", member is self.value, bool))
        return feats

    def as_multi_hot(self) -> List[Feature]:
        """
        For a single Enum value or a list of Enum values: produce multi-hot counts (type=int).
        Names are "<EnumClass>.<MEMBER>".

        Behavior:
        - If value is None -> all features have value=None (type=int).
        - If value is a single Enum -> that member gets 1, others 0.
        - If value is a list:
            * If empty or all elements are None -> all features have value=None (type=int).
            * Otherwise -> counts reflect occurrences (duplicates allowed).
        """

        if not (isinstance(self.declared_type, type) and issubclass(self.declared_type, Enum)):
            raise TypeError("as_multi_hot requires declared_type to be a concrete Enum subclass.")

        enum_cls = self.declared_type
        feats: list[Feature] = []

        # value is None -> all None
        if self.value is None:
            for member in enum_cls:
                feats.append(Feature(f"{self.name}.{member.name}", None, int))
            return feats

        # single Enum -> 1 for that member, 0 for others
        if isinstance(self.value, self.declared_type):
            for member in enum_cls:
                feats.append(Feature(f"{self.name}.{member.name}", 1 if member is self.value else 0, int))
            return feats

        # list case
        if isinstance(self.value, list):
            # if empty or all None -> treat as unknown => all None
            if len(self.value) == 0 or all(elem is None for elem in self.value):
                for member in enum_cls:
                    feats.append(Feature(f"{self.name}.{member.name}", None, int))
                return feats

            # otherwise we already validated no Nones and all are correct Enum members
            counts: dict[Enum, int] = {}
            for elem in self.value:
                counts[elem] = counts.get(elem, 0) + 1
            for member in enum_cls:
                feats.append(Feature(f"{self.name}.{member.name}", counts.get(member, 0), int))
            return feats

        # should be unreachable due to __post_init__
        raise TypeError("Invalid enum value state for multi-hot.")


class FeaturesList(list[Feature]):
    def expand_enums(self) -> FeaturesList:
        expanded = FeaturesList()
        for f in self:
            if issubclass(f.declared_type, Enum):
                if isinstance(f.value, list):
                    expanded.extend(f.as_multi_hot())
                else:
                    expanded.extend(f.as_one_hot())
            else:
                expanded.append(f)
        return expanded

    def get_values(self) -> List:
        """Return a list of all feature values in order."""
        return [f.value for f in self]

    def get_types(self) -> List[type]:
        """Return a list of declared types for all features in order."""
        return [f.declared_type for f in self]

    def get_names(self) -> List[str]:
        """Return a list of feature names in order."""
        return [f.name for f in self]
