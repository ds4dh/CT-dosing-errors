from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Set, Callable
import os


class MedDRALevel(str, Enum):
    SOC = "SOC"
    HLGT = "HLGT"
    HLT = "HLT"
    PT = "PT"
    LLT = "LLT"

    @classmethod
    def ordered_levels(cls) -> List["MedDRALevel"]:
        # Defined inside the method to avoid enum-member confusion
        return [cls.SOC, cls.HLGT, cls.HLT, cls.PT, cls.LLT]

    def parent(self) -> "MedDRALevel":
        idx = self.ordered_levels().index(self)
        return self.ordered_levels()[idx - 1] if idx > 0 else self

    def child(self) -> "MedDRALevel":
        idx = self.ordered_levels().index(self)
        return self.ordered_levels()[idx + 1] if idx < len(self.ordered_levels()) - 1 else self

    def is_above(self, other: "MedDRALevel") -> bool:
        return self.ordered_levels().index(self) < self.ordered_levels().index(other)

    def is_below(self, other: "MedDRALevel") -> bool:
        return self.ordered_levels().index(self) > self.ordered_levels().index(other)

    @classmethod
    def from_str(cls, value: str) -> "MedDRALevel":
        for level in cls:
            if level.name.upper() == value.upper():
                return level
        raise ValueError(f"Unknown MedDRALevel: {value}")

    def __str__(self) -> str:
        return self.value


@dataclass
class Node:
    code: str
    term: str
    level: MedDRALevel
    parents: Set["Node"] = field(default_factory=set)

    def __hash__(self):
        return hash((self.code, self.level))

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        return self.code == other.code and self.level == other.level


class MedDRA:
    def __init__(self) -> None:
        self.nodes: Dict[Tuple[MedDRALevel, str], Node] = {}

    def add_node(self, code: str, term: str, level: MedDRALevel) -> None:
        key = (level, code)
        if key not in self.nodes:
            self.nodes[key] = Node(code, term, level)

    def add_edge(self, code1: str, level1: MedDRALevel, code2: str, level2: MedDRALevel) -> None:
        key1 = (level1, code1)
        key2 = (level2, code2)
        if key1 in self.nodes and key2 in self.nodes:
            self.nodes[key2].parents.add(self.nodes[key1])

    def find_paths(self, code: str, level: MedDRALevel, pad_levels: bool = True) -> List[List[str]]:
        key = (level, code)
        if key not in self.nodes:
            return []

        node = self.nodes[key]
        if not node.parents:
            return [[f"{node.code}@{node.level}"]]

        paths = []
        for parent in node.parents:
            parent_paths = self.find_paths(parent.code, parent.level, pad_levels)
            for path in parent_paths:
                path_with_padding = []
                if pad_levels:
                    parent_index = MedDRALevel.ordered_levels().index(parent.level)
                    current_index = MedDRALevel.ordered_levels().index(node.level)
                    padding_levels = current_index - parent_index - 1

                    for i in range(padding_levels):
                        padding_level = MedDRALevel.ordered_levels()[parent_index + i + 1]
                        path_with_padding.append(f"BYPASS@{padding_level}")

                paths.append(path + path_with_padding + [f"{node.code}@{node.level}"])

        return paths

    def find_node_by_term(
            self,
            term: str,
            levels: Set[MedDRALevel] | None = None,
            preprocess: Callable[[str], str] | None = None,
    ) -> List[Node]:
        if preprocess:
            term = preprocess(term)

        matches = []
        for node in self.nodes.values():
            node_term = preprocess(node.term) if preprocess else node.term
            if node_term == term and (levels is None or node.level in levels):
                matches.append(node)
        return matches

    def load_data(self, path: str) -> None:
        # Create nodes from MedDRA .asc files
        for filename, level in [
            ("soc.asc", MedDRALevel.SOC),
            ("hlgt.asc", MedDRALevel.HLGT),
            ("hlt.asc", MedDRALevel.HLT),
            ("pt.asc", MedDRALevel.PT),
            ("llt.asc", MedDRALevel.LLT),
        ]:
            with open(os.path.join(path, filename), "r") as file:
                for line in file:
                    parts = line.strip().split("$")
                    code, term = parts[0], parts[1]
                    self.add_node(code, term, level)

        # Create explicit parent-child edges
        for filename, parent_level, child_level in [
            ("soc_hlgt.asc", MedDRALevel.SOC, MedDRALevel.HLGT),
            ("hlgt_hlt.asc", MedDRALevel.HLGT, MedDRALevel.HLT),
            ("hlt_pt.asc", MedDRALevel.HLT, MedDRALevel.PT),
        ]:
            with open(os.path.join(path, filename), "r") as file:
                for line in file:
                    parent_code, child_code = line.strip().split("$")[:2]
                    self.add_edge(parent_code, parent_level, child_code, child_level)

        # Handle indirect parent mappings (e.g., PT ➝ SOC, etc.)
        for filename, child_level in [("pt.asc", MedDRALevel.PT), ("llt.asc", MedDRALevel.LLT)]:
            with open(os.path.join(path, filename), "r") as file:
                for line in file:
                    parts = line.strip().split("$")
                    code = parts[0]
                    parent_code = parts[2] if child_level == MedDRALevel.LLT else parts[3]

                    above_levels = (
                        [MedDRALevel.PT, MedDRALevel.HLT, MedDRALevel.HLGT, MedDRALevel.SOC]
                        if child_level == MedDRALevel.LLT
                        else [MedDRALevel.HLT, MedDRALevel.HLGT, MedDRALevel.SOC]
                    )
                    for parent_level in above_levels:
                        if (parent_level, parent_code) in self.nodes:
                            self.add_edge(parent_code, parent_level, code, child_level)
