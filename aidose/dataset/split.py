from .attribute import AttributesList
from typing import List, Any, Tuple, Callable
from datetime import datetime
import random


class ListSplitter:
    def __init__(self, split_proportions: Tuple[float, float, float]):
        if not abs(sum(split_proportions) - 1.0) < 1e-6:
            raise ValueError("Split proportions must sum to 1.0")
        self.split_proportions = split_proportions

    def get_split_indices(
            self,
            data: List[Any],
            key: Callable[[Any], Any] | None = None,
            seed: int | None = None
    ) -> Tuple[List[int], List[int], List[int]]:
        n = len(data)
        indices = list(range(n))

        if key is not None:
            indices.sort(key=lambda i: key(data[i]))
        else:
            if seed is not None:
                random.seed(seed)
            random.shuffle(indices)

        n_train = int(n * self.split_proportions[0])
        n_valid = int(n * self.split_proportions[1])

        return (
            indices[:n_train],
            indices[n_train:n_train + n_valid],
            indices[n_train + n_valid:]
        )

    @staticmethod
    def get_index_of_intended_field(row: AttributesList, field_name: str) -> int:
        for i, attr in enumerate(row):
            if attr.name == field_name:
                if not isinstance(attr.value, datetime):
                    raise TypeError(
                        f"Expected a datetime object for field '{field_name}', but got {type(attr.value).__name__}"
                    )
                return i
        raise ValueError(f"The attribute with name '{field_name}' was not found in AttributesList")

    @staticmethod
    def chronological_key(
            rows: List['AttributesList'],
            date_field_name: str
    ) -> Callable[['AttributesList'], datetime]:

        date_field_index = ListSplitter.get_index_of_intended_field(rows[0], date_field_name)

        def key(row: 'AttributesList') -> datetime:
            value = row[date_field_index].value
            if not isinstance(value, datetime):
                raise TypeError(
                    f"Expected a datetime object for field '{date_field_name}', but got {type(value).__name__}"
                )
            return value if value is not None else datetime.max

        return key
