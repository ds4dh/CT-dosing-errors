from datasets import Dataset, DatasetDict
from typing import Optional
from datetime import datetime, date
import math


# TODO: I think this should be done simply on a list, rather than a nested `Dataset` object.
class DatasetSplit:
    def __init__(self, train_ratio: float, valid_ratio: float, test_ratio: float):
        total = train_ratio + valid_ratio + test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(f"Ratios must sum to 1.0, got {total} "
                             f"({train_ratio}, {valid_ratio}, {test_ratio})")
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def split_chronologically(
            self,
            ds: Dataset,
            date_path: str,  # e.g. "metadata.completionDate"
            drop_missing: bool = True,
    ) -> DatasetDict:
        """
        Sort by a nested date column (supports dot-path), then split:
          train = oldest, valid = middle, test = newest.
        Works with date32 (preferred) or timestamp[...] or datetime -> coerced to date.
        """
        parts = tuple(date_path.split("."))

        def _get_nested(ex):
            v = ex
            for p in parts:
                v = v[p]
            return v

        if drop_missing:
            ds = ds.filter(lambda ex: _get_nested(ex) is not None)

        # Add a temporary scalar column to sort by (avoid flattening the whole schema)
        def _add_sort_key(ex):
            v = _get_nested(ex)
            # Coerce datetime -> date if needed for consistent ordering at day precision
            if isinstance(v, datetime):
                v = v.date()
            # If it's a string like "YYYY-MM-DD", that's fine too; Arrow will sort lexicographically OK.
            if isinstance(v, date):
                return {"__sort_key": v}
            return {"__sort_key": v}

        ds = ds.map(_add_sort_key)
        ds = ds.sort("__sort_key")
        ds = ds.remove_columns(["__sort_key"])

        return self._slice(ds)

    def split_randomly(self, ds: Dataset, seed: Optional[int] = None) -> DatasetDict:
        # Use built-in train_test_split for reproducibility and speed, then split valid from train.
        # This keeps alignment without manual shuffling.
        if seed is None:
            seed = 0
        # First split off test
        test_size = self.test_ratio
        tmp = ds.train_test_split(test_size=test_size, seed=seed, shuffle=True)
        # Then split validation from the remaining train
        # validation proportion relative to remaining
        valid_rel = self.valid_ratio / (self.train_ratio + self.valid_ratio)
        tmp2 = tmp["train"].train_test_split(test_size=valid_rel, seed=seed, shuffle=True)
        return DatasetDict(
            train=tmp2["train"],
            validation=tmp2["test"],
            test=tmp["test"],
        )

    def _slice(self, ds: Dataset) -> DatasetDict:
        n = ds.num_rows
        t_n = int(math.floor(self.train_ratio * n))
        v_n = int(math.floor(self.valid_ratio * n))
        te_n = n - (t_n + v_n)

        train = ds.select(range(0, t_n))
        valid = ds.select(range(t_n, t_n + v_n))
        test = ds.select(range(t_n + v_n, t_n + v_n + te_n))
        return DatasetDict(train=train, validation=valid, test=test)
