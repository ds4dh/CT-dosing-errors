from datasets import Dataset, DatasetDict
from typing import Tuple, Optional
import math
import random

class DatasetSplit:
    """
    Utility for splitting Hugging Face Datasets into train/validation/test sets.

    Args:
        train_ratio (float): Proportion for training set.
        valid_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
    """

    def __init__(self, train_ratio: float, valid_ratio: float, test_ratio: float):
        total = train_ratio + valid_ratio + test_ratio
        if not math.isclose(total, 1.0, rel_tol=1e-9):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total:.4f} "
                f"({train_ratio}, {valid_ratio}, {test_ratio})"
            )
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

    def split_chronologically(
        self,
        ds: Dataset,
        date_col: str,
    ) -> DatasetDict:
        """
        Split dataset based on chronological order of a date column.
        Oldest → train, middle → valid, newest → test.
        """

        ds = ds.sort(date_col)
        return self._slice(ds)

    def split_randomly(
        self,
        ds: Dataset,
        seed: Optional[int] = None,
    ) -> DatasetDict:
        """
        Random split into train/valid/test sets.
        """
        if seed is not None:
            random.seed(seed)

        n = ds.num_rows
        indices = list(range(n))
        random.shuffle(indices)

        return self._slice(ds, indices)

    def _slice(self, ds: Dataset, indices: Optional[list] = None) -> DatasetDict:
        """
        Helper to slice dataset into train/valid/test subsets
        using either ordered or shuffled indices.
        """
        n = ds.num_rows
        t_n = int(math.floor(self.train_ratio * n))
        v_n = int(math.floor(self.valid_ratio * n))
        te_n = n - (t_n + v_n)

        if indices is None:
            train_idx = range(0, t_n)
            valid_idx = range(t_n, t_n + v_n)
            test_idx = range(t_n + v_n, t_n + v_n + te_n)
        else:
            train_idx = indices[0:t_n]
            valid_idx = indices[t_n:t_n + v_n]
            test_idx = indices[t_n + v_n:t_n + v_n + te_n]

        return DatasetDict(
            train=ds.select(train_idx),
            validation=ds.select(valid_idx),
            test=ds.select(test_idx),
        )
