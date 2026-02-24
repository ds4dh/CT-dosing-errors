from transformers import Trainer
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from typing import List, Iterator, Dict
import argparse


class CustomTrainer(Trainer):
    def __init__(self, config: argparse.Namespace, *args, **kwargs):
        """
        Trainer subclass that can build *class-balanced* training batches.

        When `config.negative_sampling_ratio` is not ``None``, the standard training sampler used by Transformers is replaced with a `BalancedBatchSampler`
        that constructs each batch with a fixed proportion of positive vs. negative samples. This is useful for imbalanced binary classification.

        Parameters
        ----------
        *args, **kwargs
            Forwarded to `transformers.Trainer`.
        config : argparse.Namespace
            An `argparse.Namespace` stored on `self.config` and expected
            to expose:
            - `negative_sampling_ratio` : Optional[float]
                If `None`, use the default Trainer dataloader. If a float in (0, 1],
                interpreted as the desired **positive** fraction per batch (passed as
                `positive_ratio` to `BalancedBatchSampler`).
        """

        super().__init__(*args, **kwargs)
        self.config = config

    def get_train_dataloader(self) -> DataLoader:
        """
        Build the training dataloader, optionally using a class-balanced sampler.

        If `self.config.negative_sampling_ratio` is `None`, this defers to `Trainer.get_train_dataloader()` (standard shuffled batches).
        Otherwise, it returns a `DataLoader` that draws batches from `BalancedBatchSampler`, enforcing the requested positive fraction.

        Returns
        -------
        torch.utils.data.DataLoader
            A dataloader over `self.train_dataset`. When balanced sampling is enabled, the dataloader uses `batch_sampler=BalancedBatchSampler`
            and does **not** apply an additional `sampler`/`shuffle`.
        """
        if self.config.negative_sampling_ratio is None:
            return super().get_train_dataloader()

        print(f"✅ Using BalancedBatchSampler with positive ratio: {self.config.negative_sampling_ratio}")

        sampler = BalancedBatchSampler(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            positive_ratio=self.config.negative_sampling_ratio
        )

        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )


class BalancedBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: Dataset, batch_size: int, positive_ratio: float):
        """
        Class-balanced batch sampler for imbalanced datasets.

        An epoch is defined as one complete pass over all positive samples.
        At the start of each epoch, positive and negative indices are shuffled.
        Negatives are drawn without replacement; when the negative pool is exhausted mid-epoch, it is reshuffled and sampling continues. Each batch
        contains a fixed positive fraction determined by `positive_ratio`.

        Parameters
        ----------
        dataset : Dataset
            Dataset exposing a `labels` sequence (e.g., `dataset.labels[i]`) with
            targets, where `label > 0` denotes a positive and `label == 0` denotes a negative.
        batch_size : int
            Number of samples per batch (per device).
        positive_ratio : float
            Desired fraction of positives in each batch; must satisfy `0 < r < 1`.
        """

        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio

        # check the positive_ratio
        if not (0 < positive_ratio < 1):
            raise ValueError("positive_ratio must be between 0 and 1.")

        # look for positive and negative indices
        self.positive_indices = [i for i, label in enumerate(self.dataset.labels) if label > 0]
        self.negative_indices = [i for i, label in enumerate(self.dataset.labels) if label == 0]

        if not self.positive_indices:
            raise ValueError("No positive samples found in the dataset.")
        if not self.negative_indices:
            raise ValueError("No negative samples found in the dataset.")

        # compute num of positive/negative samples per batch
        self.num_positives_per_batch = max(1, round(self.batch_size * self.positive_ratio))
        self.num_negatives_per_batch = self.batch_size - self.num_positives_per_batch

        if self.num_negatives_per_batch > len(self.negative_indices):
            raise ValueError(f"Not enough negative samples to fill a batch.")

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield batches of indices matching the target positive ratio.

        Returns
        -------
        Iterator[list[int]]
            Batches of length `batch_size` (final batch may be smaller).
        """
        # Shuffle both positive and negative indices at the start of each epoch
        np.random.shuffle(self.positive_indices)
        np.random.shuffle(self.negative_indices)

        # Create iterators/pointers for both lists
        pos_iter = iter(self.positive_indices)
        neg_iter = iter(self.negative_indices)

        # Calculate the number of batches for one epoch
        num_batches = (len(self.positive_indices) + self.num_positives_per_batch - 1) // self.num_positives_per_batch

        for _ in range(num_batches):
            positive_batch = []
            for _ in range(self.num_positives_per_batch):
                try:
                    positive_batch.append(next(pos_iter))
                except StopIteration:
                    # This handles the last batch if it's smaller
                    break

            negative_batch = []
            for _ in range(self.num_negatives_per_batch):
                try:
                    negative_batch.append(next(neg_iter))
                except StopIteration:
                    # Negative pool is exhausted. Reshuffle and start over.
                    print("\n[Sampler Info] Negative samples exhausted; reshuffling negatives.\n")
                    np.random.shuffle(self.negative_indices)
                    neg_iter = iter(self.negative_indices)
                    negative_batch.append(next(neg_iter))

            # Combine and shuffle the batch
            batch = positive_batch + negative_batch
            np.random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """
        Number of batches per epoch.

        Returns
        -------
        int
        """
        return (len(self.positive_indices) + self.num_positives_per_batch - 1) // self.num_positives_per_batch