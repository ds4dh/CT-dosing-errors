import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Dict, List
import transformers



class DosingErrorDataset(Dataset):

    def __init__(self, texts: List[str], labels: List[int] | list[float] , tokenizer: transformers.PreTrainedTokenizerBase, max_length: int, task: str, label_dtype: torch.dtype | None = None,):

        """
        Minimal dataset wrapper for ModernBERT-style sequence tasks (classification or regression).

        Each item returns tokenized inputs and a label. Tokenization is performed on-the-fly;
        padding is intentionally deferred to the collator (e.g., `DataCollatorWithPadding`)
        so that sequences are padded dynamically to the longest length in the batch.

        Parameters
        ----------
        texts : list[str]
            Raw input texts, one per example.
        labels : list[int] | list[float]
            Ground-truth labels aligned with `texts`. Cast to `torch.long` for classification and `torch.float` for regression.
        tokenizer : transformers.PreTrainedTokenizerBase
            Hugging Face tokenizer used to encode inputs.
        max_length : int
            Maximum sequence length; longer sequences are truncated.
        task : {"classification", "regression"}
            Determines the label dtype: `long` for classification, `float` for regression.
        """

        # define the task
        if task == 'regression':
            self.dtype = label_dtype if label_dtype is not None else torch.float32
        else:
            self.dtype = torch.long
    
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Encode a single example.

        Returns
        -------
        dict
            {
              "input_ids":       (seq_len,) torch.long,
              "attention_mask":  (seq_len,) torch.long,
              "labels":          () or (1,) torch.long/torch.float (per `task`)
            }

        """
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding=False,  # Padding is handled by DataCollator
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=self.dtype)
        }

    def __len__(self) -> int:
        """Number of samples in the dataset."""
        return len(self.labels)