from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import torch
import pandas as pd
import ast
import numpy as np
import argparse
from typing import Tuple
import transformers
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.metrics import brier_score_loss



class TQDMProgressBar:
    """
    Just a small utility class to be able to add a progress bar when conducting hyperparameter search using Optuna
    """

    def __init__(self, n_trials, desc):
        self.pbar = tqdm(total=n_trials, desc=desc)

    def __call__(self, study, trial):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()



def binary_metrics_hf(pred: transformers.EvalPrediction) -> dict[str, float]:
    """
    Adapter for Hugging Face `Trainer.compute_metrics` in **binary classification**.

    Parameters
    ----------
    pred : transformers.EvalPrediction
        Container with:
        - `predictions` : np.ndarray
            Model outputs for the eval set. 
        - `label_ids` : np.ndarray
            Ground-truth labels (shape (n,) or (n, 1)).

    Returns
    -------
    dict[str, float]
        Metrics produced by `binary_metrics`,
    """
    proba_pred = torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()[:, 1]

    return binary_metrics(predictions=np.argmax(pred.predictions, axis=-1), proba_predictions=proba_pred, label=pred.label_ids)


def binary_metrics(predictions, proba_predictions, label, dataset: str | None = None, risk_bins: Tuple[float, float, float] = (0.02, 0.05, 0.10)) -> dict[str, float]:
    """
    Compute binary classification metrics with explicit probability thresholds.

    Threshold-independent metrics:
      - ROC-AUC
      - Brier Score
      - Precision
      - Recall
      - F1 (binary and macro average)
      - Accuracy and balanced accuracy

    Parameters
    ----------
    proba_predictions : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    label : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    dataset : str, optional
        If provided, prints a header.
    thresholds : tuple[float, ...]
        Probability thresholds to evaluate.

    Returns
    -------
    dict[str, float]
        Flat dictionary of metrics for easy printing/logging.
    """
    if dataset is not None:
        print('#' * 50)
        print("Performance on " + dataset + " dataset.")
        print('#' * 50)

    proba_predictions = np.asarray(proba_predictions, dtype=float)
    label = np.asarray(label, dtype=int)

    metrics: dict[str, float] = {
        "ROC-AUC": roc_auc_score(label, proba_predictions),
        "Brier Score": brier_score_loss(label, proba_predictions),
        "Precision": precision_score(label, predictions, pos_label=1, zero_division=0),
        "Recall": recall_score(label, predictions, pos_label=1, zero_division=0),
        "F1": f1_score(label, predictions, pos_label=1, average="binary", zero_division=0),
        "F1 Macro": f1_score(label, predictions, average="macro", zero_division=0),
        "Accuracy": accuracy_score(label, predictions),
        "Balanced Accuracy": balanced_accuracy_score(label, predictions),
    }

    # risk stratification
    prevalence = float(label.mean())
    metrics["Prevalence"] = prevalence
    proba = np.asarray(proba_predictions, dtype=float)

    t1, t2, t3 = risk_bins

    tier_defs = [
        ("Low",       (proba < t1)),
        ("Moderate",  ((proba >= t1) & (proba < t2))),
        ("High",      ((proba >= t2) & (proba < t3))),
        ("Very_High",  (proba >= t3)),
    ]

    for tier_name, mask in tier_defs:
        n = int(mask.sum())
        n_pos = int(label[mask].sum()) if n > 0 else 0
        event_rate = (n_pos / n) if n > 0 else np.nan
        rel_risk = (event_rate / prevalence) if (n > 0 and prevalence > 0) else np.nan

        metrics[f"{tier_name}.n"] = float(n)
        metrics[f"{tier_name}.n_pos"] = float(n_pos)
        metrics[f"{tier_name}.event_rate"] = float(event_rate) if not np.isnan(event_rate) else np.nan
        metrics[f"{tier_name}.relative_risk"] = float(rel_risk) if not np.isnan(rel_risk) else np.nan

    return metrics


def compute_batch_size(param: argparse.Namespace) -> Tuple[int, int]:
    """
    Compute **per-device** train/eval batch sizes from global settings, validating divisibility.


    Parameters
    ----------
    param : argparse.Namespace
        Must define:
        - `train_batch_size` (int): global training batch size (effective batch).
        - `eval_batch_size`  (int): global evaluation batch size.
        - `gradient_accumulation_step` (int): accumulation steps.

    Returns
    -------
    tuple[int, int]
        `(train_batch_per_device, eval_batch_per_device)`
    """

    num_gpus = torch.cuda.device_count()

    # training batch
    train_denominator = num_gpus * param.gradient_accumulation_step
    if param.train_batch_size % train_denominator != 0:
        raise ValueError(
            f"Invalid configuration: train_batch_size ({param.train_batch_size}) "
            f"is not divisible by the product of num_gpus ({num_gpus}) and "
            f"GRADIENT_ACCUMULATION_STEPS ({param.gradient_accumulation_step}), which is {train_denominator}. "
            f"Please choose a GLOBAL_TRAIN_BATCH_SIZE that is a multiple of {train_denominator}."
        )
    train_batch_per_device = param.train_batch_size // train_denominator

    # evaluation batch
    eval_denominator = num_gpus
    if param.eval_batch_size % eval_denominator != 0:
        raise ValueError(
            f"Invalid configuration: eval_batch_size ({param.eval_batch_size}) "
            f"is not divisible by num_gpus ({num_gpus}). "
            f"Please choose a eval_batch_size that is a multiple of {num_gpus}."
        )
    
    eval_batch_per_device = param.eval_batch_size // eval_denominator

    print(f"Discovered {num_gpus} GPUs.")
    print(f"Global Train Batch Size: {param.train_batch_size}")
    print(f"Gradient Accumulation Steps: {param.gradient_accumulation_step}")
    print(f"=> Computed Per-Device Train Batch Size: {train_batch_per_device}")
    print(f"Global Eval Batch Size: {param.eval_batch_size}")
    print(f"=> Computed Per-Device Eval Batch Size: {eval_batch_per_device}")

    return train_batch_per_device, eval_batch_per_device


def create_one_global_text_feature(row: pd.Series, param: argparse.Namespace) -> str:
    """
    Build a single Markdown-style text field by concatenating all **non-label** columns
    from a DataFrame row.

    For each column `col` in `row` except `param.label`, this function appends a section:
    `## {col}\n\n{value}\n\n`. Sections are emitted in the same order as the row’s columns.

    Parameters
    ----------
    row : pandas.Series
        A single record from the dataset. Keys are column names; values are the row entries.
    param : argparse.Namespace
        Configuration object with at least `label` (str), the name of the target column to skip.

    Returns
    -------
    str
        Concatenated Markdown text of the form:
        "## col1\\n\\nvalue1\\n\\n## col2\\n\\nvalue2\\n\\n...".

    """

    global_text = []
    for col, value in row.items():
        if col != f"LABEL_{param.label}":
            text_to_print = col.removeprefix("FEATURE_")
            global_text.append(f"## {text_to_print}\n\n{value}\n\n")

    return "".join(global_text)


def logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Numerically-stable logit transform."""
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))

