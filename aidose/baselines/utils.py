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


def regression_metrics_hf(pred: transformers.EvalPrediction) -> dict[str, float]:
    """
    Adapter for Hugging Face `Trainer.compute_metrics`.

    Parameters
    ----------
    pred : transformers.EvalPrediction
        Contains:
        - predictions : np.ndarray
            Model outputs for the eval set (for regression typically shape (n,) or (n, 1)).
        - label_ids : np.ndarray
            Ground-truth labels (shape (n,) or (n, 1)).

    Returns
    -------
    dict[str, float]
        Regression metrics as returned by `regression_metrics`.
    """

    preds = pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    labels = np.asarray(pred.label_ids, dtype=np.float32).reshape(-1)
    return regression_metrics(predictions=preds, label=labels)



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


def regression_metrics(predictions, label, dataset: str | None = None) -> dict[str, float]:
    """
    Compute standard regression metrics: RMSE, MAE, and R².

    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        Model-predicted continuous values.
    label : array-like of shape (n_samples,)
        Ground-truth continuous values.
    dataset : str, optional
        If provided, prints a header (e.g., "Performance on <dataset> dataset.").


    Returns
    -------
    dict[str, float]
        A mapping with keys:
        - "RMSE": root mean squared error
        - "MAE" : mean absolute error
        - "R2"  : coefficient of determination
    """

    if dataset is not None:
        print('#' * 50)
        print("Performance on " + dataset + " dataset.")
        print('#' * 50)

    metrics = {
        "RMSE": root_mean_squared_error(label, predictions),
        "MAE": mean_absolute_error(label, predictions),
        "R2": r2_score(label, predictions)
    }
    return metrics


def binary_metrics(predictions, proba_predictions, label, dataset: str | None = None) -> dict[str, float]:
    """
    Compute standard metrics for a binary classification task.

    Calculates F1-score, precision, recall, and accuracy with ``pos_label=1``.
    Expects hard predictions in {0, 1}.

    Parameters
    ----------
    predictions : array-like of shape (n_samples,)
        Model-predicted binary labels (0 or 1).
    label : array-like of shape (n_samples,)
        Ground-truth binary labels (0 or 1).
    dataset : str, optional
        If provided, prints a header (e.g., "Performance on <dataset> dataset.").

    Returns
    -------
    dict[str, float] containing performances metrics

    """

    if dataset is not None:
        print('#' * 50)
        print("Performance on " + dataset + " dataset.")
        print('#' * 50)

    metrics = {
        'ROC-AUC': roc_auc_score(label, proba_predictions),
        "F1 Macro": f1_score(label, predictions, pos_label=1, average='macro'),
        'Balanced Accuracy': balanced_accuracy_score(label, predictions),
        "F1 Score": f1_score(label, predictions, pos_label=1, average='binary'),
        "Precision": precision_score(label, predictions, pos_label=1, average='binary'),
        "Recall": recall_score(label, predictions, pos_label=1, average='binary'),
        "Accuracy": accuracy_score(label, predictions)
    }

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
        if col != param.label:
            global_text.append(f"## {col}\n\n{value}\n\n")

    return "".join(global_text)