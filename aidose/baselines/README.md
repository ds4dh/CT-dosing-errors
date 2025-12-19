# Baselines

## 1. Overview

This repository contains the codebase used to train and evaluate baseline models reported in the paper called `Establishing a benchmark for the prediction of dosing
errors in interventional clinical trials`. Specifically, the following models are considered here

- **ClinicalModernBERT** — a BERT style model that consider only text features
- **XGBoost** — an XGBoost that uses only categorical features
- **LateFusionModelMultimodal** — a basic multimodal baseline that combines `ClinicalModernBERT` and `XGBoost` predictions

---

## 2. Repository Structure

The repository is organized as follows:

* **`main.py`** — Main entry point to train and evaluate all baselines.
* **`constant.py`** — Contains fixed variable names (e.g., folder paths).
* **`construct_hyperparameter_search.py`** — Contains functions that construct the hyperparameter searching space for
  the specific model and task. This searching space is used by Optuna during hyperparameter search. Specifically, it is used by the `OurXGBoost` and `LateFusionModel` models.
* **`CustomTrainer.py`** — Contains two classes used by `OurClinicalModernBERT`:

    * **`CustomTrainer`** — A Hugging Face `Trainer` subclass that can build *class-balanced* training batches.
    * **`BalancedBatchSampler`** — Class-balanced batch sampler for imbalanced datasets, used by `CustomTrainer`.
* **`DosingErrorDataset.py`** — Defines `DosingErrorDataset`, a minimal dataset wrapper for ModernBERT-style sequence
  tasks (classification or regression). Padding is handled later by a data collator.
* **`our_clinicalModernBERT.py`** — Implements `OurClinicalModernBERT`, a ClinicalModernBERT-based model that operates **exclusively on text features**.
* **`our_xgboost.py`** — Implements `OurXGBoost`, an XGBoost baseline that operates **exclusively on categorical
  features**.
* **`preprocessing.py`** — Functions to preprocess the AIDosE dataset before feeding it to the different models.
* **`utils.py`** — Utility functions.



