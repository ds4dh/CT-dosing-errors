import os
import pickle
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import *                      # binary_metrics, f1_score, TQDMProgressBar
from constants import LR_DIR             # add e.g. LR_DIR = os.path.join(RESULTS_DIR, "LogisticRegression")

optuna.logging.set_verbosity(optuna.logging.WARNING)


class OurLogisticRegression:

    def __init__(self, param: argparse.Namespace, dataset: dict[str, pd.DataFrame], logdir=None,
                 features: list[str] | None = None, id_col: str = "METADATA_nctId"):
        """
        Initialize a logistic regression baseline adapted to our dataset.

        Same structure as OurXGBoost. Unlike XGBoost, logistic regression needs explicit
        preprocessing: missing values are imputed (with missingness indicators), numerical
        features are standardized, and categorical features are one-hot encoded.

        Parameters
        ----------
            param: general config (see main.py)
            dataset: dictionary with the keys "train", "validation", "test" and "test_nctid"
                     (i.e. dataset['categorical_data'], as passed to OurXGBoost).
            logdir: optional output directory.
            features: optional subset of feature columns (e.g. a few key structured covariates).
                      If None, all structured features are used.
        """
        self.param = param
        self.features = features
        self.id_col = id_col
        self.best_threshold = None

        # trial identifiers of the test set, stored separately from the features
        # (same convention as the late-fusion model: dataset['test_nctid']["METADATA_nctId"])
        self.test_ids = np.asarray(dataset['test_nctid'][id_col])

        self.X_train, self.y_train = self._split_feature_label(dataset['train'])
        self.X_val, self.y_val = self._split_feature_label(dataset['validation'])
        self.X_test, self.y_test = self._split_feature_label(dataset['test'])

        self._log_dir = logdir if logdir is not None else os.path.join(LR_DIR, self.param.label)
        os.makedirs(self._log_dir, exist_ok=True)

        self.model = self._build_pipeline(C=1.0)

    # ------------------------------------------------------------------------------------------
    # public API (mirrors OurXGBoost)
    # ------------------------------------------------------------------------------------------
    def hyperparameter_search_and_evaluation(self) -> None:
        """Run the Optuna search, refit with the best hyperparameters, evaluate, and save them."""
        study = self._hyperparameter_search()
        self.model = self._build_pipeline(**study.best_params)
        self.model.fit(self.X_train, self.y_train)
        self._model_evaluation()
        self._save_hyperparameters(study=study)

    def load_and_evaluate(self, with_calibration=True) -> None:
        """Load the best hyperparameters, retrain, evaluate, and optionally calibrate."""
        self.model = self._build_pipeline(**self._load_hyperparameters())
        self.model.fit(self.X_train, self.y_train)

        if with_calibration:
            print('#' * 50)
            print('BEFORE CALIBRATION')
            print('#' * 50)
        self._model_evaluation()

        if with_calibration:
            self._calibration()
            print('#' * 50)
            print('AFTER MODEL CALIBRATION')
            print('#' * 50)
            self._model_evaluation()

        self.save_test_predictions()

    def predict_all_splits(self, datasets=None):
        """Probability predictions for train/validation/test, or validation only."""
        if datasets is None:
            return (self.model.predict_proba(self.X_train), self.model.predict_proba(self.X_val),
                    self.model.predict_proba(self.X_test))
        elif datasets == 'validation':
            return self.model.predict_proba(self.X_val)

    def save_test_predictions(self, filename="test_predictions.csv") -> str:
        """
        Save test-set predictions to <logdir>/<filename> with columns NCTID, proba_pred, pred, label.
        `pred` uses the F1-optimal threshold selected on the validation set by the last evaluation.
        """
        if self.best_threshold is None:
            raise RuntimeError("Run load_and_evaluate() or hyperparameter_search_and_evaluation() first.")
        proba = self.model.predict_proba(self.X_test)[:, 1]
        out = pd.DataFrame({
            "NCTID": self.test_ids,
            "proba_pred": proba,
            "pred": (proba >= self.best_threshold).astype(int),
            "label": self.y_test.to_numpy(),
        })
        path = os.path.join(self._log_dir, filename)
        out.to_csv(path, index=False)
        print(f"Test predictions saved to {path}")
        return path

    # ------------------------------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------------------------------
    def _split_feature_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Separate the label column 'LABEL_<param.label>' from the features (optionally a subset)."""
        label = df['LABEL_' + self.param.label]
        feature = df.drop(columns=['LABEL_' + self.param.label])
        if self.id_col in feature.columns:          # identifiers must never be used as features
            feature = feature.drop(columns=[self.id_col])
        if self.features is not None:
            feature = feature[self.features]
        bool_cols = feature.select_dtypes(include="bool").columns
        feature[bool_cols] = feature[bool_cols].astype(int)
        return feature, label

    def _build_pipeline(self, C: float) -> Pipeline:
        """Imputation + scaling / one-hot encoding + logistic regression."""
        cat_cols = self.X_train.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = [c for c in self.X_train.columns if c not in cat_cols]

        preprocess = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
        ])
        # L2-regularized (scikit-learn default); class_weight mirrors XGBoost's scale_pos_weight
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                 random_state=self.param.random_seed)
        return Pipeline([("preprocess", preprocess), ("clf", clf)])

    def _hyperparameter_search(self) -> optuna.study.Study:
        """Optuna TPE search over the regularization strength C, maximizing validation AUROC."""

        def objective(trial):
            params = {"C": trial.suggest_float("C", 1e-4, 1e2, log=True)}
            model = self._build_pipeline(**params)
            model.fit(self.X_train, self.y_train)
            return roc_auc_score(self.y_val, model.predict_proba(self.X_val)[:, 1])

        print("We begin the hyperparameter search using Optuna.")
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=self.param.random_seed))
        progress_bar = TQDMProgressBar(self.param.num_trials, desc="Hyperparameter search")
        study.optimize(objective, n_trials=self.param.num_trials, callbacks=[progress_bar])
        progress_bar.close()
        return study

    def _calibration(self) -> None:
        """Isotonic calibration on the validation set (post hoc; same procedure as XGBoost)."""
        if self.param.label != "wilson_label":
            raise NotImplementedError("Calibration is implemented only for 'wilson_label'.")
        calibrator = CalibratedClassifierCV(FrozenEstimator(self.model), method="isotonic")
        calibrator.fit(self.X_val, self.y_val)
        self.model = calibrator

    def _load_hyperparameters(self) -> dict:
        with open(os.path.join(self._log_dir, "lr_best_hyperparam.pkl"), "rb") as f:
            return pickle.load(f)

    def _save_hyperparameters(self, study: optuna.Study) -> None:
        with open(os.path.join(self._log_dir, "lr_best_hyperparam.pkl"), "wb") as f:
            pickle.dump(study.best_params, f)
        print("Best logistic regression hyperparameters have been saved.")

    def _model_evaluation(self):
        """Same evaluation as OurXGBoost: F1-optimal threshold on validation, metrics on all splits."""
        print('#' * 50)
        print("Best model evaluation.")
        print('#' * 50)

        val_proba = self.model.predict_proba(self.X_val)[:, 1]
        best_threshold, best_f1 = None, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            f1 = f1_score(self.y_val, (val_proba >= t).astype(int), average="binary", pos_label=1,
                          zero_division=0)
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(t)

        self.best_threshold = best_threshold
        print('#' * 50)
        print(f"Selected threshold on validation (max F1): {best_threshold:.3f} (F1={best_f1:.5f})")
        print('#' * 50)

        for name, X, y in [("training", self.X_train, self.y_train),
                           ("validation", self.X_val, self.y_val),
                           ("test", self.X_test, self.y_test)]:
            proba = self.model.predict_proba(X)[:, 1]
            metrics = binary_metrics(predictions=(proba >= best_threshold).astype(int),
                                     proba_predictions=proba, label=y, dataset=name)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.5f}")


