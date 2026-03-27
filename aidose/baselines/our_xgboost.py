from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from optuna.samplers import TPESampler
import optuna
from utils import *
import os
from constants import XGB_DIR
import pickle
import argparse
import pandas as pd
from typing import Tuple
from construct_hyperparameter_search import construct_hyperparameter_search
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

# to suppress annoying optuna default logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OurXGBoost:

    def __init__(self, param: argparse.Namespace, dataset: dict[str, pd.DataFrame], logdir=None):
        """
        Initialize an XGBoost model specifically adapted to our dataset.

        This constructor:
          - Save the training, validation and test sets.
          - Creates the logging directory for outputs.
          - Initializes the appropriate XGBoost model

        Parameters
        ----------
            param: general config (see main.py)
            dataset: dictionary containing the three datasets with the keywords "training", "validation" and "test".

        Returns
        -------
        None
        """

        # general config
        self.param = param

        # training, validation, test sets
        self.X_train, self.y_train = self._split_feature_label(dataset['train'])
        self.X_val, self.y_val = self._split_feature_label(dataset['validation'])
        self.X_test, self.y_test = self._split_feature_label(dataset['test'])

        # log dir
        if logdir is None:
            self._log_dir = os.path.join(XGB_DIR, self.param.label)
            os.makedirs(self._log_dir, exist_ok=True)
        else:
            self._log_dir = logdir

        # initialization of the XGBoost model
        self.model = XGBClassifier()


    def hyperparameter_search_and_evaluation(self) -> None:
        """
        Perform hyperparameter optimization and evaluate the final model.

        This method:
            - Runs hyperparameter optimization with Optuna.
            - Trains an XGBoost model using the best hyperparameters found.
            - Evaluates the trained model on the test set.
            - Saves the best hyperparameters for future use.

        Returns
        -------
        None
        """

        # conduct hyperpameter search using Optuna
        study = self._hyperparameter_search()

        # train XGBoost model using the best hyperparemter
        best_param = study.best_params
        best_param.pop("sample_alpha", None)  # pop to avoid a warning
        self.model = XGBClassifier(**best_param) 

        self.model.fit(self.X_train, self.y_train)

        # evaluation
        self._model_evaluation()

        # hyperparameter saving
        self._save_hyperparameters(study=study)

    def load_and_evaluate(self, with_calibration=True) -> None:
        """
        Load the best hyperparameters, retrain the model, and evaluate it.

        Parameters
        ----------
        with_calibration : bool, optional
            Whether to apply calibration after training (default is True).

        Returns
        -------
        None
        """
        # loading model
        best_params = self._load_hyperparameters()

        # model initialization and training
        best_params.pop("sample_alpha", None)  # pop to avoid warning
        self.model = XGBClassifier(**best_params)

        self.model.fit(self.X_train, self.y_train)

        if with_calibration:
            print('#' * 50)
            print('BEFORE CALIBRATION')
            print('#' * 50)
        
        # evaluation
        self._model_evaluation()

        if with_calibration:
            self._calibration()

            print('#'*50)
            print('AFTER MODEL CALIBRATION')
            print('#'*50)

            self._model_evaluation()

    def _split_feature_label(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split a DataFrame into features and label.

        This method separates the target label column, defined in `self.param.label`, from the rest of the dataset.

        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame containing both features and the target label column.

        Returns
        -------
        feature : pandas.DataFrame
            DataFrame containing all columns except the label column.
        label : pandas.Series
            Series containing the target label column.
        """
        # extract label
        label = df['LABEL_' + self.param.label]
        # drop label
        feature = df.drop(columns=['LABEL_' + self.param.label])
        return feature, label

    def _load_hyperparameters(self) -> dict:
        """
        Load the best XGBoost hyperparameters that is expected to be in self._log_dir + /xgb_best_hyperparam.pkl.

        Returns
        -------
        dict
            Dictionary containing the best XGBoost hyperparameters.
        """

        file_path = os.path.join(self._log_dir, "xgb_best_hyperparam.pkl")
        with open(file_path, "rb") as f:
            best_params = pickle.load(f)
        return best_params

    def _save_hyperparameters(self, study: optuna.Study) -> None:
        """
        Save the best XGBoost hyperparameters from an Optuna study.

        Parameters
        ----------
        study : optuna.Study
            The Optuna study object containing the best hyperparameters in its ``study.best_params`` attribute.

        Returns
        -------
        None
        """

        with open(self._log_dir + '/xgb_best_hyperparam.pkl', "wb") as f:
            pickle.dump(study.best_params, f)
        print("Best XGBoost hyperparameters have been saved.")

    def _model_evaluation(self):
        """
        Evaluate the performance of the trained model on training, validation, and test sets.

        Please not that this method does not train the model; it only evaluate the performance of self.model

        Returns
        -------
        None
            The method prints evaluation results for each dataset but does not return values.
        """

        print('#' * 50)
        print("Best model evaluation.")
        print('#' * 50)

        # optimize the threshold for binary classification
        best_threshold = None

        # Get validation probabilities
        val_proba = self.model.predict_proba(self.X_val)[:, 1]

        # Search thresholds; avoid exactly 0 and 1 for numerical stability
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1.0
        for t in thresholds:
            val_pred = (val_proba >= t).astype(int)
            f1 = f1_score(self.y_val, val_pred, average="binary", pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        print('#' * 50)
        print(f"Selected threshold on validation (max F1): {best_threshold:.3f} (F1={best_f1:.5f})")
        print('#' * 50)

        # training
        train_proba = self.model.predict_proba(self.X_train)[:, 1]
        train_pred = (train_proba >= best_threshold).astype(int)
        train_metrics = binary_metrics(predictions=train_pred, 
                                       proba_predictions=self.model.predict_proba(self.X_train)[:, 1],
                                       label=self.y_train,
                                       dataset="training")
        for name, value in train_metrics.items():
            print(f"{name}: {value:.5f}")

        # validation
        val_proba = self.model.predict_proba(self.X_val)[:, 1]
        val_pred = (val_proba >= best_threshold).astype(int)
        validation_metrics = binary_metrics(predictions=val_pred, 
                                            proba_predictions=self.model.predict_proba(self.X_val)[:, 1],
                                            label=self.y_val,
                                            dataset="validation")
        for name, value in validation_metrics.items():
            print(f"{name}: {value:.5f}")

        # test
        test_proba = self.model.predict_proba(self.X_test)[:, 1]
        test_pred = (test_proba >= best_threshold).astype(int)
        test_metrics = binary_metrics(predictions=test_pred, 
                                      proba_predictions=self.model.predict_proba(self.X_test)[:, 1],
                                      label=self.y_test,
                                      dataset="test")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.5f}")

    
    def _calibration(self) -> None:
        """
        Callibrate the XGBoost model using isotonic regression.
        """
        if self.param.label != "wilson_label":
            raise NotImplementedError(
                "Calibration is implemented only for 'wilson_label' (binary classification). "
                "For regression targets, probability calibration is not applicable."
        )

        # Fit isotonic calibrator on validation set (post-hoc; XGBoost is already trained)
        calibrator = CalibratedClassifierCV(FrozenEstimator(self.model), method="isotonic")
        calibrator.fit(self.X_val, self.y_val)

        # Replace the model with calibrated wrapper
        self.model = calibrator


    def _hyperparameter_search(self) -> optuna.study.Study:
        """
        Run hyperparameter optimization for the XGBoost model using Optuna.

        The search uses Optuna's TPE sampler with a fixed random seed for reproducibility, and runs for `self.param.num_trials` iterations with a progress bar.

        Returns
        -------
        optuna.study.Study
            Optuna study object containing the optimization history and best parameters (accessible via ``study.best_params``).
        """

        # use to manage the imbalanced dataset
        num_pos = len(self.y_train[self.y_train > 0])
        num_neg = len(self.y_train[self.y_train == 0])
        scale_pos_weight = num_neg / num_pos

        def objective(trial):
            """
            Objective function for Optuna to optimize the hyperparameters of the XGBoost model.
    
            : param trial: Optuna trial object
            : return: accuracy of the model on the validation setRe
            """

            # hyperparameter searching space
            params = construct_hyperparameter_search(self.param, trial, scale_pos_weight)

            model = XGBClassifier(**params)
            performance_evaluation = roc_auc_score

            # model training and performance evaluation
            model.fit(self.X_train, self.y_train, eval_set=[(self.X_val, self.y_val)],
                      verbose=False)

            y_pred = model.predict_proba(self.X_val)[:, 1]

            score = performance_evaluation(self.y_val, y_pred)
            return score

        # run hyperparameter search using Optuna
        print("We begin to the hyperparameter search using Optuna.")

        # manual TPESampler construction to be able to fix the random seed
        sampler = TPESampler(seed=self.param.random_seed)
        study = optuna.create_study(direction="maximize",sampler=sampler)

        # add a progress bar
        progress_bar = TQDMProgressBar(self.param.num_trials, desc="Hyperparameter search")

        # conduct the hyperparameter search
        study.optimize(objective, n_trials=self.param.num_trials, callbacks=[progress_bar])
        progress_bar.close()

        return study


    def predict_all_splits(self, datasets=None):
        """
        Generate probability predictions for one or more dataset splits.

        If ``datasets`` is ``None``, predictions are generated for the training,
        validation, and test sets. If ``datasets == 'validation'``, only validation
        predictions are returned.

        Parameters
        ----------
        datasets : str or None, optional
            If ``None``, return predictions for train/validation/test.
            If ``'validation'``, return predictions for the validation set only.

        Returns
        -------
        tuple of np.ndarray or np.ndarray
            - If ``datasets is None``: a tuple ``(train_preds, val_preds, test_preds)``.
            - If ``datasets == 'validation'``: a single array of validation predictions.
        """
        if datasets is None:
                return self.model.predict_proba(self.X_train), self.model.predict_proba(self.X_val), self.model.predict_proba(self.X_test)

        elif datasets == 'validation':
            return self.model.predict_proba(self.X_val)