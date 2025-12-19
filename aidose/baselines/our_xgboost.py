from xgboost import XGBClassifier, XGBRegressor
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
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

# to suppress annoying optuna default logs
optuna.logging.set_verbosity(optuna.logging.WARNING)


class OurXGBoost:

    def __init__(self, param: argparse.Namespace, dataset: dict[str, pd.DataFrame], logdir=None):
        """
        Initialize an XGBoost model specifically adapted to our dataset.

        This constructor:
          - Save the training, validation and test sets.
          - Creates the logging directory for outputs.
          - Initializes the appropriate XGBoost model (classifier or regressor)
            depending on the specified prediction task.

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

        # initialization of the XGBoost model according to the task
        if self.param.label == 'wilson_label':
            self.model = XGBClassifier()
        elif self.param.label in ['sum_dosing_errors', 'dosing_error_rate']:
            self.model = XGBRegressor()
        else:
            raise NotImplementedError('XGBoost model is only able to handle wilson_label, sum_dosing_errors and dosing_error_rate labels')

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
        sample_alpha = best_param.pop("sample_alpha", None)  # pop to avoid a warning
        self.model = XGBClassifier(**best_param) if self.param.label == 'wilson_label' else XGBRegressor(**best_param)

        sample_weight = None if self.param.label == "wilson_label" else np.where(self.y_train > 0, sample_alpha, 1.0)
        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weight)

        # evaluation
        self._model_evaluation()

        # hyperparameter saving
        self._save_hyperparameters(study=study)

    def load_and_evaluate(self) -> None:
        """
        Load the best hyperparameters, retrain the model, and evaluate it.

        Returns
        -------
        None
        """
        # loading model
        best_params = self._load_hyperparameters()

        # model initialization and training
        sample_alpha = best_params.pop("sample_alpha", None)  # pop to avoid warning
        self.model = XGBClassifier(**best_params) if self.param.label == 'wilson_label' else XGBRegressor(**best_params)

        sample_weight = None if self.param.label == "wilson_label" else np.where(self.y_train > 0, sample_alpha, 1.0)

        self.model.fit(self.X_train, self.y_train, sample_weight=sample_weight)

        # evaluation
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
        label = df[self.param.label]
        # drop label
        feature = df.drop(columns=[self.param.label])
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

        # training
        train_metrics = binary_metrics(predictions=self.model.predict(self.X_train), label=self.y_train,
                                       dataset="training") if self.param.label == 'wilson_label' else regression_metrics(
            predictions=self.model.predict(self.X_train), label=self.y_train, dataset="training")
        for name, value in train_metrics.items():
            print(f"{name}: {value:.4f}")

        # validation
        validation_metrics = binary_metrics(predictions=self.model.predict(self.X_val), label=self.y_val,
                                            dataset="validation") if self.param.label == 'wilson_label' else regression_metrics(
            predictions=self.model.predict(self.X_val), label=self.y_val, dataset="validation")
        for name, value in validation_metrics.items():
            print(f"{name}: {value:.4f}")

        # test
        test_metrics = binary_metrics(predictions=self.model.predict(self.X_test), label=self.y_test,
                                      dataset="test") if self.param.label == 'wilson_label' else regression_metrics(
            predictions=self.model.predict(self.X_test), label=self.y_test, dataset="test")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")

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

            # hyperparameter searching space that is independent of the task
            params = construct_hyperparameter_search(self.param, trial, scale_pos_weight)

            # specific to the binary task
            if self.param.label == 'wilson_label':
                model = XGBClassifier(**params)
                performance_evaluation = roc_auc_score  # aux_precision
                sample_weight = None

            # specific to the regression task
            if self.param.label in ['dosing_error_rate', 'sum_dosing_errors']:
                # sample_weight is used to weight the loss function -> is used to manage class imbalance
                # sample_alpha = trial.suggest_float("sample_alpha", 5.0, 100.0, log=True) # F1 score label_frequencies 22.68 with mae on positive label
                # sample_alpha = trial.suggest_float("sample_alpha", 5.0, 200.0, log=True) # best for label_sum test F1 25.95 with mae
                sample_alpha = trial.suggest_float("sample_alpha", 5.0, 200.0, log=True)

                model = XGBRegressor(**params)
                sample_weight = np.where(self.y_train > 0, sample_alpha, 1.0)
                performance_evaluation = mean_absolute_error

            # model training and performance evaluation
            model.fit(self.X_train, self.y_train, sample_weight=sample_weight, eval_set=[(self.X_val, self.y_val)],
                      verbose=False)

            # average precision score requires proba
            if self.param.label == 'wilson_label':
                y_pred = model.predict_proba(self.X_val)[:, 1]
            else:
                y_pred = model.predict(self.X_val)

            score = performance_evaluation(self.y_val, y_pred)
            return score

        # run hyperparameter search using Optuna
        print("We begin to the hyperparameter search using Optuna.")

        # manual TPESampler construction to be able to fix the random seed
        sampler = TPESampler(seed=self.param.random_seed)
        study = optuna.create_study(direction="maximize" if self.param.label == 'wilson_label' else "minimize",
                                    sampler=sampler)

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