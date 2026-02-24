import pandas as pd
import argparse
from our_xgboost import OurXGBoost
from our_clinicalModernBERT import OurClinicalModernBERT
import os
from constants import LATEFUSION_MULTIMODAL_DIR
from utils import *
from optuna.samplers import TPESampler
import optuna
from sklearn.metrics import roc_auc_score, f1_score
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression



class LateFusionModel:
    def __init__(self, param: argparse.Namespace, dataset: dict[str, dict[str, pd.DataFrame]]):
        """"
        Implementing a Multimodal model that uses a late fusion strategy. 
        
        Specifically, it combines the predictions from a fine-tuned ClinicalModernBERT model trained on textual feature and a XGBoost model trained on categorical features.

        Parameters
        ----------
        param : argparse.Namespace
            Experiment configuration, see main.py

        dataset : dict[str, dict[str, pd.DataFrame]]
            A composite dictionary with the following expected keys:
              - ``"categorical_data"`` : DatasetDict-like
                    Dataset prepared for XGBoost, containing numeric/categorical features and the target label.
              - ``"text_data"`` : DatasetDict-like
                    Tokenized/text-prepared dataset used by the BERT module
        """

        # general config
        self.param = param

        # logdir 
        self._log_dir = os.path.join(LATEFUSION_MULTIMODAL_DIR, self.param.label) 
        os.makedirs(self._log_dir, exist_ok=True)

        # save datasets
        self._text_data = dataset['text_data']
        self._categorical_data = dataset['categorical_data']

        # save test nctid
        self._test_nctid_text = dataset['text_data']['test_nctid']["METADATA_nctId"]
        self._test_nctid_categorical = dataset['categorical_data']['test_nctid']["METADATA_nctId"]

        # BERT model use to make prediction based on text feature
        self._bert_model = OurClinicalModernBERT(param=param, dataset=dataset['text_data'], logdir=self._log_dir)

        # XGBoost model used to make prediction based on numeric/categorical feature
        self._our_xgboost = OurXGBoost(param=param, dataset=dataset['categorical_data'], logdir=self._log_dir)

        # weights used in the late fusion strategy
        self.weight = 0.5 # equal weight to both models; will be tuned

        # calibrator model
        self._calibrator = LogisticRegression(
            solver="lbfgs",
            penalty="l2",
            C=1.0,
            max_iter=2000,
            class_weight=None,
        )


    def train_and_evaluate(self) -> None:
        """
        End-to-end training and evaluation pipeline for the multimodal model.

        This method orchestrates the full workflow:
        1. Prepare the ClinicalModernBERT model: either fine-tune it 
             (if ``param.multimodal_train_bert_model`` is True) or load  an existing checkpoint.
        2. Prepare the XGBoost model: either conduct hyperparameter search and training
             (if ``param.load_xgboost_model`` is False) or load an existing trained model.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        # prepare the XGBoost model
        self._prepare_xgb_model()
        
        # prepare the BERT model
        self._prepare_bert_model()

        # fine-tune the weight used in the fusion strategy
        study = self._fine_tune_weight_param()
        self.weight = study.best_params['weight']
        print('The optimal weight found is ', self.weight)

        print('#'* 50)
        print('BEFORE CALIBRATION')
        print('#'* 50)
        # LateFusion model evaluation before calibration
        self.evaluation(title="before_calibration")

        self._calibrate()

        print('#'* 50)
        print('AFTER CALIBRATION')
        print('#'* 50)
        # LateFusion model evaluation after calibration
        self.evaluation(calibrate=True, title="after_calibration")
    
    def _calibrate(self) -> None:
        """
        Calibrate the FINAL fused probabilities using Platt scaling (logistic calibration).

        This fits a 1D logistic regression mapping from the fused score to calibrated probability:
            p_cal = sigmoid(a * score + b)

        We fit on the validation set only, then store the calibrator for later use.
        """

        # construct proba predictions on validation set
        xgb_val_proba = self._our_xgboost.predict_all_splits(datasets="validation")  # expected shape (N, 2)

        # BERT validation logits -> probs
        bert_val_logits = self._bert_model.predict_all_splits(dataset="validation")
        bert_val_proba = softmax(bert_val_logits.predictions, axis=-1)  # shape (N, 2)

        # Fused probabilities
        fused_val_proba = self._fusion_strategy_proba(bert_val_proba, xgb_val_proba)

        # Positive-class probability
        p_val = np.asarray(fused_val_proba)[:, 1].astype(float)

        # Labels
        y_val = np.asarray(self._our_xgboost.y_val).astype(int)

        # for stability reason, we calibrate logit(p) instead of p directly
        X = logit(p_val).reshape(-1, 1)
        
        # fit the calibrator
        self._calibrator.fit(X, y_val)
    
    def _apply_calibration(self, fused_proba_2col: np.ndarray) -> np.ndarray:
        """
        Apply the stored Platt calibrator to fused probabilities.

        Parameters
        ----------
        fused_proba_2col : np.ndarray of shape (N, 2)
            Fused probabilities (class 0, class 1).

        Returns
        -------
        np.ndarray of shape (N, 2)
            Calibrated probabilities (class 0, class 1).
        """

        p = np.asarray(fused_proba_2col)[:, 1].astype(float)
        X = logit(p).reshape(-1, 1)
        p_cal = self._calibrator.predict_proba(X)[:, 1]
        p_cal = np.clip(p_cal, 1e-12, 1 - 1e-12)

        out = np.zeros_like(fused_proba_2col, dtype=float)
        out[:, 1] = p_cal
        out[:, 0] = 1.0 - p_cal
        return out
        
    
    def _prepare_xgb_model(self) -> OurXGBoost:
        """
        Prepare the XGBoost model.
        
        This method loads the hyperparameter configuration or conduct a hyperparameter search if requested by the self.param configuration.

        Parameters
        ----------
        None

        Returns
        -------
        OurXGBoost
            A trained OurXGBoost model ready for evaluation.
        """
        if self.param.load_xgboost_model:
            # load a previously trained XGBoost model
            print('#'*70)
            print('Performance of the loaded XGBoost model.')
            print('#'*70)
            self._our_xgboost.load_and_evaluate(with_calibration=False)
        else:
            raise NotImplementedError("Hyperparameter search for XGBoost inside LateFusionModel is not implemented.")
    
    def _fusion_strategy(self, bert_preds, xgb_preds, weight=None):
        """
        Late fusion strategy to combine predictions from BERT and XGBoost models.

        This method combines the predictions from the ClinicalModernBERT and XGBoost models using a weighted average approach.
        The weight is defined according to the self.weights attribute.

        Parameters
        ----------
        bert_preds : array-like
            Predictions from the ClinicalModernBERT model.
        xgb_preds : array-like
            Predictions from the XGBoost model.

        Returns
        -------
        combined_preds : array-like
            The combined predictions after applying the late fusion strategy.
        """
        w = self.weight if weight is None else weight 
        combined_preds = (w * xgb_preds) + ((1 - w)*bert_preds)
        return np.argmax(combined_preds, axis=-1)
    
    def _fusion_strategy_proba(self, bert_preds, xgb_preds, weight=None):
        """
        Late fusion strategy to combine predictions from BERT and XGBoost models, but return probabilities instead of class

        This method combines the predictions from the ClinicalModernBERT and XGBoost models using a weighted average approach.
        The weight is defined accordint to the self.weights attribute.

        Parameters
        ----------
        bert_preds : array-like
            Predictions from the ClinicalModernBERT model.
        xgb_preds : array-like
            Predictions from the XGBoost model.

        Returns
        -------
        combined_preds : array-like
            The combined predictions after applying the late fusion strategy.
        """
        w = self.weight if weight is None else weight 
        combined_preds = (w * xgb_preds) + ((1 - w)*bert_preds)
        return combined_preds


    def evaluation(self, calibrate=False, title=None) -> None:
        print('#'*70)
        print('LateFusion model evaluation.')
        print('#'*70)

        # models predictions
        xgb_train_preds, xgb_val_preds, xgb_test_preds = self._our_xgboost.predict_all_splits()

        bert_train_logits, bert_val_logits, bert_test_logits = self._bert_model.predict_all_splits()
        bert_train_preds = softmax(bert_train_logits.predictions, axis=-1) if self.param.label == 'wilson_label' else bert_train_logits.predictions[:, -1]
        bert_val_preds = softmax(bert_val_logits.predictions, axis=-1) if self.param.label == 'wilson_label' else bert_val_logits.predictions[:, -1]
        bert_test_preds = softmax(bert_test_logits.predictions, axis=-1) if self.param.label == 'wilson_label' else bert_test_logits.predictions[:, -1]

        # weighted combined predictions
        combined_train_proba_preds = self._fusion_strategy_proba(bert_train_preds, xgb_train_preds)
        combined_val_proba_preds = self._fusion_strategy_proba(bert_val_preds, xgb_val_preds)
        combined_test_proba_preds = self._fusion_strategy_proba(bert_test_preds, xgb_test_preds)

        # calibrate probabilities if requested
        if calibrate:
            combined_train_proba_preds = self._apply_calibration(combined_train_proba_preds)
            combined_val_proba_preds   = self._apply_calibration(combined_val_proba_preds)
            combined_test_proba_preds  = self._apply_calibration(combined_test_proba_preds)

        # opztimize threshold for binary classification
        best_threshold = None
        best_f1 = -1.0

        y_val = np.asarray(self._our_xgboost.y_val).astype(int)
        val_pos_proba = np.asarray(combined_val_proba_preds)[:, 1]

        thresholds = np.linspace(0.01, 0.99, 99)
        for t in thresholds:
            val_pred = (val_pos_proba >= t).astype(int)
            f1 = f1_score(y_val, val_pred, average="binary", pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        print('#'*70)
        print(f"Selected threshold on validation (max F1): {best_threshold:.3f} (F1={best_f1:.5f})")
        print('#'*70)

        # Apply the chosen threshold to all splits
        train_pos_proba = np.asarray(combined_train_proba_preds)[:, 1]
        test_pos_proba  = np.asarray(combined_test_proba_preds)[:, 1]

        combined_train_preds = (train_pos_proba >= best_threshold).astype(int)
        combined_val_preds   = (val_pos_proba   >= best_threshold).astype(int)
        combined_test_preds  = (test_pos_proba  >= best_threshold).astype(int)


        # training
        train_metrics = binary_metrics(predictions=combined_train_preds, proba_predictions=combined_train_proba_preds[:, 1], label=self._our_xgboost.y_train, dataset="training") if self.param.label == 'wilson_label' else regression_metrics(predictions=combined_train_proba_preds, label=self._our_xgboost.y_train, dataset="training") 
        for name, value in train_metrics.items():
            print(f"{name}: {value:.5f}")

        # validation
        validation_metrics = binary_metrics(predictions=combined_val_preds, proba_predictions=combined_val_proba_preds[:, 1], label=self._our_xgboost.y_val, dataset="validation") if self.param.label == 'wilson_label' else regression_metrics(predictions=combined_val_proba_preds, label=self._our_xgboost.y_val, dataset="validation") 
        for name, value in validation_metrics.items():
            print(f"{name}: {value:.5f}")

        # test
        test_metrics = binary_metrics(predictions=combined_test_preds, proba_predictions=combined_test_proba_preds[:, 1], label=self._our_xgboost.y_test, dataset="test") if self.param.label == 'wilson_label' else regression_metrics(predictions=combined_test_proba_preds , label=self._our_xgboost.y_test, dataset="test") 
        for name, value in test_metrics.items():
            print(f"{name}: {value:.5f}")
        
        # writing test predictions to disk for later analysis
        test_output_df = pd.DataFrame({
            "nctid": self._test_nctid_text,
            "true_label": self._our_xgboost.y_test,
            "xgb_proba": xgb_test_preds[:, 1],
            "bert_proba": bert_test_preds[:, 1],
            "combined_proba": combined_test_proba_preds[:, 1],
            "combined_pred": combined_test_preds        })
        
        if title is not None:
            test_output_df.to_csv(os.path.join(self._log_dir, f"test_predictions_{title}.csv"), index=False)   
        else:
            test_output_df.to_csv(os.path.join(self._log_dir, f"test_predictions.csv"), index=False)

    
    def _fine_tune_weight_param(self):
        """
        Used to fine-tuned the weight to in the fusion strategy
        """

        # models predictions        
        xgb_val_preds = self._our_xgboost.predict_all_splits(datasets='validation')
        bert_logit_pred = self._bert_model.predict_all_splits(dataset='validation')
        bert_val_pred = softmax(bert_logit_pred.predictions, axis=-1) if self.param.label == 'wilson_label' else bert_logit_pred.predictions


        def objective(trial):
            """
            Objectve function for Optuna to optimize the self.weight param

            :param trial: Optuna trial object
            :return score we aim to optimize
            """
            # first we define the hyperparameter searching space
            w = trial.suggest_float('weight', 0.0, 1.0)
            performance_evaluation = roc_auc_score          
                
            
            y_pred = self._fusion_strategy_proba(bert_val_pred, xgb_val_preds, weight=w)
            score = performance_evaluation(self._our_xgboost.y_val.to_numpy(), y_pred[:, -1])
            return score
        
        print("We begin to fine-tune the weight using Optuna.")

        # manual TPESampler construction to be able to fix the random seed
        sampler = TPESampler(seed=self.param.random_seed) 
        study = optuna.create_study(direction="maximize", sampler=sampler)

        # add a progress bar
        progress_bar = TQDMProgressBar(self.param.late_fusion_num_trials, desc="Hyperparameter search")

        # conduct the hyperparameter search
        study.optimize(objective, n_trials=self.param.late_fusion_num_trials, callbacks=[progress_bar])
        progress_bar.close()

        return study

    
    def _load_and_evaluate_bert_model(self) -> None:
        """
        Load a previously fine-tuned ClinicalModernBERT model and evaluate its performance.

        This method assumes that the ClinicalModernBERT model has already been fine-tuned and saved to disk inside ``self._log_dir``. 
        It reloads the model and calls its evaluation routine.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self._bert_model.load_and_evaluate()


    def _prepare_bert_model(self) -> None:
        """
        Prepare the ClinicalModernBERT model.

        Depending on the configuration, this method either:
            - Fine-tunes the ClinicalModernBERT model on the training data 
                (if ``param.multimodal_train_bert_model`` is True).
            - Loads a previously fine-tuned model from disk (otherwise).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # fine-tune or load a model
        if self.param.multimodal_train_bert_model:
            print('#'*70)
            print('# We start to fine-tune the ClinicalModernBERT model.')
            print('#'*70)
            self._bert_model.train_and_evaluate()
        else:
            print('#'*70)
            print('# We load a fine-tuned ClinicalModernBERT model.')
            print('#'*70)
            self._bert_model.load_model()

    
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
        feature =  df.drop(columns=[self.param.label])
        return feature, label


    def _check_param(self) -> None:
        """
        Validate that required dataset and model files exist given the current parameters.

        This method ensures consistency between runtime parameters and available resources:
            - If ``param.embed_text_feature`` is False, it checks that a precomputed dataset with text embeddings already exists in ``self._log_dir``.
            - If ``param.multimodal_train_bert_model`` is False, it checks that the necessary files for loading a pretrained ClinicalModernBERT model are present in ``self._log_dir``.

        Parameters
        ----------
        None

        Raises
        ------
        FileNotFoundError
            If any of the required files are missing, depending on the parameter settings.

        """
        
        if not self.param.multimodal_train_bert_model:
            # list of files we are expected to found
            required_model_files = ["config.json", "model.safetensors",  "tokenizer.json"]
            if any(not os.path.exists(os.path.join(self._log_dir, file)) for file in required_model_files):
                raise FileNotFoundError(f"If --multimodal_train_bert_model is set to False, the files required to load a ClinicalModernBERT are expected to be found in {self._log_dir}")

