import pandas as pd
import argparse
from our_xgboost import OurXGBoost
from our_clinicalModernBERT import OurClinicalModernBERT
import os
from constants import LATEFUSION_MULTIMODAL_DIR
from utils import *
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
import optuna
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
from scipy.special import softmax



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
              - ``"all_data"`` : dict[str, pd.DataFrame]
                    Cleaned split DataFrames (e.g., ``{"train": df, "validation": df, "test": df}``)
                    that contain numeric/categorical features only (text removed).
              - ``"historical_dataset"`` : DatasetDict-like
                    Original Hugging Face dataset with schema information used to
                    detect text columns (accessed via ``.features``).
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

        # BERT model use to make prediction based on text feature
        self._bert_model = OurClinicalModernBERT(param=param, dataset=dataset['text_data'], logdir=self._log_dir)

        # XGBoost model used to make prediction based on numeric/categorical feature
        self._our_xgboost = OurXGBoost(param=param, dataset=dataset['categorical_data'], logdir=self._log_dir)

        # weights used in the late fusion strategy
        self.weight = 0.5 # equal weight to both models; can be tuned


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

        # LateFusion model evaluation
        self.evaluation()

    
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
            self._our_xgboost.load_and_evaluate()
        else:
            print('#'*70)
            print('We start the hyperparameter search for the XGBoost model.')
            print('#'*70)
            self._our_xgboost.hyperparameter_search_and_evaluation()
    
    def _fusion_strategy(self, bert_preds, xgb_preds, weight=None):
        """
        Late fusion strategy to combine predictions from BERT and XGBoost models.

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


    def evaluation(self) -> None:
        print('#'*70)
        print('LateFusion model evaluation.')
        print('#'*70)

        # models predictions
        xgb_train_preds, xgb_val_preds, xgb_test_preds = self._our_xgboost.predict_all_splits()

        bert_train_logits, bert_val_logits, bert_test_logits = self._bert_model.predict_all_splits()
        bert_train_preds = softmax(bert_train_logits.predictions, axis=-1)
        bert_val_preds = softmax(bert_val_logits.predictions, axis=-1)
        bert_test_preds = softmax(bert_test_logits.predictions, axis=-1)

        # combined predictions
        combined_train_preds = self._fusion_strategy(bert_train_preds, xgb_train_preds)
        combined_val_preds = self._fusion_strategy(bert_val_preds, xgb_val_preds)
        combined_test_preds = self._fusion_strategy(bert_test_preds, xgb_test_preds)

        # training
        train_metrics = binary_metrics(predictions=combined_train_preds, label=self._our_xgboost.y_train, dataset="training") if self.param.label == 'wilson_label' else regression_metrics(predictions=combined_train_preds, label=self._our_xgboost.y_train, dataset="training") 
        for name, value in train_metrics.items():
            print(f"{name}: {value:.4f}")

        # validation
        validation_metrics = binary_metrics(predictions=combined_val_preds, label=self._our_xgboost.y_val, dataset="validation") if self.param.label == 'wilson_label' else regression_metrics(predictions=combined_val_preds, label=self._our_xgboost.y_val, dataset="validation") 
        for name, value in validation_metrics.items():
            print(f"{name}: {value:.4f}")

        # test
        test_metrics = binary_metrics(predictions=combined_test_preds, label=self._our_xgboost.y_test, dataset="test") if self.param.label == 'wilson_label' else regression_metrics(predictions=combined_test_preds, label=self._our_xgboost.y_test, dataset="test") 
        for name, value in test_metrics.items():
            print(f"{name}: {value:.4f}")

    
    def _fine_tune_weight_param(self):
        """
        Used to fine-tuned the weight to in the fusion strategy
        """

        # models predictions        
        xgb_val_preds = self._our_xgboost.predict_all_splits(datasets='validation')
        bert_logit_pred = self._bert_model.predict_all_splits(dataset='validation')
        bert_val_pred = softmax(bert_logit_pred.predictions, axis=-1)


        def objective(trial):
            """
            Objectve function for Optuna to optimize the self.weight param

            :param trial: Optuna trial object
            :return score we aim to optimize
            """
            # first we define the hyperparameter searching space
            w = trial.suggest_float('weight', 0.0, 1.0)

            if self.param.label == 'wilson_label':
                performance_evaluation = f1_score

            # define the metric according to the task
            if self.param.label in ['dosing_error_rate', 'sum_dosing_errors']:
                performance_evaluation = mean_absolute_error
                
            
            y_pred = self._fusion_strategy_proba(bert_val_pred, xgb_val_preds, weight=w)
            score = performance_evaluation(self._our_xgboost.y_val, y_pred)
            return score
        
        print("We begin to fine-tune the weight using Optuna.")

        # manual TPESampler construction to be able to fix the random seed
        sampler = TPESampler(seed=self.param.random_seed) 
        study = optuna.create_study(direction="maximize" if self.param.label == 'wilson_label' else "minimize", sampler=sampler)

        # add a progress bar
        progress_bar = TQDMProgressBar(self.param.num_trials, desc="Hyperparameter search")

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
        Prepare the ClinicalModernBERT model for text embedding.

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
            print('# The load and evaluate a fine-tuned ClinicalModernBERT model.')
            print('#'*70)
            self._bert_model.load_model()
            # self._load_and_evaluate_bert_model()

    
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

