import pandas as pd
import argparse
import torch
from utils import *
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from DosingErrorDataset import DosingErrorDataset
import os
from constants import BERT_DIR
from transformers import DataCollatorWithPadding
from transformers import (EarlyStoppingCallback, TrainingArguments)
from CustomTrainer import CustomTrainer
from typing import Tuple
import transformers


class OurClinicalModernBERT:
    def __init__(self, param: argparse.Namespace, dataset: dict[str, pd.DataFrame], logdir=None):

        """
        Initialize a ClinicalModernBERT model configured for our settings.

        This constructor:
          - Stores the experiment configuration (`param`).
          - Creates the logging/output directory under `BERT_DIR/<param.label>`.
          - Selects the compute device and dtype (bf16 on supported CUDA GPUs, else fp32).
          - Loads the pretrained ModernBERT model and its tokenizer from the Hugging Face Hub
            (`'Simonlee711/Clinical_ModernBERT'`).
          - Builds a padding data collator for dynamic, batch-wise padding.
          - Computes per-device batch sizes.
          - Converts the provided pandas DataFrames into datasets ready for BERT-style training.

          
        Parameters
        ----------
        param : argparse.Namespace
            General configuration object (see main.py)
        dataset : dict[str, pandas.DataFrame]
            A dict containing keys {'train, 'validation', test'} and where the values are the associated datasets.
        """

        # general config
        self.param = param

        # model name
        self.nlp_model_name = 'Simonlee711/Clinical_ModernBERT'

        # log dir 
        if logdir is None:
            self._log_dir = os.path.join(BERT_DIR, self.param.label)
            os.makedirs(self._log_dir, exist_ok=True)
        else:
            self._log_dir = logdir

        # defining device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # check the type we could work with
        self._dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and "cuda" in self.device.lower() and torch.cuda.is_bf16_supported())
            else torch.float32
        )

        # compute batch size per device
        self.train_batch_per_device, self.eval_batch_per_device = compute_batch_size(param=self.param)

        # load Clinical_ModernBERT model and associated tokenizer
        self._nlp_model, self._tokenizer = self._load_model_and_tokenizer(self.param)

        # load the datacollator
        self._data_collator = DataCollatorWithPadding(tokenizer=self._tokenizer)

        # datasets
        self.train_dataset = self.prepare_dataset_for_bert(dataset['train'])
        self.validation_dataset = self.prepare_dataset_for_bert(dataset['validation'])
        self.test_dataset = self.prepare_dataset_for_bert(dataset['test'])

    def train_and_evaluate(self) -> None:
        """"
        Train the model, save weights, and evaluate model.

        Builds a `Trainer` via `_trainer_initialization()`, runs `trainer.train()`,
        saves the in-memory model to `training_args.output_dir', then calls `_evaluation(trainer)`.
        """

        trainer, training_args = self._trainer_initialization()

        # model training
        trainer.train()

        # model saving
        trainer.save_model(training_args.output_dir)

        # model evaluation
        self._evaluation(trainer=trainer)

    def load_and_evaluate(self) -> None:
        """
        Load the fine-tuned model from `self._log_dir` and run evaluation.

        Requirements
        ------------
        `self._log_dir` must contain a valid Transformers save
        """

        # loading fine-tuned model
        self._nlp_model = AutoModelForSequenceClassification.from_pretrained(self._log_dir).to(self.device)

        # construct model to easily evaluate the model
        trainer, _ = self._trainer_initialization()

        # model evaluation
        self._evaluation(trainer=trainer)
    
    
    def predict_all_splits(self, dataset=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate prediction scores for one or more dataset splits.

        If ``dataset`` is ``None``, predictions are returned for the train,
        validation, and test sets. If ``dataset == 'validation'``, only
        validation predictions are returned.

        Parameters
        ----------
        dataset : str or None, optional
            If ``None``, return predictions for train/validation/test.
            If ``'validation'``, return predictions for the validation set.

        Returns
        -------
        tuple of np.ndarray or np.ndarray
            - If ``dataset is None``: ``(train_preds, val_preds, test_preds)``.
            - If ``dataset == 'validation'``: validation predictions.
        """

        # construct trainer to easily evaluate the model
        trainer, _ = self._trainer_initialization()

        if dataset is None:
            train_prediction = trainer.predict(self.train_dataset)
            val_prediction = trainer.predict(self.validation_dataset)
            test_prediction = trainer.predict(self.test_dataset)

            return train_prediction, val_prediction, test_prediction

        elif dataset == 'validation':
            val_prediction = trainer.predict(self.validation_dataset)
            return val_prediction
    

    def load_model(self) -> None:
        """
        Load the fine-tuned model from `self._log_dir`.

        Requirements
        ------------
        `self._log_dir` must contain a valid Transformers save
        """
        # loading fine-tuned model
        self._nlp_model = AutoModelForSequenceClassification.from_pretrained(self._log_dir, torch_dtype=self._dtype).to(self.device)

    def _trainer_initialization(self) -> Tuple[CustomTrainer, TrainingArguments]:
        """
        Build and return a configured Hugging Face `Trainer` and its `TrainingArguments`.

        Behavior
        --------
        - Selects task-specific metric & comparator from `self.param.label`
        - Adds `EarlyStoppingCallback(patience=self.param.early_stopping_patience)`.
        - Evaluates and saves every `self.param.eval_and_save_steps`, keeps 1 checkpoint,
            and reloads the best model at the end.
        - Uses bf16 when supported, gradient accumulation/checkpointing, and a padding collator.

        Returns
        -------
        trainer : transformers.Trainer
            Trainer bound to `self._nlp_model`, datasets, collator, metrics, and callbacks.
        training_args : transformers.TrainingArguments
            Arguments controlling training, logging, evaluation, and checkpointing.
        """

        # early stopping callback
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=self.param.early_stopping_patience)

        # specific training arguments associated to the task
        if self.param.label in ['dosing_error_rate', 'sum_dosing_errors']:
            metric_for_best_model = "eval_MAE"  # "mse"
            greater_is_better = False
            compute_metrics = regression_metrics_hf
        elif self.param.label == 'wilson_label':
            metric_for_best_model = "F1 Score"
            greater_is_better = True
            compute_metrics = binary_metrics_hf
        else:
            raise ValueError(
                f"Label {self.param.label} is not supported for the BERT model training. Only wilson_label, dosing_error_rate and sum_dosing_errors are supported.")

        # training arguments
        training_args = TrainingArguments(
            output_dir=self._log_dir,
            num_train_epochs=self.param.num_epoch,
            # batch sizes
            per_device_train_batch_size=self.train_batch_per_device,
            per_device_eval_batch_size=self.eval_batch_per_device,
            gradient_accumulation_steps=self.param.gradient_accumulation_step,
            # Other arguments
            learning_rate=self.param.learning_rate,
            weight_decay=self.param.weight_decay,
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=self.param.eval_and_save_steps,
            save_steps=self.param.eval_and_save_steps,
            logging_dir=self._log_dir + '/logs',
            logging_strategy="steps",
            logging_steps=self.param.eval_and_save_steps,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            report_to="tensorboard",
            seed=self.param.random_seed,
            bf16=True if self._dtype == torch.bfloat16 else False,
            max_grad_norm=1.0,
            dataloader_num_workers=4,
            gradient_checkpointing=True
        )

        # trainer
        trainer = CustomTrainer(
            model=self._nlp_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.validation_dataset,
            processing_class=self._tokenizer,
            data_collator=self._data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
            config=self.param,
        )

        return trainer, training_args

    def _evaluation(self, trainer: CustomTrainer) -> None:
        """
        Evaluate the model on the validation set and the held-out test set, printing metrics.


        Parameters
        ----------
        trainer : CustomTrainer
            A configured Hugging Face Trainer with:
            - `eval_dataset` set (validation split),
            - `compute_metrics` defined (to produce metrics),
        """

        # evaluation set
        print("\n Evaluating the best model on the validation set...")
        eval_results = trainer.evaluate()

        print("\n--- Validation Set Results ---")
        for key, value in eval_results.items():
            print(f"{key.replace('eval_', '')}: {value:.4f}")

        # test set
        print("\n Evaluating the final model on the HELD-OUT TEST set...")
        test_results = trainer.predict(self.test_dataset)

        print("\n --- Final Test Set Results ---")
        for key, value in test_results.metrics.items():
            print(f"{key.replace('test_', '')}: {value:.4f}")
        print("---------------------------------")

    def prepare_dataset_for_bert(self, df: pd.DataFrame) -> DosingErrorDataset:
        """
        Build a tokenized dataset for ModernBERT/Trainer from a pandas DataFrame.

        Steps
        -----
        1. Create a unified text field `global_text_feature` per row using
        `create_one_global_text_feature(row, self.param)`.
        2. Infer the task from `self.param.label`:
        - Regression: {'dosing_error_rate', 'sum_dosing_errors'}
        - Classification: 'wilson_label'
        3. Instantiate and return a `DosingErrorDataset` with tokenization and truncation.

        Parameters
        ----------
        df : pandas.DataFrame
            Input table containing all text feature and the target column called `self.param.label`.

        Returns
        -------
        DosingErrorDataset
            A dataset yielding tokenized inputs (and labels) compatible with
            Hugging Face `Trainer`.
        """

        # add a global text feature to the dataframe
        df['global_text_feature'] = df.apply(lambda row: create_one_global_text_feature(row, self.param), axis=1)

        # adapt the task to the label
        if self.param.label in ['dosing_error_rate', 'sum_dosing_errors']:
            task = 'regression'
            labels = pd.to_numeric(df[self.param.label], errors='coerce').fillna(0.0).astype('float32').tolist()
            # label_dtype = torch.bfloat16 if (self._dtype == torch.bfloat16) else torch.float32
            label_dtype = torch.float32
        elif self.param.label == 'wilson_label':
            task = 'classification'
            labels = pd.to_numeric(df[self.param.label], errors='raise').astype('int64').tolist()
            label_dtype = torch.long
        else:
            raise ValueError(
                f"Label {self.param.label} is not supported for the BERT model training. Only wilson_label, dosing_error_rate and sum_dosing_errors are supported.")

        # create the dataset
        dataset = DosingErrorDataset(
            texts=df['global_text_feature'].tolist(),
            labels=labels,
            tokenizer=self._tokenizer,
            max_length=self.param.max_length,
            task=task,
            label_dtype=label_dtype,
        )
        return dataset

    def _load_model_and_tokenizer(self, param: argparse.Namespace) -> Tuple[
        transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
        """
        Load the pretrained ModernBERT sequence-classification model and its tokenizer.


        Parameters
        ----------
        param : argparse.Namespace
            Experiment configuration; must provide `label` specifying the prediction task.

        Returns
        -------
        model : transformers.PreTrainedModel
            ModernBERT sequence-classification model loaded on `self.device`.
        tokenizer : transformers.PreTrainedTokenizerBase
            Tokenizer associated with `self.nlp_model_name`.
        """
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.nlp_model_name)

        # load config
        config = AutoConfig.from_pretrained(self.nlp_model_name)
        if hasattr(config, "reference_compile"):
            config.reference_compile = False

        # required small changes if regression task
        if param.label in ['dosing_error_rate', 'sum_dosing_errors']:
            config.num_labels = 1
            config.problem_type = "regression"

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(self.nlp_model_name, config=config).to(self.device)

        return model, tokenizer

