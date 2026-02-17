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
import torch.nn.functional as F


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

        # for evaluation and thereshold optimization
        self.validation_labels = dataset["validation"][f"LABEL_{self.param.label}"].astype(int).to_numpy()
        self.test_labels = dataset["test"][f"LABEL_{self.param.label}"].astype(int).to_numpy()

        # for calibration
        self._platt_a = 1.0
        self._platt_b = 0.0

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

    def load_and_evaluate(self, with_calibration=True) -> None:
        """
        Load the fine-tuned model from `self._log_dir` and run evaluation.

        Parameters
        ----------
        with_calibration : bool, optional
            Whether to apply calibration after training (default is True).

        Requirements
        ------------
        `self._log_dir` must contain a valid Transformers save
        """

        # loading fine-tuned model
        self._nlp_model = AutoModelForSequenceClassification.from_pretrained(self._log_dir).to(self.device)

        # construct model to easily evaluate the model
        trainer, _ = self._trainer_initialization()

        if with_calibration:
            print('#'*50)
            print('BEFORE CALIBRATION')
            print('#'*50)

        # model evaluation
        self._evaluation(trainer=trainer)

        if with_calibration:
            self._calibration(trainer=trainer)

            print('#'*50)
            print('AFTER MODEL CALIBRATION')
            print('#'*50)

            self._evaluation(trainer=trainer, use_calibration=True)

    def _calibration(self, trainer) -> None:
        """
        Calibrate the binary classifier using Platt scaling on the validation set.

        Platt scaling fits a logistic regression on the model's *logit margin*: 
            p_cal = sigmoid(a * m + b),
        where m is the difference between the positive and negative logits (logit margin).

        Parameters
        ----------
        trainer : CustomTrainer
            A configured Hugging Face Trainer. (We reuse it; do not re-initialize.)
        """
        # Get validation logits
        val_pred = trainer.predict(self.validation_dataset)
        logits_np = val_pred.predictions  # (N, 2) for binary sequence classification

        logits = torch.tensor(logits_np, dtype=torch.float32, device=self.device)  # (N, 2)
        y = torch.tensor(self.validation_labels, dtype=torch.float32, device=self.device)  # (N,)

        # Build a 1D "score" for Platt: logit margin m = z_pos - z_neg
        margin = logits[:, 1] - logits[:, 0]  # (N,)

        # Fit a and b by minimizing NLL (binary cross-entropy with logits)
        #    p = sigmoid(a*m + b)
        a = torch.nn.Parameter(torch.ones((), device=self.device))
        b = torch.nn.Parameter(torch.zeros((), device=self.device))

        optimizer = torch.optim.LBFGS([a, b], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

        def _eval_loss():
            # logits for BCE are (a*m + b)
            calibrated_logits = a * margin + b
            return F.binary_cross_entropy_with_logits(calibrated_logits, y)

        def closure():
            optimizer.zero_grad()
            loss = _eval_loss()
            loss.backward()
            return loss

        optimizer.step(closure)

        a_opt = float(a.detach().cpu().item())
        b_opt = float(b.detach().cpu().item())

        # Store parameters for use during evaluation
        self._platt_a = a_opt
        self._platt_b = b_opt


        print("#" * 50)
        print(f"Platt scaling fitted on validation: a = {self._platt_a:.4f}, b = {self._platt_b:.4f}")
        print("#" * 50)



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
        metric_for_best_model = "ROC-AUC"
        greater_is_better = True
        compute_metrics = binary_metrics_hf

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

    def _evaluation(self, trainer: CustomTrainer, use_calibration: bool = False) -> None:
        """
        Evaluate the model on the validation set and the held-out test set, printing metrics.

        Parameters
        ----------
        trainer : CustomTrainer
            A configured Hugging Face Trainer with:
            - `eval_dataset` set (validation split),
            - `compute_metrics` defined (to produce metrics),
        """

        # prediction on validation set
        evaluation_prediction = trainer.predict(self.validation_dataset)
        val_logits = torch.tensor(evaluation_prediction.predictions, dtype=torch.float32).to(self.device)

        if use_calibration:
            margin = val_logits[:, 1] - val_logits[:, 0]
            val_proba = torch.sigmoid(self._platt_a * margin + self._platt_b).detach().cpu().numpy()
        else:
            val_proba = torch.softmax(val_logits, dim=1).detach().cpu().numpy()[:, 1]
        
        # optimize the threshold for binary classification
        best_threshold = None

        # Search thresholds; avoid exactly 0 and 1 for numerical stability
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = -1.0
        for t in thresholds:
            val_pred = (val_proba >= t).astype(int)
            f1 = f1_score(self.validation_labels, val_pred, average="binary", pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(t)

        print('#' * 50)
        print(f"Selected threshold on validation (max F1): {best_threshold:.3f} (F1={best_f1:.5f})")
        print('#' * 50)

        print("\n Evaluating the best model on the validation set...")
        print("\n--- Validation Set Results ---")

        val_pred = (val_proba >= best_threshold).astype(int)
        validation_metrics = binary_metrics(predictions=val_pred, 
                                            proba_predictions=val_proba,
                                            label=self.validation_labels,
                                            dataset="validation") 
        for name, value in validation_metrics.items():
            print(f"{name}: {value:.5f}")

        # test set
        print("\n Evaluating the final model on the HELD-OUT TEST set...")
        test_results = trainer.predict(self.test_dataset)        
        test_logits = torch.tensor(test_results.predictions, dtype=torch.float32).to(self.device)
        
        if use_calibration:
            margin = test_logits[:, 1] - test_logits[:, 0]
            test_proba = torch.sigmoid(self._platt_a * margin + self._platt_b).detach().cpu().numpy()
        else:
            test_proba = torch.softmax(test_logits, dim=1).detach().cpu().numpy()[:, 1]
        
        
        test_pred = (test_proba >= best_threshold).astype(int)
        test_metrics = binary_metrics(predictions=test_pred, 
                                            proba_predictions=test_proba,
                                            label=self.test_labels,
                                            dataset="test") 

        print("\n --- Final Test Set Results ---")
        for name, value in test_metrics.items():
            print(f"{name}: {value:.5f}")
        print("---------------------------------")

        # writting test set predictions
        test_set_prediction = torch.softmax(torch.tensor(test_results.predictions), dim=1).numpy()[:, 1]
        output_df = pd.DataFrame({
            "prediction": test_set_prediction,
            'true_label': self.test_labels
        })
        output_path = os.path.join(self._log_dir, "test_set_predictions.txt")
        output_df.to_csv(output_path, index=False)


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

        # define task and labels
        task = 'classification'
        labels = pd.to_numeric(df[f"LABEL_{self.param.label}"], errors='raise').astype('int64').tolist()
        label_dtype = torch.long

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

    def _load_model_and_tokenizer(self, param: argparse.Namespace) -> Tuple[transformers.PreTrainedModel, transformers.PreTrainedTokenizerBase]:
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

        # load model
        model = AutoModelForSequenceClassification.from_pretrained(self.nlp_model_name, config=config).to(self.device)

        return model, tokenizer

