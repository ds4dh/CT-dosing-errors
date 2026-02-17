from aidose import END_POINT_HF_DATASET_PATH
from aidose.baselines.preprocessing import dataset_preparation
from datasets import DatasetDict, load_from_disk
import argparse
from our_xgboost import OurXGBoost
from our_clinicalModernBERT import OurClinicalModernBERT
from sklearn.preprocessing import StandardScaler
from LateFusionMultimodal import LateFusionModel
from datasets import load_dataset, Dataset
from our_svm import OurSVM

import os


def parse_args():

    parser = argparse.ArgumentParser()

    # random seed
    parser.add_argument('--random_seed', action='store', type=int, default=42, 
                        help='random seed used to conduct these experiments.')
    
    # label
    parser.add_argument('--label', action='store', type=str, default='wilson_label',  
                        help='Name of the feature that should be considered as the label. Only wilson_label is possible here.')
    
    #######################################
    # ML model
    #######################################
    parser.add_argument('--model', action='store', type=str, default='SVM', 
                        help='model used to conduct the predictions. Potential values are: SVM, XGBoost, ClinicalModernBERT, LateFusionModel')
    

    ################################
    # specific to XGBoost
    ################################
    parser.add_argument('--num_trials', action='store', type=int, default=2,
                        help='Number of trials used to conduct the hyperparemeter search associated to the XGBoost model.')
    
    
    #######################################
    # specific to ClinicalModernBERT
    #######################################
    parser.add_argument('--max_length', action='store', type=int, default=8192, # 8192, 16
                        help='Maximum lenght used by NLP model. Please note that this value is clipped to the maximum length allowed by the NLP model.')

    parser.add_argument('--early_stopping_patience', action='store', type=int, default=5,help='Number of epochs with no improvement after which training will be stopped.')

    parser.add_argument('--num_epoch', action='store', type=int, default=100, # 100
                        help='Number of epochs the NLP model will be trained.')
    
    parser.add_argument('--learning_rate', action='store', type=float, default=2.5e-5,
                        help='Learning rate used to train the BERT model.')
    parser.add_argument('--weight_decay', action='store', type=float, default=0.01,
                        help='weight decay used to train the BERT model.')
    parser.add_argument('--eval_and_save_steps', action='store', type=int, default=88,    # 88
                        help='Step-based evaluation.')
    parser.add_argument('--negative_sampling_ratio', action='store', type=float, default=0.5,
                        help='Negative sampling ratio.')
    
    # batch size
    parser.add_argument('--train_batch_size', action='store', type=int, default=32,     # official 32
                        help='Global batch size used to train the BERT model.')
    parser.add_argument('--eval_batch_size', action='store', type=int, default=32,      # official 32
                        help='Global batch size used to evaluate the BERT model.')
    parser.add_argument('--gradient_accumulation_step', action='store', type=int, default=4,
                        help='Gradient accumulation step.')
    
    
    ##################################
    # specific to LateFusion model
    ##################################
    parser.add_argument('--load_xgboost_model', action='store', type=bin, default=True,
                        help='Define if the XGBoost model included in the LateFusionModel should be loaded or a new hyperparameter search should be conducted.')
    
    parser.add_argument('--multimodal_train_bert_model', action='store', type=bool, default=False,    
                        help='If set to True we fine-tuned the BERT model in the Multimodal model. Otherwise, we load an already fine-tuned model.')
    parser.add_argument('--late_fusion_num_trials', action='store', type=int, default=100, 
                        help='Number of trials used to fine-tune the weight used to combine XGBoost and BERT predictions in the LateFusionModel.')


    args = parser.parse_args()

    return args


def main():

    # hyperparameter loading
    param = parse_args()

    # loading loading the dataset
    dataset = DatasetDict({
        "train": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "train")),
        "validation": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "validation")),
        "test": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "test")),
     })
    
    # prepare the dataste according to the missing data strategy and the chosen model
    dataset = dataset_preparation(param=param, dataset=dataset)

    # train and evaluate the specific model
    if param.model == 'XGBoost':

        # model construction
        model = OurXGBoost(param=param, dataset=dataset)

        # hyperparameter search and model evaluation
        model.hyperparameter_search_and_evaluation()

        model.load_and_evaluate()
    
    elif param.model == 'SVM':
        # model construction
        model = OurSVM(param=param, dataset=dataset)

        # hyperparameter search and model evaluation
        model.hyperparameter_search_and_evaluation()

        model.load_and_evaluate()
        
    elif param.model == 'ClinicalModernBERT':

        # model construction
        model = OurClinicalModernBERT(param=param, dataset=dataset)

        # model fine-tuning and evaluation
        # model.train_and_evaluate()

        # loading fine-tuned model and evaluate it
        model.load_and_evaluate()

    elif param.model == 'LateFusionModel':
        model = LateFusionModel(param=param, dataset=dataset)

        # trainining, hyperparameter search and evaluation
        model.train_and_evaluate()


if __name__ == '__main__':
    main()