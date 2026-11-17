from aidose import END_POINT_HF_DATASET_PATH
from aidose.baselines.preprocessing import dataset_preparation
from datasets import DatasetDict, load_from_disk
import argparse
from our_xgboost import OurXGBoost
from our_clinicalModernBERT import OurClinicalModernBERT
from sklearn.preprocessing import StandardScaler
from LateFusionMultimodal import LateFusionModel
from datasets import load_dataset, Dataset
import pandas as pd
from datasets import DatasetDict, Value
from statsmodels.stats.proportion import proportion_confint
import numpy as np
from constants import XGB_DIR



import os


THRESHOLD_P0_LIST = [0.0, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005]


def parse_args():

    parser = argparse.ArgumentParser()

    # random seed
    parser.add_argument('--random_seed', action='store', type=int, default=42, 
                        help='random seed used to conduct these experiments.')
    
    # label
    parser.add_argument('--label', action='store', type=str, default='wilson_label',  
                        help='Name of the feature that should be considered as the label. Only wilson_label is possible here.')

    parser.add_argument('--model', action='store', type=str, default='XGBoost', 
                            help='model used to conduct the predictions. Potential values are: XGBoost.')

    

    ################################
    # specific to XGBoost
    ################################
    parser.add_argument('--num_trials', action='store', type=int, default=100,
                        help='Number of trials used to conduct the hyperparemeter search associated to the XGBoost model.')
    


    args = parser.parse_args()

    return args


def construct_customized_wilson_label(data_split, p0, alpha=0.05):
    """Binary label: 1 if at least one dosing error is reported and the Wilson lower bound
    of the dosing-error rate exceeds p0; 0 otherwise (including zero population)."""
    count = np.asarray(data_split['LABEL_sum_dosing_errors'])
    nobs = np.asarray(data_split['LABEL_ct_level_ade_population'])

    new_label = np.zeros(len(nobs), dtype=int)
    valid = nobs > 0
    lb, _ = proportion_confint(count[valid], nobs[valid], alpha=alpha, method='wilson')
    new_label[valid] = ((lb > p0) & (count[valid] > 0)).astype(int)
    return new_label



def dataset_preparation_ablation(dataset: DatasetDict, param: argparse.Namespace, p0: float) -> dict[str, pd.DataFrame]:
    """
    Prepare a dataset for XGBoost according to a specific p0 for the threshold.

    Per split, this:
        - drops string (text) features,
        - keeps only the target label ``param.label`` (removes {sum_dosing_error, dosing_error_rate, wilson_label} \\ {param.label}),

    Parameters
    ----------
    dataset : datasets.DatasetDict
    param : argparse.Namespace containing all required hyperaparameter

    Returns
    -------
    dict[str, pandas.DataFrame]
        Cleaned splits as pandas DataFrames.
    
    """

    cleaned_dataset = DatasetDict()

    # iterate over each split
    for set_name, data_split in dataset.items():

        # extract nct_id
        nctID = data_split.select_columns(['METADATA_nctId'])

        new_label = construct_customized_wilson_label(data_split, p0=p0)
        data_split = data_split.remove_columns('LABEL_wilson_label').add_column('LABEL_wilson_label', new_label.tolist())
        
        # drop string features, METADATA columns an not targeted labels 
        string_cols = [name for name, feature in data_split.features.items() if isinstance(feature, Value) and feature.dtype == "string"]
        metadata_cols = [name for name in data_split.column_names if name.startswith("METADATA_")]
        label_cols_to_drop = [name for name in data_split.column_names if name.startswith("LABEL_") and name != f"LABEL_{param.label}"]

        cols_to_drop = set(string_cols) | set(metadata_cols) | set(label_cols_to_drop)      
        dataset_reduced = data_split.remove_columns(list(cols_to_drop))

        # convert to datafrane
        df_set = dataset_reduced.to_pandas()

        # multihot encoding for specific categorical features
        need_to_encode = ['FEATURE_armGroupTypes', 'FEATURE_phases', 'FEATURE_interventionTypes']
        for col in need_to_encode:
            exploded = df_set[col].explode()

            counts = pd.crosstab(exploded.index, exploded)
            counts = counts.add_prefix(col + '_').astype(float)
            df_set = pd.concat([df_set.drop(columns=[col]), counts], axis=1)
        
        # specific preprocessing
        df_set['FEATURE_healthyVolunteers'] = df_set['FEATURE_healthyVolunteers'].astype(float)
        df_set['FEATURE_oversightHasDmc'] = df_set['FEATURE_oversightHasDmc'].astype(float)

        cleaned_dataset[set_name] = df_set.infer_objects(copy=False)
        cleaned_dataset[set_name + '_nctid'] = nctID

    return cleaned_dataset

def main():

    # hyperparameter loading
    param = parse_args()

    # loading loading the dataset
    raw_dataset = DatasetDict({
        "train": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "train")),
        "validation": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "validation")),
        "test": load_from_disk(os.path.join(END_POINT_HF_DATASET_PATH, "test")),
     })


    for p0 in THRESHOLD_P0_LIST:

        print('#'*70)
        print(f"Running ablation study for threshold p_0 = {p0} ...")
        print('#'*70)
        
        # prepare the dataste according to the threshold p_0
        dataset = dataset_preparation_ablation(param=param, dataset=raw_dataset, p0=p0)

        logdir = os.path.join(XGB_DIR, f"ablation_study_p0_{p0}")
        
        # model construction
        model = OurXGBoost(param=param, dataset=dataset, logdir=logdir)

        model.hyperparameter_search_and_evaluation()

        model.save_test_predictions()
        



if __name__ == '__main__':
    main()