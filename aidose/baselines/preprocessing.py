from datasets import DatasetDict, Value
import pandas as pd
import argparse
from typing import Dict
from utils import *



def dataset_preparation(dataset: DatasetDict, param: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Prepare a dataset for the requested model and following the missing data strategy defined in param.

    Dispatches to a model specific routine based on param.model

    Parameters
    ----------
    dataset : datasets.Dataset or datasets.DatasetDict
        Hugging Face dataset (or split dict) containing raw examples.
    param : argparse.Namespace
        Configuration namespace. Must include at least ``model`` and any options
        required by the target model (e.g., label name).
    
    Returns
    -------
    Any
        The dataset transformed into the format expected by the target model.
    """
    if param.model == 'XGBoost':
        return prepare_dataset_for_xgboost(dataset, param)
    elif param.model == 'ClinicalModernBERT':
        return prepare_dataset_for_bert(dataset, param)
    elif param.model == 'LateFusionModel':
        return prepare_dataset_for_latefusion_model(dataset, param)
    else:
        raise NotImplementedError(f"There is no implemented method to prepare the dataset for {param.model}.")



def prepare_dataset_for_xgboost(dataset: DatasetDict, param: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Prepare a dataset for XGBoost and SVM.

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


def prepare_dataset_for_bert(dataset: DatasetDict, param: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Convert a Hugging Face `DatasetDict` into per-split pandas DataFrames that keep only string features and the target label, with light cleaning applied.

    For each split (e.g., "train", "validation", "test"), this function:
      - selects columns whose schema type is `Value("string")` plus `param.label`,
      - converts the split to a pandas DataFrame,
      - fills missing values with "UNKNOWN",
      - casts all columns to `str` for uniform downstream text processing,

    Parameters
    ----------
    dataset : datasets.DatasetDict
        Mapping from split names to `datasets.Dataset`. Each split must define
        its schema in `.features`, with text columns typed as `Value("string")`.
    param : argparse.Namespace
        Experiment/config object that must contain:
        - `label` (str): name of the target column to retain.

    Returns
    -------
    dict[str, pandas.DataFrame]
        A dictionary keyed by split name containing cleaned DataFrames.
    """

    cleaned_dataset = DatasetDict()

    # iterate over each split
    for set_name, data_split in dataset.items():

        # filter text features
        string_cols = [name for name, feature in data_split.features.items() if isinstance(feature, Value) 
                       and feature.dtype == "string" and not name.startswith("METADATA_")]

        # we dont use the full protocol
        string_cols.remove("FEATURE_protocolPdfText")

        cols_to_keep = list(dict.fromkeys(string_cols + [f"LABEL_{param.label}"])) 

        nctID = data_split.select_columns(['METADATA_nctId'])
        
        dataset_reduced = data_split.select_columns(cols_to_keep)
        
        # concert to dataframe
        df_set = dataset_reduced.to_pandas()

        # fill missing values
        df_set = df_set.fillna("UNKNOWN").astype(str)

        cleaned_dataset[set_name] = df_set
        cleaned_dataset[set_name + '_nctid'] = nctID

    return cleaned_dataset


def prepare_dataset_for_latefusion_model(dataset: DatasetDict, param: argparse.Namespace) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Prepare a dataset for a late-fusion model by enforcing a shared sample selection
    across modalities.

    Parameters
    ----------
    dataset : datasets.DatasetDict
        Input dataset with standard train/validation/test splits.
    param : argparse.Namespace
        Configuration namespace defining ``label`` and optionally ``drop_sample``.

    Returns
    -------
    dict[str, dict[str, pandas.DataFrame]]
        Dictionary with keys ``"text_data"`` and ``"categorical_data"``, each containing
        per-split DataFrames that are row-aligned across modalities.
    """

    # Now both preparers will see identical rows across modalities
    return {
        "text_data": prepare_dataset_for_bert(dataset, param),
        "categorical_data": prepare_dataset_for_xgboost(dataset, param),
    }






