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
    elif param.model == 'SVM':
        return prepare_dataset_for_svm(dataset, param)
    elif param.model == 'ClinicalModernBERT':
        return prepare_dataset_for_bert(dataset, param)
    elif param.model == 'LateFusionModel':
        return prepare_dataset_for_latefusion_model(dataset, param)
    else:
        raise NotImplementedError(f"There is not implemented method to prepare the dataset for {param.model}.")


def prepare_dataset_for_svm(dataset: DatasetDict, param: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Prepare a dataset for XGBoost and SVM.

    Per split, this:
        - drops string (text) features,
        - keeps only the target label ``param.label`` (removes {sum_dosing_error, dosing_error_rate, wilson_label} \\ {param.label}),
        - handles missing data:
            for each prefix in ``param.drop_sample``: drop rows with NaN in columns starting with that prefix;
            for each prefix in ``param.add_most_frequent_categorical``: treat matching one-hot columns as a group, fill NaN with 0,
            set the majority column to 1 for previously-missing rows, cast group to bool.
    """

    cleaned_dataset = DatasetDict()

    # iterate over each split
    for set_name, data_split in dataset.items():

        # drop string features, METADATA columns and not targeted labels
        string_cols = [name for name, feature in data_split.features.items()
                       if isinstance(feature, Value) and feature.dtype == "string"]
        metadata_cols = [name for name in data_split.column_names if name.startswith("METADATA_")]
        label_cols_to_drop = [name for name in data_split.column_names
                              if name.startswith("LABEL_") and name != f"LABEL_{param.label}"]

        cols_to_drop = set(string_cols) | set(metadata_cols) | set(label_cols_to_drop)
        dataset_reduced = data_split.remove_columns(list(cols_to_drop))

        # convert to dataframe
        df_set = dataset_reduced.to_pandas()

        # multihot encoding for specific categorical features (robust to NaNs)
        need_to_encode = ['FEATURE_armGroupTypes', 'FEATURE_phases', 'FEATURE_interventionTypes']
        for col in need_to_encode:
            if col not in df_set.columns:
                continue

            # treat missing lists as empty lists
            s = df_set[col].apply(lambda x: x if isinstance(x, (list, tuple, set, np.ndarray))else ([] if pd.isna(x) else [x]))
            exploded = s.explode()

            counts = pd.crosstab(exploded.index, exploded).add_prefix(col + '_').astype(float)

            # Ensure all rows are present and fill absent one-hot entries with 0
            counts = counts.reindex(df_set.index, fill_value=0.0)

            df_set = pd.concat([df_set.drop(columns=[col]), counts], axis=1)

        # fill NaN values for the dosing error rate label -> TODO fix that
        label_col = f"LABEL_{param.label}"
        if label_col in df_set.columns:
            df_set[label_col] = df_set[label_col].fillna(0.0)

        # specific preprocessing
        if 'FEATURE_healthyVolunteers' in df_set.columns:
            df_set['FEATURE_healthyVolunteers'] = df_set['FEATURE_healthyVolunteers'].astype(float)
        if 'FEATURE_oversightHasDmc' in df_set.columns:
            df_set['FEATURE_oversightHasDmc'] = df_set['FEATURE_oversightHasDmc'].astype(float)

        cleaned_dataset[set_name] = df_set.infer_objects(copy=False)

    # ---- Minimal fix for SVC: impute remaining NaNs (fit on train only; apply to all splits) ----
    if "train" in cleaned_dataset:
        train_df = cleaned_dataset["train"]

        # exclude label from imputation statistics
        label_col = f"LABEL_{param.label}"
        feature_cols = [c for c in train_df.columns if c != label_col]

        # compute per-column fill values on train
        fill_values = {}
        for c in feature_cols:
            # common case: one-hot columns after crosstab -> fill with 0
            if any(c.startswith(prefix + "_") for prefix in ['FEATURE_armGroupTypes', 'FEATURE_phases', 'FEATURE_interventionTypes']):
                fill_values[c] = 0.0
            else:
                # numeric columns: use median
                fill_values[c] = train_df[c].median() if pd.api.types.is_numeric_dtype(train_df[c]) else 0.0

        # apply to all splits
        for split_name in list(cleaned_dataset.keys()):
            df = cleaned_dataset[split_name]
            df[feature_cols] = df[feature_cols].fillna(fill_values)

            # if anything is still NaN, fail early with useful debug info
            remaining = df[feature_cols].isna().sum()
            remaining = remaining[remaining > 0]
            if len(remaining) > 0:
                raise ValueError(f"NaNs remain in split '{split_name}' after imputation: {remaining.to_dict()}")

            cleaned_dataset[split_name] = df

    return cleaned_dataset

def prepare_dataset_for_xgboost(dataset: DatasetDict, param: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Prepare a dataset for XGBoost and SVM.

    Per split, this:
        - drops string (text) features,
        - keeps only the target label ``param.label`` (removes {sum_dosing_error, dosing_error_rate, wilson_label} \\ {param.label}),
        - handles missing data:
            for each prefix in ``param.drop_sample``: drop rows with NaN in columns starting with that prefix;
            for each prefix in ``param.add_most_frequent_categorical``: treat matching one-hot columns as a group, fill NaN with 0, set the majority column to 1 for previously-missing rows, cast group to bool.

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
            # col = 'FEATURE_' + feature_prefix
            exploded = df_set[col].explode()

            counts = pd.crosstab(exploded.index, exploded)
            counts = counts.add_prefix(col + '_').astype(float)
            df_set = pd.concat([df_set.drop(columns=[col]), counts], axis=1)
        
        # fill NaN values for the dosing error rate label -> TODO fix that
        df_set[f"LABEL_{param.label}"] = df_set[f"LABEL_{param.label}"].fillna(0.0)
        
        # specific preprocessing
        df_set['FEATURE_healthyVolunteers'] = df_set['FEATURE_healthyVolunteers'].astype(float)
        df_set['FEATURE_oversightHasDmc'] = df_set['FEATURE_oversightHasDmc'].astype(float)

        cleaned_dataset[set_name] = df_set.infer_objects(copy=False)

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
        
        dataset_reduced = data_split.select_columns(cols_to_keep)
        
        # concert to dataframe
        df_set = dataset_reduced.to_pandas()

        # fill NaN values for the dosing error rate label -> TODO fix that
        df_set[f"LABEL_{param.label}"] = df_set[f"LABEL_{param.label}"].fillna(0.0)

        # fill missing values
        df_set = df_set.fillna("UNKNOWN").astype(str)

        cleaned_dataset[set_name] = df_set

    return cleaned_dataset


def prepare_dataset_for_latefusion_model(dataset: DatasetDict, param: argparse.Namespace) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Prepare a dataset for a late-fusion model by enforcing a shared sample selection
    across modalities.

    For each split, rows with missing values in features whose names start with any
    prefix in ``param.drop_sample`` are removed. The filtered ``DatasetDict`` is then
    passed to the text- and tabular-specific preparation functions so that both
    modalities are built from the exact same samples.

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

    filtered = DatasetDict()

    #for split_name, split in dataset.items():
    #    df_features = pd.DataFrame(split["features"])
    #    df_labels = pd.DataFrame(split["labels"])[param.label]
    #    df = pd.concat([df_features, df_labels], axis=1)

        # Start with "keep everything"
    #    keep_mask = pd.Series(True, index=df.index)

        # For every prefix in drop_sample, drop rows with any NaN in columns starting with that prefix
    #    for prefix in getattr(param, "drop_sample", []):
    #        cols = [c for c in df.columns if c.startswith(prefix)]
    #        if cols:
    #            keep_mask &= ~df[cols].isna().any(axis=1)

        # Convert mask to absolute row indices and filter the *original* split (preserves schema & types)
    #
    # keep_indices = df.index[keep_mask].tolist()
    #    filtered[split_name] = split.select(keep_indices)

    # Now both preparers will see identical rows across modalities
    return {
        "text_data": prepare_dataset_for_bert(dataset, param),
        "categorical_data": prepare_dataset_for_xgboost(dataset, param),
    }






