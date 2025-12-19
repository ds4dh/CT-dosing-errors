from datasets import DatasetDict
import pandas as pd
import argparse
from typing import Dict


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
        raise NotImplementedError(f"There is not implemented method to prepare the dataset for {param.model}.")


def prepare_dataset_for_xgboost(dataset: DatasetDict, param: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """
    Prepare a dataset for XGBoost.

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
    for set_name, set in dataset.items():

        # convert to datafrane
        df_features = pd.DataFrame(set["features"])

        # filter string feature
        string_cols = [name for name, dtype in set.features['features'].items() if dtype.dtype == 'string']

        # drop text features
        df_features = df_features.drop(columns=string_cols)

        # extract the label
        df_labels = pd.DataFrame(set['labels'])[param.label]

        # concatenate label and target
        df_set = pd.concat([df_features, df_labels], axis=1)

        # delete samples wit missing data in the specific columns
        for feature_name in param.drop_sample:
            df_set = df_set.dropna(subset=[col for col in df_set.columns if col.startswith(feature_name)])

        for feature_prefix in param.add_most_frequent_categorical:
            targeted_columns = [col for col in df_set.columns if col.startswith(feature_prefix)]

            # find missing values
            mask = df_set[targeted_columns].isna().any(axis=1)

            # fill missing data with zeros
            df_set[targeted_columns] = df_set[targeted_columns].fillna(0.0)

            # find most frequent class
            majority_col = df_set[targeted_columns].mean(skipna=True).idxmax()

            # assign '1' to the most frequent class
            df_set.loc[mask, majority_col] = 1.0

            df_set[targeted_columns] = df_set[targeted_columns].astype('bool')

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
    for set_name, set in dataset.items():
        # convert to datafrane
        df_features = pd.DataFrame(set["features"])

        # filter string feature
        string_cols = [name for name, dtype in set.features['features'].items() if dtype.dtype == 'string']
        df_features = df_features[string_cols]

        # extract the label
        df_labels = pd.DataFrame(set['labels'])[param.label]

        # concatenate label and target
        df_set = pd.concat([df_features, df_labels], axis=1)

        # manage missing data
        df_set = df_set.fillna("UNKNOWN").astype(str)

        # TODO check why we have to do that? Why are there missing values in the labels if 'dosing_error_rate' is the label
        df_set[param.label] = df_set[param.label].replace('UNKNOWN', 0.0)

        cleaned_dataset[set_name] = df_set.infer_objects(copy=False)

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

    for split_name, split in dataset.items():
        df_features = pd.DataFrame(split["features"])
        df_labels = pd.DataFrame(split["labels"])[param.label]
        df = pd.concat([df_features, df_labels], axis=1)

        # Start with "keep everything"
        keep_mask = pd.Series(True, index=df.index)

        # For every prefix in drop_sample, drop rows with any NaN in columns starting with that prefix
        for prefix in getattr(param, "drop_sample", []):
            cols = [c for c in df.columns if c.startswith(prefix)]
            if cols:
                keep_mask &= ~df[cols].isna().any(axis=1)

        # Convert mask to absolute row indices and filter the *original* split (preserves schema & types)
        keep_indices = df.index[keep_mask].tolist()
        filtered[split_name] = split.select(keep_indices)

    # Now both preparers will see identical rows across modalities
    return {
        "text_data": prepare_dataset_for_bert(filtered, param),
        "categorical_data": prepare_dataset_for_xgboost(filtered, param),
    }






