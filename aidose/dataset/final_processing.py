import numpy as np
from statsmodels.stats.proportion import proportion_confint
import pandas as pd



def add_sum_dosing_error(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a row called 'sum_dosing_error' the the dataset, which is the sum of all values contained in the feature starting with 'label'.

    :df: a dataframe containing the dataset
    :return: the dataset with an additional column as discussed above.
    """

    # find all column that start with 'label'
    label_cols = [col for col in df.columns if col.startswith('label')]
    df["sum_dosing_error"] = df[label_cols].sum(axis=1)

    return df



def add_dosing_error_rate(df:pd.DataFrame) -> pd.DataFrame:
    """
    Add a row called 'dosing_error_rate' the the dataset, which is the proportion of number of dosing error by the number of people at risk (i.e., the number of people that might have an ADE).

    :df: a dataframe contaning the dataset
    :return: the dataset with an additional column as discussed above.
    """
    df['dosing_error_rate'] = df['sum_dosing_error'] / (df['ct_level_ade_population'])

    # in some cased we have ct_level_ade_population of zeros value -> leading to NaN -> we replace with '0' (we are sure there is no error)
    df['dosing_error_rate'] = df['dosing_error_rate'].fillna(0.0)

    return df



def add_wilson_label(df:pd.DataFrame, alpha:float, proba_threshold:float) -> pd.DataFrame:
    """
    Add a column called 'wilson_label' to the DataFrame, which takes a value of 1 if we are (1- alpha)% confident that the dosing error rate is higher than the proba_threshold and '0' otherwise.
    In other words, we assign '1', if the lower bound of the confidence interval is higher than proba_threshold

    :param df: DataFrame containing the dataset with 'label_sum' and 'enrollmentCount' columns.
    :param alpha: Significance level for the confidence interval.
    :return: DataFrame with the 'wilson_label' column added.
    """

    # compute lower bound of the Wilson score interval for each row
    df['wilson_lower'] = df.apply(lambda row: wilson_lower_bound(row['sum_dosing_error'], 
                                                                 row['ct_level_ade_population'], alpha=alpha), axis=1)

    # New label: 1 if lower > proba_threshold
    df['wilson_label'] = (df['wilson_lower'] >= proba_threshold).astype(int)

    # drop the 'wilson_lower' column to avoid data leakage
    df = df.drop(columns=['wilson_lower'])

    return df



def wilson_lower_bound(x, n, alpha=0.05):
    """
    Computes the lower bound of the Wilson confident interval.

    :param x: Number of successes (e.g., number errors in the trials).
    :param n: Total number of person involved in the trials.
    :param alpha: Significance level for the confidence interval (default is 0.05).
    :return: Lower bound of the Wilson score interval.
    """
    if n == 0:
        # in some trials, we have no people at risk -> in this case, we are sure the error rate will be 0 -> lower bound is also zer0
        lower = 0
    else:
        lower, _ = proportion_confint(count=x, nobs=n, alpha=alpha, method='wilson')
    return lower



def dataset_spliting(df:pd.DataFrame, train_percent: float, validation_percent: float, test_percent:float) -> tuple:
        
        """
        Split the DataFrame into training, validation, and test sets.
        Specifically, the dataset is sorted according to the completion date of the study and then 
        Specifically, the first param.train_percent of the data will be used for training, the next param.validation_percent for validation, and the last param.test_percent for testing.

        :param df: Input DataFrame
        :param train_percent: porportion of the dataset used for the training
        :param validation_percent: proportion of the dataset used for the validation
        :param test_percent: proportion of the dataset used for the test set
        :return: Tuple of DataFrames (train, validation, test)
        """

        # convert completionDate column to datetime
        df['completionDate'] = pd.to_datetime(df['completionDate'], format="mixed", errors='coerce')

        # drop rows where criteria is NaT (not a time)
        df = df.dropna(subset=['completionDate'])

        # sort the DataFrame by the criteria column
        df = df.sort_values(by='completionDate')

        # check the validity of the split percentages
        if train_percent + validation_percent + test_percent != 1.0:
            raise ValueError("The sum of train, validation, and test percentages must equal 1.0")
    
        # Calculate split indices
        train_size = int(train_percent * len(df))
        val_size = int(validation_percent * len(df))

        # Split the DataFrame
        df_train = df[:train_size]
        df_val = df[train_size:train_size + val_size]
        df_test = df[train_size + val_size:]

        return df_train, df_val, df_test