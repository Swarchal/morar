from morar import utils
import numpy as np
import pandas as pd

def aggregate(df, on, method="median"):
    """
    Aggregate dataset

    Parameters
    -----------
    df : pandas DataFrame
        DataFrame

    on : string or list of strings
        column(s) with which to group by and aggregate the dataset.

    method : string (default="median")
        method to average each group. options = "median" or "mean"

    Returns
    -------
    agg_df : pandas DataFrame
        aggregated dataframe, with a row per value of 'on'
    """
    _check_inputs(df, on, method)
    # keep track of original column order
    df_columns = df.columns.tolist()
    grouped = df.groupby(on, as_index=False)
    if method == "mean":
        agg = grouped.aggregate(np.mean)
    if method == "median":
        agg = grouped.aggregate(np.median)
    df_metadata = df[utils.get_metadata(df)].copy()
    # add indexing column to metadata if not already present
    df_metadata[on] = df[on]
    # drop metadata to the same level as aggregated data
    df_metadata.drop_duplicates(subset=on, inplace=True)
    merged_df = pd.merge(agg, df_metadata, on=on)
    # re-arrange to columns are in original order
    merged_df = merged_df[df_columns]
    return merged_df


def _check_inputs(df, on, method):
    valid_methods = ["median", "mean"]
    if not isinstance(df, pd.DataFrame):
        raise ValueError("not a a pandas DataFrame")
    if method not in valid_methods:
        raise ValueError("{} is not a valid method, options: median or mean".format(method))
    df_columns = df.columns.tolist()
    if isinstance(on, str):
        if on not in df_columns:
            raise ValueError("{} not a column in df".format(on))
    elif isinstance(on, list):
        for col in on:
            if col not in df_columns:
                raise ValueError("{} not a column in df".format(col))
