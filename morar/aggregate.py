from morar import utils
import numpy as np
import pandas as pd

"""
Functions(s) for aggregating data from fine level data to higher-level
measurements such as object-level to image or well level data
"""

def aggregate(data, on, method="median", **kwargs):
    """
    Aggregate dataset

    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    on : string or list of strings
        column(s) with which to group by and aggregate the dataset.
    method : string (default="median")
        method to average each group. options = "median" or "mean"
    **kwargs : additional args to utils.get_metadata / utils.get_featuredata

    Returns
    -------
    agg_df : pandas DataFrame
        aggregated dataframe, with a row per value of 'on'
    """
    _check_inputs(data, on, method)
    _check_featuredata(data, on, **kwargs)
    # keep track of original column order
    df_cols = data.columns.tolist()
    grouped = data.groupby(on, as_index=False)
    if method == "mean":
        agg = grouped.aggregate(np.mean)
    if method == "median":
        agg = grouped.aggregate(np.median)
    df_metadata = data[utils.get_metadata(data, **kwargs)].copy()
    # add indexing column to metadata if not already present
    df_metadata[on] = data[on]
    # drop metadata to the same level as aggregated data
    df_metadata.drop_duplicates(subset=on, inplace=True)
    # merge aggregated and feature data
    merged_df = pd.merge(agg, df_metadata, on=on, how="outer",
                         suffixes=("remove_me", ""))
    # merge untracked columns with merged data
    merged_df = merged_df[df_cols]
    # re-arrange to columns are in original order
    assert len(merged_df.columns) == len(data.columns)
    return merged_df


def _check_inputs(data, on, method):
    """ internal function for aggregate() to check validity of inputs """
    valid_methods = ["median", "mean"]
    if not isinstance(data, pd.DataFrame):
        raise ValueError("not a a pandas DataFrame")
    if method not in valid_methods:
        raise ValueError("{} is not a valid method, options: median or mean".format(method))
    df_columns = data.columns.tolist()
    if isinstance(on, str):
        if on not in df_columns:
            raise ValueError("{} not a column in df".format(on))
    elif isinstance(on, list):
        for col in on:
            if col not in df_columns:
                raise ValueError("{} not a column in df".format(col))


def _check_featuredata(data, on, **kwargs):
    """
    Check feature data is numerical
    """
    feature_cols = utils.get_featuredata(data, **kwargs)
    cols_to_check = [col for col in feature_cols if col not in [on]]
    df_to_check = data[cols_to_check]
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    if any(is_number(df_to_check.dtypes) == False):
        raise ValueError("non-numeric column found in feature data")

