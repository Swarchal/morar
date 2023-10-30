from collections.abc import Callable

import numpy as np
import pandas as pd

from morar import utils

"""
Functions(s) for aggregating data from fine level data to higher-level
measurements such as object-level to image or well level data
"""


def aggregate(
    data: pd.DataFrame,
    on: str | list[str],
    method: str | Callable = "median",
    metadata_string: str = "Metadata_",
    prefix: bool = True,
) -> pd.DataFrame:
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
    _check_featuredata(data, on, metadata_string, prefix)
    if isinstance(on, str):
        on = [on]
    # keep track of original column order
    orig_cols = data.columns.tolist()
    fcols = utils.get_featuredata(data, metadata_string, prefix)
    fcols.extend(on)
    mcols = utils.get_metadata(data, metadata_string, prefix)
    for i in on:
        assert i in mcols, "on must be one or more metadata columns"
    fdata_agg = data[fcols].groupby(on).agg(method).reset_index().drop(columns=on)
    mdata_agg = data[mcols].groupby(on).agg("first").reset_index()
    agg_merged = pd.concat([fdata_agg, mdata_agg], axis=1)
    return agg_merged[orig_cols]


def _check_inputs(data, on, method):
    """internal function for aggregate() to check validity of inputs"""
    valid_methods = ["median", "mean"]
    if not isinstance(data, pd.DataFrame):
        raise ValueError("not a a pandas DataFrame")
    if method not in valid_methods:
        msg = "{} is not a valid method, options: median or mean".format(method)
        raise ValueError(msg)
    df_columns = data.columns.tolist()
    if isinstance(on, str):
        if on not in df_columns:
            raise ValueError("{} not a column in df".format(on))
    elif isinstance(on, list):
        for col in on:
            if col not in df_columns:
                raise ValueError("{} not a column in df".format(col))


def _check_featuredata(data, on, metadata_string, prefix):
    """
    Check feature data is numerical
    """
    feature_cols = utils.get_featuredata(data, metadata_string, prefix)
    cols_to_check = [col for col in feature_cols if col not in [on]]
    df_to_check = data[cols_to_check]
    is_number = np.vectorize(lambda x: np.issubdtype(x, np.number))
    if any(is_number(df_to_check.dtypes) == False):
        # return column name
        nn_col = df_to_check.columns[is_number(df_to_check.dtypes) == False]
        err_msg = "{} is a non-numeric featuredata columns".format(nn_col)
        raise ValueError(err_msg)
