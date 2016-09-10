from morar import stats
from morar import utils
import pandas as pd
import numpy as np


def _check_control(df, plate_id, compound="Metadata_compound",
                  neg_compound="DMSO"):
    """
    check each plate contains at least 1 negative control value. Raise an error
    if this is not the case.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame

    plate_id : string
        column containing plate ID/label

    compound : string (default="Metadata_compound")
        column containing compound name/ID

    neg_compound : string (default="DMSO")
        name of negative control compound in compound col
    """
    for name, group in df.groupby(plate_id):
        group_cmps = list(set(group[compound]))
        if neg_compound not in group_cmps:
            raise ValueError("{} does not contain any negative control values".format(name))


def normalise(df, plate_id, compound="Metadata_compound",
              neg_compound="DMSO", method="subtract", **kwargs):
    """
    Normalise values against negative controls values per plate.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame

    plate_id : string
        column containing plate ID/label

    compound : string (default="Metadata_compound")
        column containing compound name/ID

    neg_compound : string (default="DMSO")
        name of negative control compound in compound col

    method :string (default="subtract")
        method to normalise against negative control

    **kwargs : additional arguments to utils.get_featuredata/metadata

    Returns
    --------
    df_out : pandas DataFrame
        DataFrame of normalised feature values
    """
    valid_methods = ["subtract", "divide"]
    if method not in valid_methods:
        raise ValueError("Invalid method, options: 'subtract', 'divide'")
    # check there are some negative controls on each plate
    _check_control(df, plate_id, compound, neg_compound)
    # identify feature columns
    f_cols = utils.get_featuredata(df, **kwargs)
    # dataframe for output
    df_out = pd.DataFrame()
    # group by plate
    grouped = df.groupby(plate_id, as_index=False)
    # calculate the average negative control values for each plate
    for _, group in grouped:
        dmso_med = group[group[compound] == neg_compound][f_cols].median()
        assert len(dmso_med) == group[f_cols].shape[1]
        if method == "subtract":
            tmp = group[f_cols].sub(dmso_med)
        if method == "divide":
            tmp = group[f_cols].divide(dmso_med)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, tmp])
    # check we have not lost any rows
    assert df.shape[0] == df_out.shape[0]
    return df_out


def robust_normalise(df, plate_id, compound="Metadata_compound",
                     neg_compound="DMSO", **kwargs):
    """
    Method used in the Carpenter lab. Substract the median feature value for
    each plate negative control from the treatment feature value and divide by
    the median absolute deviation.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame

    plate_id : string
        column containing plate ID/label

    compound : string (default="Metadata_compound")
        column containing compound name/ID

    neg_compound : string (default="DMSO")
        name of negative control compound in compound col

    **kwargs : additional arguments to utils.get_featuredata/metadata

    Returns
    --------
    df_out : pandas DataFrame
        DataFrame of normalised feature values
    """
    _check_control(df, plate_id, compound, neg_compound)
    f_cols = utils.get_featuredata(df, **kwargs)
    grouped = df.groupby(plate_id, as_index=False)
    df_out = pd.DataFrame()
    # calculate the average negative control values per plate_id
    for _, group in grouped:
        # find the median and mad dmso value for each plate
        dmso_vals = group[group[compound] == neg_compound]
        dmso_med = dmso_vals[f_cols].median().values
        dmso_mad = dmso_vals[f_cols].apply(lambda x: stats.mad(x), axis=0).values
        assert len(dmso_med) == group[f_cols].shape[1]
        # subtract each row of the group by that group's DMSO values
        tmp = group[f_cols].sub(dmso_med)
        # divide by the MAD of the negative control
        tmp = tmp.apply(lambda x: (x/dmso_mad)*1.4826, axis=1)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, tmp])
    # check we have not lost any rows
    assert df.shape[0] == df_out.shape[0]
    return df_out
