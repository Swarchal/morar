from morar import stats
import pandas as pd
import numpy as np


def get_featuredata(df, metadata_prefix="Metadata"):
    """
    identifies columns in a dataframe that are not labelled with the
    metadata prefix. Its assumed everything not labelled metadata is
    featuredata
    -----------------------------------------------------------------------
    df: dataframe
    metadata_prefix: string, prefix for metadata columns
    """
    f_cols = [i for i in df.columns if not i.startswith(metadata_prefix)]
    return f_cols


def check_control(df, plate_id, compound="Metadata_compound",
                  neg_compound="DMSO"):
    """
    check each plate contains at least 1 negative control value
    -------------------------------------------------------------------------
    df: dataframe
    plate_id: string for column containing plate ID/label
    compound: string for column containing compound name/ID
    neg_compound: string name of negative control compound in compound col
    """
    for name, group in df.groupby(plate_id):
        group_cmps = list(set(group[compound]))
        if neg_compound not in group_cmps:
            raise RuntimeError("{} does not contain any negative control values".format(name))


def normalise(df, plate_id, compound="Metadata_compound",
              neg_compound="DMSO", method="divide",
              metadata_prefix="Metadata"):
    """
    Normalise values against DMSO values per plate.
    -----------------------------------------------------------------------
    df: dataframe
    plate_id: string for column containing plate ID/label
    compound: string for column containing compound name/ID
    neg_compound: string name of negative control compound in compound col
    method: method to normalise against negative control
    metadata_prefix: string, prefix for metadata columns
    """
    valid_methods = ["subtract", "divide"]
    if method not in valid_methods:
        raise ValueError("Invalid method, options: 'subtract', 'divide'")
    # check there are some negative controls on each plate
    check_control(df, plate_id, compound, neg_compound)
    # identify feature columns
    f_cols = get_featuredata(df, metadata_prefix)
    df_out = pd.DataFrame()
    # group by plate
    grouped = df.groupby(plate_id, as_index=False)
    # calculate the average DMSO values for each plate
    for _, group in grouped:
        # TODO keep metadata columns
        dmso_med = list(group[group[compound] == neg_compound][f_cols].median().values)
        assert len(dmso_med) == group[f_cols].shape[1]
        if method == "subtract":
            # TODO fix subtraction calculation
            tmp = group[f_cols].sub(dmso_med)
        if method == "divide":
            tmp = group[f_cols].divide(dmso_med)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, tmp])
    # check we have not lost any rows
    assert df.shape[1] == df_out.shape[1]
    return df_out


def robust_normalise(df, plate_id, compound="Metadata_compound",
                     neg_compound="DMSO", metadata_prefix="Metadata"):
    """
    Method used in the Carpenter lab. Substract the median feature value for
    each plate DMSO from the treatment feature value and divide by the
    median absolute deviation.
    Returns a pandas DataFrame.
    -----------------------------------------------------------------------
    df: dataframe
    plate_id: string for column containing plate ID/label
    compound: string for column containing compound name/ID
    neg_compound: string name of negative control compound in compound col
    metadata_prefix: string, prefix for metadata columns
    """
    check_control(df, plate_id, compound, neg_compound)
    f_cols = get_featuredata(df, metadata_prefix)
    grouped = df.groupby(plate_id, as_index=False)
    df_out = pd.DataFrame()
    # calculate the average DMSO values per plate_id
    for _, group in grouped:
        # find the median and mad dmso value for each plate
        dmso_vals = group[group[compound] == neg_compound]
        dmso_med = dmso_vals[f_cols].median().values.tolist()
        dmso_mad = dmso_vals[f_cols].apply(lambda x: stats.mad(x), axis=0).values.tolist()
        assert len(dmso_med) == group[f_cols].shape[1]
        # subtract each row of the group by that group's DMSO values
        tmp = group[f_cols].sub(dmso_med, axis=1)
        # divide by the MAD of the negative control
        tmp = tmp.apply(lambda x: (x/dmso_mad)*1.4826, axis=1)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, tmp])
    # check we have not lost any rows
    assert df.shape[1] == df_out.shape[1]
    return df_out


def scale_features(df):
    """
    scale and centre features
    --------------------------------------------------------------------------
    df: dataframe
    """
    pass
