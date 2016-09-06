import numpy as np
import pandas as pd

def get_featuredata(df, metadata_prefix="Metadata"):
    """
    identifies columns in a dataframe that are not labelled with the
    metadata prefix. Its assumed everything not labelled metadata is
    featuredata

    @param df dataframe
    @param metadata_prefix string, prefix for metadata columns
    @return list of column names
    """
    f_cols = [i for i in df.columns if not i.startswith(metadata_prefix)]
    return f_cols


def get_metadata(df, metadata_prefix="Metadata"):
    """
    identifies column in a dataframe that are labelled with the metadata_prefix

    @param df pandas DataFrame
    @param metadata_prefix metadata prefix in column name
    @return list of column names
    """
    m_cols = [i for i in df.columns if i.startswith(metadata_prefix)]
    return m_cols


def is_all_nan(df):
    """
    Returns column name if all values in that column are np.nan

    @param df pandas DataFrame
    """
    is_null = df.isnull().sum()
    nrows = df.shape[0]
    out_cols = []
    for i in is_null.index:
        if is_null[i] == nrows:
            out_cols.append(i)
    return out_cols


def get_image_quality(df):
    """
    Returns list of column names from the CelLProfiler ImageQuality module
    that are present in df.

    @param df pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(df, "is not a pandas DataFrame")
    colnames = df.columns.tolist()
    im_qc_cols = [col for col in colnames if "ImageQuality" in col]
    if len(im_qc_cols) == 0:
        raise ValueError("no ImageQuality measurements found")
    else:
        return im_qc_cols