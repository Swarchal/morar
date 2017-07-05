from morar import stats
from morar import utils
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

# stop copy warning as not using chained assignment
pd.options.mode.chained_assignment = None  # default='warn'

def _check_control(data, plate_id, compound="Metadata_compound",
                   neg_compound="DMSO"):
    """
    check each plate contains at least 1 negative control value. Raise an error
    if this is not the case.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    plate_id : string
        column containing plate ID/label
    compound : string (default="Metadata_compound")
        column containing compound name/ID
    neg_compound : string (default="DMSO")
        name of negative control compound in compound col
    """
    for name, group in data.groupby(plate_id):
        group_cmps = list(set(group[compound]))
        if neg_compound not in group_cmps:
            msg = "{} does not contain any negative control values".format(name)
            raise RuntimeError(msg)


def robust_normalise(data, plate_id, compound="Metadata_compound",
                     neg_compound="DMSO", **kwargs):
    """
    Method used in the Carpenter lab. Substract the median feature value for
    each plate negative control from the treatment feature value and divide by
    the median absolute deviation.

    Parameters
    ----------
    data : pandas DataFrame
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
    _check_control(data, plate_id, compound, neg_compound)
    f_cols = utils.get_featuredata(data, **kwargs)
    grouped = data.groupby(plate_id, as_index=False)
    df_out = pd.DataFrame()
    # calculate the average negative control values per plate_id
    for _, group in grouped:
        # find the median and mad dmso value for each plate
        dmso_vals = group[group[compound] == neg_compound]
        dmso_med = dmso_vals[f_cols].median().values
        dmso_mad = dmso_vals[f_cols].apply(stats.mad, axis=0).values
        assert len(dmso_med) == group[f_cols].shape[1]
        # subtract each row of the group by that group's DMSO values
        group[f_cols] = group[f_cols].sub(dmso_med)
        # divide by the MAD of the negative control
        group[f_cols] = group[f_cols].apply(lambda x: (x/dmso_mad)*1.4826, axis=1)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, group])
    # check we have not lost any rows
    assert data.shape == df_out.shape
    return df_out


def normalise(data, plate_id, parallel=False, **kwargs):
    """
    Normalise values against negative controls values per plate.

    Parameters
    ----------
    data : pandas DataFrame
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
    if parallel:
        return p_normalise(data, plate_id, **kwargs)
    else:
        return s_normalise(data, plate_id, **kwargs)



def s_normalise(data, plate_id, compound="Metadata_compound",
                neg_compound="DMSO", method="subtract", **kwargs):
    """
    Normalise values against negative controls values per plate.

    Parameters
    ----------
    data : pandas DataFrame
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
        raise ValueError("Invalid method, options: {}".format(valid_methods))
    # check there are some negative controls on each plate
    _check_control(data, plate_id, compound, neg_compound)
    # identify feature columns
    f_cols = utils.get_featuredata(data, **kwargs)
    # dataframe for output
    df_out = pd.DataFrame()
    # group by plate
    grouped = data.groupby(plate_id, as_index=False)
    # calculate the average negative control values for each plate
    for _, group in grouped:
        dmso_med_ = group[group[compound] == neg_compound]
        dmso_med = dmso_med_[f_cols].median()
        if method == "subtract":
            group[f_cols] = group[f_cols].sub(dmso_med)
        if method == "divide":
            group[f_cols] = group[f_cols].divide(dmso_med)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, group])
    # check we have not lost any rows
    assert data.shape == df_out.shape
    return df_out


def _norm_group(group, neg_compound, compound, f_cols):
    """simple normalisation funcion for use with p_normalise"""
    dmso_med = group[group[compound] == neg_compound][f_cols].median()
    copy = group.copy()
    copy[f_cols] = copy[f_cols].sub(dmso_med)
    return copy


def _apply_parallel(grouped_df, func, neg_compound, compound, f_cols, n_jobs):
    """internal parallel gubbins for p_normalise"""
    n_cpu = multiprocessing.cpu_count()
    output = Parallel(n_jobs=n_jobs)(delayed(func)(
        group, neg_compound, compound, f_cols) for _, group in grouped_df)
    return pd.concat(output)


def p_normalise(data, plate_id, compound="Metadata_compound",
                neg_compound="DMSO", n_jobs=-1, **kwargs):
    """
    parallelised version of normalise, currently only works with subtraction
    normalisation.
    """
    _check_control(data, plate_id, compound, neg_compound)
    if n_jobs == -1:
        # use all available cpu cores
        n_jobs = multiprocessing.cpu_count()
    if "method" in kwargs:
        msg = "only implemented subtraction based normalisation in parallel"
        raise NotImplementedError(msg)
    f_cols = utils.get_featuredata(data, **kwargs)
    grouped = data.groupby(plate_id, as_index=False)
    return _apply_parallel(grouped_df=grouped, func=_norm_group,
                           neg_compound=neg_compound, compound=compound,
                           f_cols=f_cols, n_jobs=n_jobs)

