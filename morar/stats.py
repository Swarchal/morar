"""
statistical functions
"""

import numpy as np
import pandas as pd

from morar import utils


def mad(data: np.ndarray, axis: int = 0) -> float:
    """
    median absolute deviation

    Parameters
    ----------
    data : array-like
        numbers with which to calculate
    axis : int (default=0)

    Returns
    -------
    mad : numpy array
        median absolute deviation
    """
    arr = np.ma.array(data).compressed().astype(float)
    med = np.median(arr, axis=axis)
    return np.median(np.abs(arr - med))


def glog(x: np.ndarray, c: float = 1.0):
    """
    generalized log transformation

    Parameters
    ----------
    x : scalar or array-like
        numbers with which to calculate
    c float (default=0.1)
        bias (normally leave as default)

    Returns
    -------
    x : scalar or numpy array
        transformed value(s)
    """
    x = np.array(x)
    return np.log10((x + (x**2 + c**2) ** 0.5) / 2)


def z_score(x: np.ndarray):
    """
    z_score values, mean=0, standard deviation=1

    Parameters
    ----------
    x : numeric, array-like
        values to z-score

    Returns
    -------
    scaled: array-like
        z-scored values
    """
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


def scale_features(
    data: pd.DataFrame, metadata_string: str = "Metadata_", prefix: bool = True
):
    """
    scale and centre features with a z-score

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame
    **kwargs : additional arguments to utils.get_featuredata/get_metadata

    Returns
    -------
    scaled : pandas DataFrame
        dataframe of same dimensions as df, with scaled feature values
    """
    data_columns = data.columns.tolist()
    feature_data = data[utils.get_featuredata(data, metadata_string, prefix)]
    metadata = data[utils.get_metadata(data, metadata_string, prefix)]
    scaled_featuredata = feature_data.apply(z_score)
    scaled_both = pd.concat([scaled_featuredata, metadata], axis=1)
    # return columns to original order
    scaled_both = scaled_both[data_columns]
    return scaled_both


def hampel(x: np.ndarray, sigma: float | int = 6, axis: int = 0):
    """
    Hampel filter without window
    (1) = positive outlier, (-1) = negative outlier, (0) = nominal value

    Parameters
    -----------
    x : array-like
        values values with which to calculate
    sigma : int (default=6)
        number of median absolute deviations away from the sample median to
        define an outlier
    axis : int (default=0)
        axis to apply filter to if x is not one dimensional.

    Returns
    --------
    outliers : numpy array
        array of same size as the input, outliers indicated as -1 or 1, nominal
        values as 0.
    """
    x = np.array(x)
    median_x = np.median(x, axis=axis)
    mad_x = mad(x, axis=axis)
    h_pos = median_x + sigma * mad_x
    l_neg = median_x - sigma * mad_x
    out = np.zeros_like(x, dtype=int)
    out[x > h_pos] = 1
    out[x < l_neg] = -1
    return out


def l1_norm(x: np.ndarray, y: np.ndarray) -> float:
    """
    l1 norm between two vectors

    Parameters:
    -----------
    x : array-like
    y : array-like

    Returns:
    ---------
    float, l1_norm
    """
    return np.sum(np.abs(np.asarray(x) - np.asarray(y)))


def cohens_d(pos_control: np.ndarray, neg_control: np.ndarray) -> float:
    """
    Cohen's d measure. The standardised difference between the positive and
    negative control means.

    Parameters:
    ------------
    pos_control : numeric array-like
    neg_control : numeric array-like

    Returns:
    ---------
    float
    """
    pos = np.asarray(pos_control)
    neg = np.asarray(neg_control)
    # check the controls values are as expected
    if neg.mean() > pos.mean():
        msg = "mean of negative control is greater than positive control"
        raise ValueError(msg)
    # if groups sizes are within 10% of each other, then use the simple
    # sigma prime calculation
    if abs(len(pos) - len(neg)) < 0.1 * len(pos):
        jacob_cohen = np.sqrt((pos.var() + neg.var()) / 2)
    else:
        # if the groups are considerably different sizes then have to
        # adjust sigma prime to account for this
        n_pos, n_neg = len(pos), len(neg)
        jacob_cohen = np.sqrt(
            (n_pos * pos.var() + n_neg * neg.var()) / (n_pos + n_neg - 2)
        )
    return (pos.mean() - neg.mean()) / jacob_cohen
