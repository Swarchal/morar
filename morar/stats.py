from morar import utils
import numpy as np
import pandas as pd


def mad(data):
    """
    median absolute deviation

    Parameters
    ----------
    data : array-like
        numbers with which to calculate

    Returns
    -------
    mad : numpy array
        median absolute deviation
    """
    arr = np.ma.array(data).compressed().astype(np.float)
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def glog(x, c=1.0):
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


def z_score(x):
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


def scale_features(data):
    """
    scale and centre features with a z-score

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame

    Returns
    -------
    scaled : pandas DataFrame
        dataframe of scaled feature values
    """
    feature_data = data[utils.get_featuredata(data)]
    return feature_data.apply(z_score)


def hampel(x, sigma=6):
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

    Returns
    --------
    outliers : numpy array
        array of same size as the input, outliers indicated as -1 or 1, nominal
        values as 0.

    """
    x = np.array(x).astype(np.float)
    med_x = np.median(x)
    mad_x = mad(x)
    h_pos = med_x + sigma * mad_x
    h_neg = med_x - sigma * mad_x
    out = np.zeros(len(x))
    for i, val in enumerate(x):
        if val > h_pos:
            out[i] = 1
        elif val < h_neg:
            out[i] = -1
    return out
