from morar import utils
import numpy as np
import pandas as pd


def mad(data):
    """
    median absolute deviation
    @param data scalar/list/array of type numeric or integer
    @return scalar or numpy array
    """
    arr = np.ma.array(data).compressed().astype(np.float)
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def glog(x, c=1):
    """
    generalized log transformation
    @param x scalar/list/array of type numeric or integer
    @param c constant, normally leave as default
    @return scalar or numpy array
    """
    x = np.array(x)
    return np.log10((x + (x**2 + c**2) ** 0.5) / 2)


def z_score(x):
    """
    z_score values, mean=0, standard deviation=1
    @param numeric
    @return scalar or numpy array
    """
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


def scale_features(data):
    """
    scale and centre features with a z-score
    @param df pandas DataFrame
    @return pandas DataFrame
    """
    feature_data = data[utils.get_featuredata(data)]
    return feature_data.apply(z_score)


def hampel(x, sigma=6):
    """
    Hampel filter without window

    @param x array or list of numerical values
    @param sigma number of median absolute deviations away from the sample
                 median to define an outlier

    @details (1) = positive outlier,
            (-1) = negative outlier,
             (0) = nominal value
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
