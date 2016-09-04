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


def find_correlation(df, threshold=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    @param df pandas DataFrame
    @param threshold correlation threshold, will remove one of pairs of features
        with a correlation greater than this value
    @return list of column names to be removed
    """
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


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


def find_low_var(df, threshold=1e-5):
    """
    Return column names of feature columns with zero or very low variance

    @param df pandas DataFrame
    @oaram threshold low variance threshold
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(df, "is not a pandas DataFrame")
    var = df[utils.get_featuredata(df)].var(axis=0)
    return var[var < threshold].index.tolist()
