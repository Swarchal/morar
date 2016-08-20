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
    return data.apply(z_score)


def find_correlation(df, thresh=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    @param df pandas DataFrame
    @param thresh correlation threshold, will remove one of pairs of features
        with a correlation greater than this value
    @return list of column names to be removed
    """
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > thresh].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat
