import numpy as np

def mad(data):
    """ median absolute deviation """
    arr = np.ma.array(data).compressed().astype(np.float)
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def glog(x, c=1):
    """
    generalized log transformation
    --------------------------------------------------------------------------
    c: constant, normally leave as default
    """
    x = np.array(x)
    return np.log10((x + (x**2 + c**2) ** 0.5) / 2)


def z_score(x):
    """
    z_score values, mean=0, standard deviation=1
    --------------------------------------------------------------------------
    x: numeric
    """
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


def scale_features(df):
    """
    scale and centre features
    --------------------------------------------------------------------------
    df: dataframe
    """
    return df.apply(z_score)