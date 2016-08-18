import numpy as np

def mad(data):
    """ median absolute deviation """
    arr = np.ma.array(data).compressed().astype(np.float)
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def glog(x, c=1):
    """
    generalized log transformation
    to be used as: pd.DataFrame.apply(glog)
    """
    return np.log10((x + (x**2 + c**2) ** 0.5) / 2)