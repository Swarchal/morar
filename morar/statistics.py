import numpy as np
from scipy import stats


def median(x):
    return np.median(x)


def mad(x):
    """ median absolute deviation"""
    return median(np.absolute(x - median(x)))


def hampel(x, sigma = 4):
    """
    - Hampel's robust outlier test.
    - 'sigma' is the number of median absolute deviations away from the
      sample median to define an outlier
    - Returns a list:
        -  1  : positive outlier
        - -1  : negative outlier
        -  0  : normal value
    """
    med_x = median(x)
    mad_x = mad(x)
    h_pos = med_x + sigma * mad_x
    h_neg = med_x - sigma * mad_x
    out = []
    for i in x:
	if i > h_pos:
	    out.append(1)
	elif i < h_neg:
	    out.append(-1)
	else:
	    out.append(0)
    assert len(out) == len(x)
    return out


def trim_mean(x, prop = 0.1):
    """ Trimmed mean """
    return stats.trim_mean(x, prop)


def winsorise(x, prop = 0.1):
    """ Winsorised numbers"""
    return stats.mstats.winsorize(x, limits = prop)


def winsor_mean(x, prop = 0.1):
    """ Winsorised mean"""
    return np.mean(winsorise(x, prop))


def iqr(x):
    out = np.percentile(x, 75) - np.percentile(x, 25)
    return out


def o_iqr(x):
    ' >1.5*IQR'
    out = []
    lim = np.median(x) + 1.5 * iqr(x)
    for i in x:
        if i > lim:
            out.append(1)
        else:
            out.append(0)
    assert len(out) == len(x)
    return out


def u_iqr(x):
    ' <1.5*IQR'
    out = []
    lim = np.median(x) - 1.5 * iqr(x)
    for i in x:
        if i < lim:
            out.append(1)
        else:
            out.append(0)
    assert len(out) == len(x)
    return out


if __name__ == "__main__":
    x = [1,2,3,4,2,4,2,2,4,2,1,3,1999]
    print "trim mean:   %f" % trim_mean(x, prop = 0.5)
    print winsor_mean(x, prop = 0.1)
    print winsorise(x, prop = 0.1)
