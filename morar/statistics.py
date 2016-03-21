import numpy as np

def median(x):
    return np.median(x)

def mad(x):
    return median(np.absolute(x - median(x)))

def hampel(x, sigma = 4):
    """
    - Hampel's robust outlier test.
    - 'sigma' is the number of median absolute deviations away from the
      sample median to define an outlier
    - Returns a list:
        - 1  : positive outlier
        - -1 : negative outlier
        - 0  : normal value
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
