from morar import outliers
import pandas as pd
import numpy as np
from nose.tools import raises

np.random.seed(0)

@raises(ValueError)
def test_get_outlier_index_errors_wrong_method():
    # example dataset for get_outlier_index
    x = np.random.random(100).tolist()
    y = np.random.random(100).tolist()
    # add outlying values to last row
    x.append(100)
    y.append(500)
    df3 = pd.DataFrame(list(zip(x, y)))
    df3.columns = ["x", "y"]
    outliers.get_outlier_index(df3, method="wrong")


@raises(ValueError)
def test_get_outlier_index_errors_non_dataframe():
    # example dataset for get_outlier_index
    x = np.random.random(100).tolist()
    y = np.random.random(100).tolist()
    # add outlying values to last row
    x.append(100)
    y.append(500)
    df3 = pd.DataFrame(list(zip(x, y)))
    df3.columns = ["x", "y"]
    outliers.get_outlier_index(df3["x"].tolist())


def test_get_outlier_index_adjust():
    # example dataset for get_outlier_index
    x = np.random.random(100).tolist()
    y = np.random.random(100).tolist()
    # add outlying values to last row
    x.append(100)
    y.append(500)
    df3 = pd.DataFrame(list(zip(x, y)))
    df3.columns = ["x", "y"]
    out = outliers.get_outlier_index(df3, adjust=True)
    assert len(out) == 1
    assert out == [100] # 0-based indexing


def test_get_outlier_index_adjust2():
    # example dataset for get_outlier_index
    x = np.random.random(100).tolist()
    y = np.random.random(100).tolist()
    # add outlying values to last row
    x.append(1)
    y.append(3)
    df3 = pd.DataFrame(list(zip(x, y)))
    df3.columns = ["x", "y"]
    out = outliers.get_outlier_index(df3, adjust=False)
    assert len(out) == 1
    assert out == [100] # 0-based indexing


def test_get_outlier_index_values():
    # example dataset for get_outlier_index
    x = np.random.random(100).tolist()
    y = np.random.random(100).tolist()
    # add outlying values to last row
    x.append(100)
    y.append(500)
    df3 = pd.DataFrame(list(zip(x, y)))
    df3.columns = ["x", "y"]
    out = outliers.get_outlier_index(df3, method="values")
    assert len(out) == 1
    assert out == [100] # 0-based indexing


def test_get_outlier_index_ImageQuality():
    # create dataframe with important ImageQuality measurements
    # FocusScore, PowerLogLogSlope
    # have a row with atypical values
    x = np.random.random(1000)
    x2 = np.random.random(1000)
    x3 = np.random.random(1000)
    x4 = np.random.random(1002)
    x5 = np.random.random(1002)
    x6 = np.random.random(1002)
    # introduce two bad images with low focus and ppls values
    x = np.append(x, [-3, -5])
    x2 = np.append(x2, [-2, -4])
    x3 = np.append(x3, [-3, -3])
    df = pd.DataFrame(list(zip(x, x2, x3, x4, x5, x6)))
    df.columns = [
        "ImageQuality_PowerLogLogSlope_ch1",
        "ImageQuality_PowerLogLogSlope_ch2",
        "ImageQuality_FocusScore_ch1",
        "ImageQuality_FocusScore_ch2",
        "vals1", "vals2"
    ]
    out = outliers.get_outlier_index(df, method="ImageQuality", sigma=6)
    assert len(out) == 2
    assert out == [1000, 1001]
