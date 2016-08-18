from morar import stats
import pandas as pd
import numpy as np

def test_mad_wikipedia_example():
    data_in = [1,1,2,2,4,6,9]
    correct = 1.0
    assert stats.mad(data_in) == correct

def test_mad_all_ones():
    data_in = [1,1,1,1,1,1]
    correct = 0.0
    assert stats.mad(data_in) == correct

def test_mad_skewed():
    data_in = [1,1,1,1,1,1,999]
    correct = 0.0
    assert stats.mad(data_in) == correct

def test_mad_negatives():
    data_in = [-1, -1, -1, -1, -1]
    correct = 0.0
    assert stats.mad(data_in) == correct


def test_mad_dataframe_row():
    x = [1,2,3]
    y = [4,10, 5]
    z = [0.4, 8, 0.6]
    df = pd.DataFrame(zip(x, y, z), columns=["x", "y", "z"])
    assert isinstance(stats.mad(df.ix[0]), float)


def test_mad_dataframe_apply():
    x = [1,2,3]
    y = [1,10, 5]
    z = [1, 8, 0.6]
    df = pd.DataFrame(zip(x, y, z), columns=["x", "y", "z"])
    out = list(df.apply(lambda x: stats.mad(x), axis=1).values)
    assert isinstance(out, list)
    assert isinstance(out[0], float)
    print out
    assert list(out)[0] == 0.0


def test_glog():
    x = [1,2,3,4,5,6,7,100]
    out = stats.glog(x)
    assert isinstance(out, np.ndarray)


def test_glog_1():
    out = stats.glog(1)
    assert abs(out - 0.08174569) < 1e-4


def test_glog_skew():
    x = np.random.randn(10000)
    y = np.append(x, 100)
    out = stats.glog(y)
    assert max(out) < 10


def test_glog_dataframe():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    z = np.random.randn(1000)
    df = pd.DataFrame(zip(x, y, z))
    glog_df = df.applymap(lambda x: stats.glog(x))
    assert isinstance(glog_df, pd.DataFrame)
