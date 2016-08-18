from morar import stats
import pandas as pd


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
    pass