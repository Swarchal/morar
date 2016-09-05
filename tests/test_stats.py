from morar import stats
import pandas as pd
import numpy as np
from nose.tools import raises

np.random.seed(0)

def test_mad_returns_correct_answer():
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
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    assert isinstance(stats.mad(df.ix[0]), float)


def test_mad_dataframe_apply():
    x = [1,2,3]
    y = [1,10, 5]
    z = [1, 8, 0.6]
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = list(df.apply(lambda x: stats.mad(x), axis=1).values)
    assert isinstance(out, list)
    assert isinstance(out[0], float)
    assert list(out)[0] == 0.0


def test_glog():
    x = [1,2,3,4,5,6,7,100]
    out = stats.glog(x)
    assert isinstance(out, np.ndarray)


def test_glog_1():
    out = stats.glog(1)
    assert abs(out - 0.08174569) < 1e-6


def test_glog_skew():
    x = np.random.randn(10000)
    y = np.append(x, 100)
    out = stats.glog(y)
    assert max(out) < 10


def test_glog_dataframe():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, y, z)))
    glog_df = df.applymap(lambda x: stats.glog(x))
    assert isinstance(glog_df, pd.DataFrame)


def test_zscore_returns_same_size():
    x = [1,2,3,4,5]
    out = stats.z_score(x)
    assert len(x) == len(out)


def test_zscore_means_to_zero():
    x = [1,2,3,4,5,6,3,2,4,5,3,2,3,4]
    out = stats.z_score(x)
    assert abs(out.mean() - 0) < 1e-6


def test_zscore_sd_to_1():
    x = [1,2,3,4,5,6,3,2,4,5,3,2,3,4]
    out = stats.z_score(x)
    assert abs(out.std() - 1) < 1e-6


def test_scale_features():
    x = np.random.normal(loc=5, scale=1, size=1000)
    y = np.random.normal(loc=10, scale=5, size=1000)
    df = pd.DataFrame(list(zip(x, y)))
    df.columns = ["x", "y"]
    out = stats.scale_features(df)
    assert abs(out["x"].mean() - 0.0) < 1e-6
    assert abs(np.std(out["x"]) - 1.0) < 1e-6
    assert abs(out["y"].mean() - 0.0) < 1e-6
    assert abs(np.std(out["y"]) - 1.0) < 1e-6


def test_find_correlation():
    x = range(1000)
    noise = np.random.randn(1000)
    y = [a+b for a,b in zip(x, noise)]
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = stats.find_correlation(df)
    assert len(out) == 1
    assert out[0] == ["x"] or ["y"]
    assert out[0] != ["z"]


def test_find_correlation_threshold_works():
    x = range(1000)
    noise = np.random.randn(1000)
    y = [a+b for a,b in zip(x, noise)]
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = stats.find_correlation(df, threshold=1.0)
    assert len(out) == 0


def test_find_correlation_multiple_correlated():
    x = range(1000)
    noise = np.random.randn(1000)
    y = [a+b for a,b in zip(x, noise)]
    xx = [a+b for a,b in zip(x, noise)]
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, xx, y, z)), columns=["x", "xx", "y", "z"])
    out = stats.find_correlation(df)
    assert len(out) == 2
    assert "z" not in out


def test_find_correlation_large_n():
    x = range(100000)
    noise = np.random.randn(100000)
    y = [a+b for a,b in zip(x, noise)]
    z = np.random.randn(100000)
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = stats.find_correlation(df)
    assert len(out) == 1
    assert out[0] == ["x"] or ["y"]
    assert out[0] != ["z"]


def test_hampel():
    x = np.random.random(100)
    x_new = np.append(x, 100)
    out = stats.hampel(x_new)
    ans = np.zeros(100)
    ans = np.append(ans, 1.0)
    assert len(out) == 101
    assert sum(out) == 1.0
    assert all(out == ans)


def test_hampel_negative():
    x = np.random.random(100)
    x_new = np.append(x, -100)
    out = stats.hampel(x_new)
    ans = np.zeros(100)
    ans = np.append(ans, -1.0)
    assert len(out) == 101
    assert sum(out) == -1.0
    assert all(out == ans)


def test_hampel_sigma():
    x = np.random.random(100)
    # append a value slightly higher than the others
    x_new = np.append(x, 3)
    # v. high sigma value
    out = stats.hampel(x_new, sigma=20)
    ans = np.zeros(101)
    assert all(out == ans)


a = np.random.random(100)
b = np.zeros(100)
c = np.random.random(100)
df = pd.DataFrame(list(zip(a, b, c)))
df.columns = ["a", "b", "c"]


def test_find_low_var():
    out = stats.find_low_var(df)
    assert len(out) == 1
    assert out == ["b"]

@raises(ValueError)
def test_find_low_var_errors():
    stats.find_low_var(df["a"].tolist())


def test_fine_low_var_threshold():
    sigma = 0.5
    var = sigma**2
    x = np.random.normal(loc=1, scale=sigma, size=1000)
    y = np.random.normal(loc=1, scale=1.0, size=1000)
    df2 = pd.DataFrame(list(zip(x, y)))
    df2.columns = ["x", "y"]
    out = stats.find_low_var(df2, threshold=var*2)
    assert out == ["x"]


def test_find_low_var_nan():
    # dataset containing NaN values
    x = [np.nan]*10
    y = range(10)
    z = [1]*10
    df_nan = pd.DataFrame(list(zip(x, y, z)))
    df_nan.columns = ["x", "y", "z"]
    out = stats.find_low_var(df_nan)
    print(out)
    assert (out == ["x", "z"]) or (out == ["z", "x"])
