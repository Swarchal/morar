from morar import aggregate
import numpy as np
import pandas as pd
from nose.tools import raises

np.random.seed(0)

@raises(ValueError)
def test_aggregate_errors_wrong_column():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = list(range(1,21))*50
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber"]
    aggregate.aggregate(df, on="incorrect")


@raises(ValueError)
def test_aggregate_errors_non_dataframe():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = list(range(1,21))*50
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber"]
    aggregate.aggregate(x, on="Metadata_imagenumber")

@raises(ValueError)
def test_aggregate_errors_invalid_method():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = list(range(1,21))*50
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber"]
    aggregate.aggregate(df, on="Metdata_imagenumber", method="invalid")


def test_aggregate_correct_shape():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = list(range(1,21))*50
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber"]
    out = aggregate.aggregate(df, on="Metadata_imagenumber")
    assert out.shape[0] == 20
    assert out.shape[1] == df.shape[1]
    assert out.columns.tolist() == df.columns.tolist()


def test_aggregate_on_string():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = ["a", "b", "c", "d", "e"]*200
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber"]
    out = aggregate.aggregate(df, on="Metadata_imagenumber")
    assert out.shape[0] == 5
    assert out.shape[1] == df.shape[1]
    assert out.columns.tolist() == df.columns.tolist()


def test_aggregate_on_multiple_columns():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = ["a", "b", "c", "d", "e"]*200
    metadata_group = ["X", "Y"]*500
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber, metadata_group)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber", "Metadata_group"]
    out = aggregate.aggregate(df, on=["Metadata_imagenumber", "Metadata_group"])
    assert out.shape[0] == 10
    assert out.shape[1] == df.shape[1]
    assert out.columns.tolist() == df.columns.tolist()


def test_aggregate_methods():
    x = [1,2,10, 1,5,10]
    y = [1,2,1,  2,5,1 ]
    names = ["a", "a", "a", "b", "b", "b"]
    df = pd.DataFrame(list(zip(x, y, names)))
    df.columns = ["x", "y", "group"]
    out_median = aggregate.aggregate(df, on="group", method="median")
    out_mean = aggregate.aggregate(df, on="group", method="mean")
    assert out_median.columns.tolist() == out_mean.columns.tolist()
    assert out_median.shape == out_mean.shape
    assert out_median["x"].values.tolist() == [2, 5]
    assert out_median["y"].values.tolist() == [1, 2]
    # floating point numbers, so will have to assert for small differences
    mean_x = out_mean["x"].values.tolist()
    mean_y = out_mean["y"].values.tolist()
    assert abs(mean_x[0] - 4.333333) < 1e-5
    assert abs(mean_x[1] - 5.333333) < 1e-5
    assert abs(mean_y[0] - 1.333333) < 1e-5
    assert abs(mean_y[1] - 2.666666) < 1e-5


def test_aggregate_multiple_metadata():
    x = np.random.random(1000)
    y = np.random.random(1000)
    z = np.random.random(1000)
    metadata_imagenumber = ["a", "b", "c", "d", "e"]*200
    metadata_group = ["X", "Y"]*500
    metadata_other = np.random.random(1000)
    df = pd.DataFrame(list(zip(x, y, z, metadata_imagenumber, metadata_group, metadata_other)))
    df.columns = ["x", "y", "z", "Metadata_imagenumber", "Metadata_group", "Metadata_other"]
    out = aggregate.aggregate(df, on=["Metadata_imagenumber", "Metadata_group"])
    print(out.columns.tolist())
    print(df.columns.tolist())
    assert out.columns.tolist() == df.columns.tolist()
    assert out.shape[0] == 10

