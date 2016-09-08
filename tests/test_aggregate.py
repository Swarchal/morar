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

