import numpy as np
import pandas as pd
import pytest

from morar import normalise


def test_check_control():
    """normalise._check_control"""
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    normalise._check_control(
        data=df,
        compound="Metadata_compound",
        plate_id="Metadata_plate",
        neg_compound="DMSO",
    )


def test_check_control_within_function():
    # dataframe with missing controls in one plate
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 10) + (["drug"] * 8 + ["DMSO"] * 2) * 4
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    missing_control_df = pd.DataFrame(
        list(zip(x, y, z, plate, compound)), columns=colnames
    )
    with pytest.raises(RuntimeError):
        normalise.normalise(missing_control_df, plate_id="Metadata_plate")


def test_normalise_errors_invalid_method():
    # create test DataFrame
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    with pytest.raises(ValueError):
        normalise.normalise(df, plate_id="Metadata_plate", method="invalid")


def test_normalise_returns_dataframe_subtract():
    # create test DataFrame
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.normalise(df, plate_id="Metadata_plate", method="subtract")
    assert isinstance(out, pd.DataFrame)


def test_normalise_returns_dataframe_divide():
    # create test DataFrame
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.normalise(df, plate_id="Metadata_plate", method="divide")
    assert isinstance(out, pd.DataFrame)


def test_normalise_returns_correct_size():
    # create test DataFrame
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert out.shape[0] == df.shape[0]


def test_normalise_divide_returns_correct_values():
    # simple dataframe to check actual values
    x = [4, 4, 4, 2, 2]
    compound = ["drug"] * 3 + ["DMSO"] * 2
    plate = ["plate_1"] * 5
    colnames = ["f1", "Metadata_compound", "Metadata_plate"]
    simple_df = pd.DataFrame(list(zip(x, compound, plate)), columns=colnames)
    out = normalise.normalise(simple_df, plate_id="Metadata_plate", method="divide")
    assert isinstance(out, pd.DataFrame)
    assert out["f1"].tolist() == [2, 2, 2, 1, 1]


def test_normalise_subtract_returns_correct_values():
    # simple dataframe to check actual values
    x = [4, 4, 4, 2, 2]
    compound = ["drug"] * 3 + ["DMSO"] * 2
    plate = ["plate_1"] * 5
    colnames = ["f1", "Metadata_compound", "Metadata_plate"]
    simple_df = pd.DataFrame(list(zip(x, compound, plate)), columns=colnames)
    out = normalise.normalise(simple_df, plate_id="Metadata_plate", method="subtract")
    assert isinstance(out, pd.DataFrame)
    assert out["f1"].tolist() == [2, 2, 2, 0, 0]


def test_normalise_non_default_cols():
    # dataframe with weird columns names
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "meta_plate", "meta_cmpd"]
    non_default_df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.normalise(
        non_default_df,
        compound="meta_cmpd",
        plate_id="meta_plate",
        metadata_string="meta",
    )
    assert isinstance(out, pd.DataFrame)


def test_normalise_extra_metadata_cols():
    # dataframe with weird columns names
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    extra_metadata = ["A", "B"] * 25
    colnames = ["A", "B", "C", "meta_plate", "meta_cmpd", "metadata_extra"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound, extra_metadata)))
    df.columns = colnames
    out = normalise.normalise(
        df, metadata_string="meta", compound="meta_cmpd", plate_id="meta_plate"
    )
    assert df.shape == out.shape
    assert df.columns.tolist() == out.columns.tolist()


def test_robust_normalise_extra_metadata_cols():
    # dataframe with weird columns names
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    extra_metadata = ["A", "B"] * 25
    colnames = ["A", "B", "C", "meta_plate", "meta_cmpd", "metadata_extra"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound, extra_metadata)))
    df.columns = colnames
    out = normalise.robust_normalise(
        df, metadata_string="meta", compound="meta_cmpd", plate_id="meta_plate"
    )
    assert df.shape == out.shape
    assert df.columns.tolist() == out.columns.tolist()


def test_robust_normalise_returns_dataframe():
    # create test DataFrame
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.robust_normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)


def test_robust_normalise_returns_correct_size():
    # create test DataFrame
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.robust_normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)
    assert df.shape == out.shape


def test_robust_normalise_non_default_cols():
    # dataframe with weird columns names
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "meta_plate", "meta_cmpd"]
    non_default_df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    out = normalise.robust_normalise(
        non_default_df,
        metadata_string="meta",
        compound="meta_cmpd",
        plate_id="meta_plate",
    )
    assert isinstance(out, pd.DataFrame)
    assert out.shape == non_default_df.shape


def test_parallel_normalise():
    x = np.random.randn(50).tolist()
    y = np.random.randn(50).tolist()
    z = np.random.randn(50).tolist()
    plate = (
        ["plate1"] * 10
        + ["plate2"] * 10
        + ["plate3"] * 10
        + ["plate4"] * 10
        + ["plate5"] * 10
    )
    compound = (["drug"] * 8 + ["DMSO"] * 2) * 5
    colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
    df = pd.DataFrame(list(zip(x, y, z, plate, compound)), columns=colnames)
    standard_output = normalise.normalise(df, plate_id="Metadata_plate")
    parallel_output = normalise.normalise(df, plate_id="Metadata_plate", parallel=True)
    assert standard_output.equals(parallel_output)


def test_whitening():
    x = np.random.normal(loc=1, scale=1, size=100)
    df = pd.DataFrame(
        {
            "x": x,
            "y": x + np.random.normal(loc=0, scale=0.1, size=100),
            "z": x + np.random.normal(loc=0, scale=0.5, size=100),
            "Metadata_stuff": ["test"] * 100,
        }
    )
    df_whitened = normalise.whiten(df)
    assert "Metadata_stuff" in df_whitened.columns
    # check columns have less correlation
    df_corr = df[["x", "y", "z"]].corr()
    df_w_corr = df_whitened[["x", "y", "z"]].corr()
    assert all(df_w_corr <= df_corr)
