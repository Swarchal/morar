from morar import normalise
import pandas as pd
import numpy as np
from nose.tools import raises


def test_get_featuredata_simple():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    columns = ["colA", "colB", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z), columns=columns)
    cols = normalise.get_featuredata(test_df)
    assert cols == ["colA", "colB"]


def test_get_featuredata_middle_prefix():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "something_Metadata", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z, a), columns=columns)
    cols = normalise.get_featuredata(test_df)
    assert cols == ["colA", "colB", "something_Metadata"]


def test_get_featuredata_different_case():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "metadata_A", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z, a), columns=columns)
    cols = normalise.get_featuredata(test_df, metadata_prefix="metadata")
    assert cols == ["colA", "colB", "Metadata_A"]


def test_get_metadata_simple():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    columns = ["colA", "colB", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z), columns=columns)
    cols = normalise.get_metadata(test_df)
    assert cols == ["Metadata_A"]


def test_get_metadata_middle_prefix():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "something_Metadata", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z, a), columns=columns)
    cols = normalise.get_metadata(test_df)
    assert cols == ["Metadata_A"]


def test_get_metadata_different_case():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "metadata_A", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z, a), columns=columns)
    cols = normalise.get_metadata(test_df, metadata_prefix="metadata")
    assert cols == ["metadata_A"]


# create test DataFrame
x = np.random.randn(50).tolist()
y = np.random.randn(50).tolist()
z = np.random.randn(50).tolist()
plate = ["plate1"]*10 + ["plate2"]*10 + ["plate3"]*10 + ["plate4"]*10 + ["plate5"]*10
compound = (["drug"]*8 + ["DMSO"]*2)*5
colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
df = pd.DataFrame(zip(x, y, z, plate, compound), columns=colnames)


# dataframe with weird columns names
x = np.random.randn(50).tolist()
y = np.random.randn(50).tolist()
z = np.random.randn(50).tolist()
plate = ["plate1"]*10 + ["plate2"]*10 + ["plate3"]*10 + ["plate4"]*10 + ["plate5"]*10
compound = (["drug"]*8 + ["DMSO"]*2)*5
colnames = ["A", "B", "C", "meta_plate", "meta_cmpd"]
non_default_df = pd.DataFrame(zip(x, y, z, plate, compound), columns=colnames)


# dataframe with missing controls in one plate
x = np.random.randn(50).tolist()
y = np.random.randn(50).tolist()
z = np.random.randn(50).tolist()
plate = ["plate1"]*10 + ["plate2"]*10 + ["plate3"]*10 + ["plate4"]*10 + ["plate5"]*10
compound = (["drug"]*10) + (["drug"]*8 + ["DMSO"]*2)*4
colnames = ["A", "B", "C", "Metadata_plate", "Metadata_compound"]
missing_control_df = pd.DataFrame(zip(x, y, z, plate, compound), columns=colnames)



@raises(ValueError)
def test_check_control():
    normalise.check_control(missing_control_df, plate_id="Metadata_plate")
    # assert for ValueError


@raises(ValueError)
def test_check_control_within_function():
    normalise.normalise(missing_control_df, plate_id="Metadata_plate")
    # assert for ValueError


def test_normalise_returns_dataframe_subtract():
    out = normalise.normalise(df, plate_id="Metadata_plate", method="subtract")
    assert isinstance(out, pd.DataFrame)


def test_normalise_returns_dataframe_divide():
    out = normalise.normalise(df, plate_id="Metadata_plate", method="divide")
    assert isinstance(out, pd.DataFrame)


def test_normalise_returns_correct_size():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert out.shape[0] == df.shape[0]


def test_normalise_returns_correct_values():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == df.shape[0]
    # TODO assert correct values


def test_normalise_method_subtract():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)
    # TODO assert correct values


def test_normalise_method_divide():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)
    # TODO assert correct values


def test_normalise_non_default_cols():
    out = normalise.normalise(non_default_df, metadata_prefix="meta",
                              compound="meta_cmpd", plate_id="meta_plate")
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == non_default_df.shape[0]


def test_robust_normalise_returns_dataframe():
    out = normalise.robust_normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)


def test_robust_normalise_returns_correct_size():
    out = normalise.robust_normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)
    assert df.shape[0] == out.shape[0]
    # TODO check for correct number of columns


def test_robust_normalise_non_default_cols():
    out = normalise.robust_normalise(non_default_df, metadata_prefix="meta",
                                     compound="meta_cmpd", plate_id="meta_plate")
    assert isinstance(out, pd.DataFrame)
    assert out.shape[0] == non_default_df.shape[0]
