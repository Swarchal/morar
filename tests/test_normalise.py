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


def test_get_featuredata_middle_prefix():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "metadata_A", "Metadata_A"]
    test_df = pd.DataFrame(zip(x, y, z, a), columns=columns)
    cols = normalise.get_featuredata(test_df, metadata_prefix="metadata")
    assert cols == ["colA", "colB", "Metadata_A"]



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



@raises(RuntimeError)
def test_check_control():
    normalise.check_control(missing_control_df, plate_id="Metadata_plate")
    # assert for RuntimeError


def test_normalise_returns_dataframe():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert isinstance(out, pd.DataFrame)


def test_normalise_returns_correct_size():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    assert out.shape[1] == df.shape[1]


def test_normalise_returns_correct_values():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    pass


def test_normalise_method_subtract():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    pass


def test_normalise_method_divide():
    out = normalise.normalise(df, plate_id="Metadata_plate")
    pass


def test_normalise_non_default_cols():
    out = normalise.normalise(non_default_df, metadata_prefix="meta",
                              compound="meta_cmpd", plate_id="meta_plate")
    assert isinstance(out, pd.DataFrame)
    assert out.shape[1] == non_default_df.shape[1]



def test_robust_normalise_returns_dataframe():
    pass


def test_robust_normalise_returns_correct_size():
    pass


def test_robust_normalise_non_default_cols():
    pass