from morar import utils
import pandas as pd
import numpy as np
from nose.tools import raises

def test_get_featuredata_simple():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    columns = ["colA", "colB", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z)), columns=columns)
    cols = utils.get_featuredata(test_df)
    assert cols == ["colA", "colB"]


def test_get_featuredata_middle_prefix():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "something_Metadata", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z, a)), columns=columns)
    cols = utils.get_featuredata(test_df, prefix=True)
    assert cols == ["colA", "colB", "something_Metadata"]


def test_get_feature_data_prefix_options():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "something_Metadata", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z, a)), columns=columns)
    out = utils.get_featuredata(test_df, prefix=False)
    ans = ["colA", "colB"]
    assert out == ans

def test_get_featuredata_different_case():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "metadata_A", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z, a)), columns=columns)
    cols = utils.get_featuredata(test_df, metadata_string="metadata")
    assert cols == ["colA", "colB", "Metadata_A"]


def test_get_metadata_simple():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    columns = ["colA", "colB", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z)), columns=columns)
    cols = utils.get_metadata(test_df)
    assert cols == ["Metadata_A"]


def test_get_metadata_middle_prefix():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "something_Metadata", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z, a)), columns=columns)
    cols = utils.get_metadata(test_df)
    assert cols == ["Metadata_A"]


def test_get_metadata_different_case():
    x = [1,2,3,4]
    y = [4,3,2,1]
    z = [1,2,3,4]
    a = [4,3,5,1]
    columns = ["colA", "colB", "metadata_A", "Metadata_A"]
    test_df = pd.DataFrame(list(zip(x, y, z, a)), columns=columns)
    cols = utils.get_metadata(test_df, metadata_string="metadata")
    assert cols == ["metadata_A"]


def test_is_all_nan():
    x = [np.nan]*10
    y = list(range(10))
    z = list(range(9)) + [np.nan]
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["x", "y", "z"]
    out = utils.is_all_nan(df)
    print(out)
    assert out == ["x"]



# create simple dataframe with ImageQuality columns
x = [1, 2, 3]
y = [2, 4, 1]
z = [2, 5, 1]

df = pd.DataFrame(list(zip(x, y, z)))
df.columns = ["vals", "ImageQuality_test", "other"]


def test_get_image_quality():
    out = utils.get_image_quality(df)
    print(out)
    assert out == ["ImageQuality_test"]


# column has ImageQuality in middle of string
df2 = pd.DataFrame(list(zip(x, y, z)))
df2.columns = ["vals", "ImageQuality_test", "Cells_ImageQuality"]


def test_get_image_quality_not_beginning():
    out = utils.get_image_quality(df2)
    assert out == ["ImageQuality_test", "Cells_ImageQuality"]


test_list = df["ImageQuality_test"].tolist()

@raises(ValueError)
def test_get_image_quality_fails_non_dataframe():
    utils.get_image_quality(test_list)


@raises(ValueError)
def test_get_image_quality_no_im_qc_cols():
    x = [1,2,3,4]
    y = [2,3,4,5]
    df = pd.DataFrame(list(zip(x, y)))
    df.columns = ["x", "y"]
    utils.get_image_quality(df)


def test_impute():
    x = [1, 2, 3, np.nan]
    y = [2, 4, 9, 10]
    dataframe = pd.DataFrame(list(zip(x, y)))
    dataframe.columns = ["x", "y"]
    out = utils.impute(dataframe)
    assert (out["x"].values.tolist() == [1, 2, 3, 2])
    assert out.shape == dataframe.shape
    assert out.columns.tolist() == dataframe.columns.tolist()


def test_impute_mean():
    x = [1, 10, np.nan]
    y = [1,2,3]
    dataframe = pd.DataFrame(list(zip(x, y)))
    dataframe.columns = ["x", "y"]
    out = utils.impute(dataframe, method="mean")
    assert out["x"].values.tolist() == [1, 10, 5.5]


def test_impute_with_metadata():
    x = [1, 2, 3, 4, np.nan]
    y = [1, 2, 3, 4, 5]
    metadata_something = ["a", "b", "c", "d", "e"]
    dataframe = pd.DataFrame(list(zip(x, y, metadata_something)))
    dataframe.columns = ["x", "y", "Metadata_x"]
    out = utils.impute(dataframe)
    assert out.columns.tolist() == dataframe.columns.tolist()
    assert out.shape == dataframe.shape


@raises(ValueError)
def test_drop_missing_bad_threshold():
    x = [1, 2, 3, np.nan]
    y = [1, 2, 3, 4]
    dataframe = pd.DataFrame(list(zip(x, y)))
    dataframe.columns = ["x", "y"]
    utils.drop_missing(dataframe, threshold=10)


def test_drop_missing_correct():
    x = [np.nan]*10
    xx = [np.nan]*10
    y = list(range(10))
    z = list(range(10))
    dataframe = pd.DataFrame(list(zip(x, xx, y, z)))
    dataframe.columns = ["x", "xx", "y", "z"]
    out = utils.drop_missing(dataframe)
    assert out.shape[0] == dataframe.shape[0]
    assert out.columns.tolist() == ["y", "z"]


def test_drop_missing_corrrect_rows():
    x = [np.nan]*50
    y = np.random.random(50)
    z = list(range(49)) + [np.nan]
    dataframe = pd.DataFrame(list(zip(x, y, z)))
    dataframe.columns = ["x", "y", "z"]
    out = utils.drop_missing(dataframe)
    assert out.columns.tolist() == ["y", "z"]
    assert out.shape[0] == 49


def test_drop_missing_threshold():
    x = list(range(50)) + [np.nan]*50
    y = [np.nan]*100
    z = np.random.random(100)
    dataframe = pd.DataFrame(list(zip(x, y, z)))
    dataframe.columns = ["x", "y", "z"]
    out = utils.drop_missing(dataframe, threshold=0.2)
    assert out.columns.tolist() == ["z"]
    assert out.shape[0] == dataframe.shape[0]