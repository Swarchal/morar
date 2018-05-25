"""
test morar.dataframe.DataFrame
"""
import os
import morar
import numpy as np
import pandas as pd

np.random.seed(42)
CELL_AREA = np.random.randn(10)
NUCLEI_SHAPE = np.random.randn(10)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
my_data_path = os.path.join(THIS_DIR, "test_data/single_cell_test_data.csv")

# create example dataframe
TEST_DICT = {
    "cell_area": CELL_AREA,
    "nuclei_shape": NUCLEI_SHAPE,
    "Metadata_compound": ["drug_{}".format(i) for i in "AABBCCDDEE"],
    "Metadata_cell_line": ["cell_line_A", "cell_line_B"] * 5,
}
TEST_DF = pd.DataFrame(TEST_DICT)
M_TEST_DF = morar.DataFrame(TEST_DF)


def test_featurecols():
    """morar.dataframe.DataFrame.featurecols"""
    output = M_TEST_DF.featurecols
    assert output == ["cell_area", "nuclei_shape"]


def test_featuredata():
    """morar.dataframe.DataFrame.featuredata"""
    output = M_TEST_DF.featuredata
    assert isinstance(output, morar.dataframe.DataFrame)
    cols_sorted = sorted(output.columns.tolist())
    assert cols_sorted == sorted(["cell_area", "nuclei_shape"])
    assert all(output.cell_area == CELL_AREA)
    assert all(output.nuclei_shape == NUCLEI_SHAPE)


def test_metacols():
    """morar.dataframe.DataFrame.metacols"""
    output = sorted(M_TEST_DF.metacols)
    assert output == sorted(["Metadata_compound", "Metadata_cell_line"])


def test_metadata():
    """morar.dataframe.DataFrame.metadata"""
    output = M_TEST_DF.metadata
    assert isinstance(output, morar.dataframe.DataFrame)
    cols_sorted = sorted(output.columns.tolist())
    assert cols_sorted == sorted(["Metadata_compound", "Metadata_cell_line"])


def test_different_metadata_prefix():
    """morar.dataframe.DataFrame(self, metadata_prefix)"""
    # change the metadata prefix on the test dataframe
    test_df = M_TEST_DF.copy()
    metadata_prefix = "Image_Metadata_"
    new_col_names = [i.replace("Metadata_", metadata_prefix) for i in test_df.columns]
    test_df.columns = new_col_names
    test_df = morar.DataFrame(test_df, metadata_string=metadata_prefix)
    metacols = test_df.metacols
    featurecols = test_df.featurecols
    assert all(i.startswith(metadata_prefix) for i in metacols)
    assert all(not i.startswith(metadata_prefix) for i in featurecols)
    assert len(metacols) > 0
    assert len(featurecols) > 0


def test_scale_features():
    """morar.dataframe.DataFrame.scale_features()"""
    output = M_TEST_DF.scale_features()
    assert isinstance(output, morar.dataframe.DataFrame)
    # get just the featuredata
    f_data = output.featuredata
    for column in f_data:
        assert f_data[column].mean() < 1e-6
        # with this few observations then variance doesn't quite equal 1
        assert abs(f_data[column].std() - 1) < 0.1
        assert abs(f_data[column].std() - 1) < 0.1


def test_scale_feature_diff_metadata_prefix():
    """morar.dataframe.DataFrame.scale_features() with diff metadata prefix"""
    test_df = M_TEST_DF.copy()
    metadata_prefix = "Image_Metadata_"
    new_col_names = [i.replace("Metadata_", metadata_prefix) for i in test_df.columns]
    test_df.columns = new_col_names
    test_df = morar.DataFrame(test_df, metadata_string="Image_Metadata_")
    output = test_df.scale_features()
    print(output)
    print(test_df.metadata_string)
    print(test_df.prefix)
    assert isinstance(output, morar.dataframe.DataFrame)
    # get just the featuredata
    f_data = output.featuredata
    for column in f_data:
        assert f_data[column].mean() < 1e-6
        # with this few observations then variance doesn't quite equal 1
        assert abs(f_data[column].std() - 1) < 0.1
        assert abs(f_data[column].std() - 1) < 0.1


def test_query():
    """morar.dataframe.DataFrame.query()"""
    # return just cell_line_A
    query_str = "Metadata_cell_line == 'cell_line_A'"
    output = M_TEST_DF.query(query_str)
    assert isinstance(output, morar.dataframe.DataFrame)
    assert output.shape[0] == 5
    assert output.Metadata_cell_line.unique().tolist() == ["cell_line_A"]


def test_merge():
    """morar.dataframe.DataFrame.merge()"""
    # TODO
    pass


def test_normalise():
    """morar.dataframe.DataFrame.normalise()"""
    # TODO
    pass


def test_aggregate():
    """morar.dataframe.DataFrame.aggregate()"""
    df = morar.DataFrame(pd.read_csv(my_data_path), prefix=False)
    out = df.agg(on="Image_ImageNumber")
    n_imagesets = len(set(df.Image_ImageNumber))
    assert out.shape[0] == n_imagesets


def test_aggregate_diff_metadata_prefix():
    """morar.dataframe.DataFrame.aggregate()"""
    df = pd.read_csv(my_data_path)
    df.columns = [i.replace("Image_Metadata_", "Image_metadata_") for i in df.columns]
    df = morar.DataFrame(df, metadata_string="Image_metadata_")
    out = df.agg(on="Image_ImageNumber")
    n_imagesets = len(df.Image_ImageNumber.unique())
    assert out.shape[0] == n_imagesets
    # check the output inherits metadata strings
    assert out.metadata_string == "Image_metadata_"


def test_pca():
    """morar.dataframe.DataFrame.pca()"""
    pca_df, var = M_TEST_DF.pca()
    assert isinstance(pca_df, morar.dataframe.DataFrame)
    assert M_TEST_DF.shape == pca_df.shape
    assert pca_df.featurecols == ["PC1", "PC2"]
    assert abs(np.sum(var) - 1) < 0.1
