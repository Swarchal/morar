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
my_data_path = os.path.join(THIS_DIR, 'test_data/single_cell_test_data.csv')

# create example dataframe
TEST_DICT = {
    "cell_area"          : CELL_AREA,
    "nuclei_shape"       : NUCLEI_SHAPE,
    "Metadata_compound"  : ["drug_{}".format(i) for i in "AABBCCDDEE"],
    "Metadata_cell_line" : ["cell_line_A", "cell_line_B"] * 5
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
    # TODO
    df = morar.DataFrame(pd.read_csv(my_data_path))
    out = df.agg(on="Image_ImageNumber", prefix=False)
    n_imagesets = len(set(df.Image_ImageNumber))
    assert out.shape[0] == n_imagesets


def test_pca():
    """morar.dataframe.DataFrame.pca()"""
    pca_df, var = M_TEST_DF.pca()
    assert isinstance(pca_df, morar.dataframe.DataFrame)
    assert M_TEST_DF.shape == pca_df.shape
    assert pca_df.featurecols == ["PC1", "PC2"]
    assert abs(np.sum(var) - 1) < 0.1
