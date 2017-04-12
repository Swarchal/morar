"""
test morar.dataframe.DataFrame
"""
import morar
import numpy as np
import pandas as pd

np.random.seed(42)
CELL_AREA = np.random.randn(10)
NUCLEI_SHAPE = np.random.randn(10)

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
    # TODO
    # check the two feature columns are actually scaled, i.e mean and std of 1


def test_query():
    """morar.dataframe.DataFrame.query()"""
    pass


def test_merge():
    """morar.dataframe.DataFrame.merge()"""
    pass


def test_pca():
    """morar.dataframe.DataFrame.pca()"""
    pass
