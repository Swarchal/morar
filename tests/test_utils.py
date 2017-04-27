from morar import utils
import pandas as pd
import numpy as np
import pytest

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


def test_get_image_quality():
    # create simple dataframe with ImageQuality columns
    x = [1, 2, 3]
    y = [2, 4, 1]
    z = [2, 5, 1]
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["vals", "ImageQuality_test", "other"]
    out = utils.get_image_quality(df)
    print(out)
    assert out == ["ImageQuality_test"]


def test_get_image_quality_not_beginning():
    # column has ImageQuality in middle of string
    # create simple dataframe with ImageQuality columns
    x = [1, 2, 3]
    y = [2, 4, 1]
    z = [2, 5, 1]
    df2 = pd.DataFrame(list(zip(x, y, z)))
    df2.columns = ["vals", "ImageQuality_test", "Cells_ImageQuality"]
    out = utils.get_image_quality(df2)
    assert out == ["ImageQuality_test", "Cells_ImageQuality"]


def test_get_image_quality_fails_non_dataframe():
    # create simple dataframe with ImageQuality columns
    x = [1, 2, 3]
    y = [2, 4, 1]
    z = [2, 5, 1]
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["vals", "ImageQuality_test", "other"]
    test_list = df["ImageQuality_test"].tolist()
    with pytest.raises(ValueError):
        utils.get_image_quality(test_list)


def test_get_image_quality_no_im_qc_cols():
    x = [1,2,3,4]
    y = [2,3,4,5]
    df = pd.DataFrame(list(zip(x, y)))
    df.columns = ["x", "y"]
    with pytest.raises(ValueError):
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


def test_drop_bad_threshold_high():
    x = [1, 2, 3, np.nan]
    y = [1, 2, 3, 4]
    dataframe = pd.DataFrame(list(zip(x, y)))
    dataframe.columns = ["x", "y"]
    with pytest.raises(ValueError):
        utils.drop(dataframe, threshold=10)


def test_drop_bad_threshold_low():
    x = [1, 2, 3, np.nan]
    y = [1, 2, 3, 4]
    dataframe = pd.DataFrame(list(zip(x, y)))
    dataframe.columns = ["x", "y"]
    with pytest.raises(ValueError):
        utils.drop(dataframe, threshold=-5)


def test_drop_correct():
    x = [np.nan]*10
    xx = [np.nan]*10
    y = list(range(10))
    z = list(range(10))
    dataframe = pd.DataFrame(list(zip(x, xx, y, z)))
    dataframe.columns = ["x", "xx", "y", "z"]
    out = utils.drop(dataframe)
    assert out.shape[0] == dataframe.shape[0]
    assert out.columns.tolist() == ["y", "z"]


def test_drop_corrrect_rows():
    x = [np.nan]*50
    y = np.random.random(50)
    z = list(range(49)) + [np.nan]
    dataframe = pd.DataFrame(list(zip(x, y, z)))
    dataframe.columns = ["x", "y", "z"]
    out = utils.drop(dataframe)
    assert out.columns.tolist() == ["y", "z"]
    assert out.shape[0] == 49


def test_drop_threshold():
    x = list(range(50)) + [np.nan]*50
    y = [np.nan]*100
    z = np.random.random(100)
    dataframe = pd.DataFrame(list(zip(x, y, z)))
    dataframe.columns = ["x", "y", "z"]
    out = utils.drop(dataframe, threshold=0.2)
    assert out.columns.tolist() == ["z"]
    assert out.shape[0] == dataframe.shape[0]


def test_merge_to_cols():
    x = [1, 2, 3, None, None]
    y = [None, None, None, 4, 5]
    df = pd.DataFrame({"x" : x, "y" : y})
    new_col = utils.merge_two_cols(df, "x", "y")
    assert isinstance(new_col, pd.Series)
    assert new_col.tolist() == [1, 2, 3, 4, 5]


def test_img_to_metadata():
    # make some example data
    colnames = ['Correlation_Correlation_W2_W3',
                'Correlation_Costes_W2_W3',
                'Correlation_Costes_W3_W2',
                'Correlation_K_W2_W3',
                'Correlation_K_W3_W2',
                'Correlation_Manders_W2_W3',
                'Correlation_Manders_W3_W2',
                'Correlation_Overlap_W2_W3',
                'Correlation_RWC_W2_W3',
                'Correlation_RWC_W3_W2',
                'Correlation_Slope_W2_W3',
                'Count_Cells',
                'Count_Nuclei',
                'ExecutionTime_01LoadData',
                'ExecutionTime_02IdentifyPrimaryObjects',
                'ExecutionTime_03ImageMath',
                'ExecutionTime_04IdentifySecondaryObjects',
                'ExecutionTime_05MeasureObjectSizeShape',
                'ExecutionTime_06MeasureImageQuality',
                'ExecutionTime_07MeasureObjectIntensity',
                'ExecutionTime_08MeasureObjectIntensity',
                'ExecutionTime_09MeasureObjectIntensityDistribution',
                'ExecutionTime_10MeasureObjectNeighbors',
                'ExecutionTime_11MeasureTexture',
                'ExecutionTime_12MeasureTexture',
                'ExecutionTime_13MeasureCorrelation',
                'ExecutionTime_14MeasureGranularity',
                'ExecutionTime_15MeasureGranularity',
                'FileName_W1',
                'FileName_W2',
                'FileName_W3',
                'FileName_W4',
                'FileName_W5',
                'Granularity_10_W1',
                'Granularity_10_W4',
                'Granularity_10_W5',
                'Granularity_11_W1',
                "Metadata_well"]
    dat = np.random.randn(10, len(colnames))
    test_df = pd.DataFrame(dat, columns=colnames)
    ans = utils.img_to_metadata(test_df, prefix="Metadata_")
    expected = ['Correlation_Correlation_W2_W3',
                'Correlation_Costes_W2_W3',
                'Correlation_Costes_W3_W2',
                'Correlation_K_W2_W3',
                'Correlation_K_W3_W2',
                'Correlation_Manders_W2_W3',
                'Correlation_Manders_W3_W2',
                'Correlation_Overlap_W2_W3',
                'Correlation_RWC_W2_W3',
                'Correlation_RWC_W3_W2',
                'Correlation_Slope_W2_W3',
                'Count_Cells',
                'Count_Nuclei',
                'Metadata_ExecutionTime_01LoadData',
                'Metadata_ExecutionTime_02IdentifyPrimaryObjects',
                'Metadata_ExecutionTime_03ImageMath',
                'Metadata_ExecutionTime_04IdentifySecondaryObjects',
                'Metadata_ExecutionTime_05MeasureObjectSizeShape',
                'Metadata_ExecutionTime_06MeasureImageQuality',
                'Metadata_ExecutionTime_07MeasureObjectIntensity',
                'Metadata_ExecutionTime_08MeasureObjectIntensity',
                'Metadata_ExecutionTime_09MeasureObjectIntensityDistribution',
                'Metadata_ExecutionTime_10MeasureObjectNeighbors',
                'Metadata_ExecutionTime_11MeasureTexture',
                'Metadata_ExecutionTime_12MeasureTexture',
                'Metadata_ExecutionTime_13MeasureCorrelation',
                'Metadata_ExecutionTime_14MeasureGranularity',
                'Metadata_ExecutionTime_15MeasureGranularity',
                'Metadata_FileName_W1',
                'Metadata_FileName_W2',
                'Metadata_FileName_W3',
                'Metadata_FileName_W4',
                'Metadata_FileName_W5',
                'Granularity_10_W1',
                'Granularity_10_W4',
                'Granularity_10_W5',
                'Granularity_11_W1',
                "Metadata_well"]
    assert ans == expected


def test_img_to_metadata_extra_str():
    # make some example data
    colnames = ['Correlation_Correlation_W2_W3',
                'Correlation_Costes_W2_W3',
                'Correlation_Costes_W3_W2',
                'Correlation_K_W2_W3',
                'Correlation_K_W3_W2',
                'Correlation_Manders_W2_W3',
                'Correlation_Manders_W3_W2',
                'Correlation_Overlap_W2_W3',
                'Correlation_RWC_W2_W3',
                'Correlation_RWC_W3_W2',
                'Correlation_Slope_W2_W3',
                'Count_Cells',
                'Count_Nuclei',
                'ExecutionTime_01LoadData',
                'ExecutionTime_02IdentifyPrimaryObjects',
                'ExecutionTime_03ImageMath',
                'ExecutionTime_04IdentifySecondaryObjects',
                'ExecutionTime_05MeasureObjectSizeShape',
                'ExecutionTime_06MeasureImageQuality',
                'ExecutionTime_07MeasureObjectIntensity',
                'ExecutionTime_08MeasureObjectIntensity',
                'ExecutionTime_09MeasureObjectIntensityDistribution',
                'ExecutionTime_10MeasureObjectNeighbors',
                'ExecutionTime_11MeasureTexture',
                'ExecutionTime_12MeasureTexture',
                'ExecutionTime_13MeasureCorrelation',
                'ExecutionTime_14MeasureGranularity',
                'ExecutionTime_15MeasureGranularity',
                'FileName_W1',
                'FileName_W2',
                'FileName_W3',
                'FileName_W4',
                'FileName_W5',
                'Granularity_10_W1',
                'Granularity_10_W4',
                'Granularity_10_W5',
                'Granularity_11_W1',
                'Metadata_well',
                'Cells_AreaShape_Area']
    dat = np.random.randn(10, len(colnames))
    test_df = pd.DataFrame(dat, columns=colnames)
    ans = utils.img_to_metadata(test_df, prefix="Metadata_", extra="Cells")
    expected = ['Correlation_Correlation_W2_W3',
                'Correlation_Costes_W2_W3',
                'Correlation_Costes_W3_W2',
                'Correlation_K_W2_W3',
                'Correlation_K_W3_W2',
                'Correlation_Manders_W2_W3',
                'Correlation_Manders_W3_W2',
                'Correlation_Overlap_W2_W3',
                'Correlation_RWC_W2_W3',
                'Correlation_RWC_W3_W2',
                'Correlation_Slope_W2_W3',
                'Count_Cells',
                'Count_Nuclei',
                'Metadata_ExecutionTime_01LoadData',
                'Metadata_ExecutionTime_02IdentifyPrimaryObjects',
                'Metadata_ExecutionTime_03ImageMath',
                'Metadata_ExecutionTime_04IdentifySecondaryObjects',
                'Metadata_ExecutionTime_05MeasureObjectSizeShape',
                'Metadata_ExecutionTime_06MeasureImageQuality',
                'Metadata_ExecutionTime_07MeasureObjectIntensity',
                'Metadata_ExecutionTime_08MeasureObjectIntensity',
                'Metadata_ExecutionTime_09MeasureObjectIntensityDistribution',
                'Metadata_ExecutionTime_10MeasureObjectNeighbors',
                'Metadata_ExecutionTime_11MeasureTexture',
                'Metadata_ExecutionTime_12MeasureTexture',
                'Metadata_ExecutionTime_13MeasureCorrelation',
                'Metadata_ExecutionTime_14MeasureGranularity',
                'Metadata_ExecutionTime_15MeasureGranularity',
                'Metadata_FileName_W1',
                'Metadata_FileName_W2',
                'Metadata_FileName_W3',
                'Metadata_FileName_W4',
                'Metadata_FileName_W5',
                'Granularity_10_W1',
                'Granularity_10_W4',
                'Granularity_10_W5',
                'Granularity_11_W1',
                "Metadata_well",
                "Cells_AreaShape_Area"]
    assert ans == expected



def test_img_to_metadata_extra_list():
    # make some example data
    colnames = ['Correlation_Correlation_W2_W3',
                'Correlation_Costes_W2_W3',
                'Correlation_Costes_W3_W2',
                'Correlation_K_W2_W3',
                'Correlation_K_W3_W2',
                'Correlation_Manders_W2_W3',
                'Correlation_Manders_W3_W2',
                'Correlation_Overlap_W2_W3',
                'Correlation_RWC_W2_W3',
                'Correlation_RWC_W3_W2',
                'Correlation_Slope_W2_W3',
                'Count_Cells',
                'Count_Nuclei',
                'ExecutionTime_01LoadData',
                'ExecutionTime_02IdentifyPrimaryObjects',
                'ExecutionTime_03ImageMath',
                'ExecutionTime_04IdentifySecondaryObjects',
                'ExecutionTime_05MeasureObjectSizeShape',
                'ExecutionTime_06MeasureImageQuality',
                'ExecutionTime_07MeasureObjectIntensity',
                'ExecutionTime_08MeasureObjectIntensity',
                'ExecutionTime_09MeasureObjectIntensityDistribution',
                'ExecutionTime_10MeasureObjectNeighbors',
                'ExecutionTime_11MeasureTexture',
                'ExecutionTime_12MeasureTexture',
                'ExecutionTime_13MeasureCorrelation',
                'ExecutionTime_14MeasureGranularity',
                'ExecutionTime_15MeasureGranularity',
                'FileName_W1',
                'FileName_W2',
                'FileName_W3',
                'FileName_W4',
                'FileName_W5',
                'Granularity_10_W1',
                'Granularity_10_W4',
                'Granularity_10_W5',
                'Granularity_11_W1',
                'Metadata_well',
                'Cells_AreaShape_Area',
                "Nuclei_AreaShape_Area"]
    dat = np.random.randn(10, len(colnames))
    test_df = pd.DataFrame(dat, columns=colnames)
    ans = utils.img_to_metadata(test_df, prefix="Metadata_", extra=["Cells", "Nuclei"])
    expected = ['Correlation_Correlation_W2_W3',
                'Correlation_Costes_W2_W3',
                'Correlation_Costes_W3_W2',
                'Correlation_K_W2_W3',
                'Correlation_K_W3_W2',
                'Correlation_Manders_W2_W3',
                'Correlation_Manders_W3_W2',
                'Correlation_Overlap_W2_W3',
                'Correlation_RWC_W2_W3',
                'Correlation_RWC_W3_W2',
                'Correlation_Slope_W2_W3',
                'Count_Cells',
                'Count_Nuclei',
                'Metadata_ExecutionTime_01LoadData',
                'Metadata_ExecutionTime_02IdentifyPrimaryObjects',
                'Metadata_ExecutionTime_03ImageMath',
                'Metadata_ExecutionTime_04IdentifySecondaryObjects',
                'Metadata_ExecutionTime_05MeasureObjectSizeShape',
                'Metadata_ExecutionTime_06MeasureImageQuality',
                'Metadata_ExecutionTime_07MeasureObjectIntensity',
                'Metadata_ExecutionTime_08MeasureObjectIntensity',
                'Metadata_ExecutionTime_09MeasureObjectIntensityDistribution',
                'Metadata_ExecutionTime_10MeasureObjectNeighbors',
                'Metadata_ExecutionTime_11MeasureTexture',
                'Metadata_ExecutionTime_12MeasureTexture',
                'Metadata_ExecutionTime_13MeasureCorrelation',
                'Metadata_ExecutionTime_14MeasureGranularity',
                'Metadata_ExecutionTime_15MeasureGranularity',
                'Metadata_FileName_W1',
                'Metadata_FileName_W2',
                'Metadata_FileName_W3',
                'Metadata_FileName_W4',
                'Metadata_FileName_W5',
                'Granularity_10_W1',
                'Granularity_10_W4',
                'Granularity_10_W5',
                'Granularity_11_W1',
                "Metadata_well",
                "Cells_AreaShape_Area",
                "Nuclei_AreaShape_Area"]
    assert ans == expected


def test_inflate_cols():
    """morar.utils.inflate_cols()"""
    pass


def test_collapse_cols():
    """morar.utils.collapse_cols()"""
    pass
