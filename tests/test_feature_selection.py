from morar import feature_selection
from morar import utils
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

np.random.seed(0)

a = np.random.random(100)
b = np.zeros(100)
c = np.random.random(100)
df = pd.DataFrame(list(zip(a, b, c)))
df.columns = ["a", "b", "c"]


def test_find_low_var():
    out = feature_selection.find_low_var(df)
    assert len(out) == 1
    assert out == ["b"]


def test_find_low_var_errors():
    with pytest.raises(ValueError):
        feature_selection.find_low_var(df["a"].tolist())


def test_find_low_var_threshold():
    sigma = 0.5
    var = sigma ** 2
    x = np.random.normal(loc=1, scale=sigma, size=1000)
    y = np.random.normal(loc=1, scale=1.0, size=1000)
    df2 = pd.DataFrame(list(zip(x, y)))
    df2.columns = ["x", "y"]
    out = feature_selection.find_low_var(df2, threshold=var * 2)
    assert out == ["x"]


def test_find_low_var_nan():
    # dataset containing NaN values
    x = [np.nan] * 10
    y = list(range(10))
    z = [1] * 10
    df_nan = pd.DataFrame(list(zip(x, y, z)))
    df_nan.columns = ["x", "y", "z"]
    out = feature_selection.find_low_var(df_nan)
    assert (out == ["x", "z"]) or (out == ["z", "x"])


def test_find_replicate_var():
    """morar.feature_selection.find_replicate_var"""
    # TODO create dataframe with groups and replicates with known variance
    # TODO calculate feature variance
    # TODO check against unknown
    # TODO check sorted sorts by variance
    pass
    # assert 2 + 2 == 5 # TODO


def test_find_correlation():
    x = range(1000)
    noise = np.random.randn(1000)
    y = [a + b for a, b in zip(x, noise)]
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = feature_selection.find_correlation(df)
    assert len(out) == 1
    assert out[0] == ["x"] or ["y"]
    assert out[0] != ["z"]


def test_find_correlation_threshold_works():
    x = range(1000)
    noise = np.random.randn(1000)
    y = [a + b for a, b in zip(x, noise)]
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = feature_selection.find_correlation(df, threshold=1.0)
    assert len(out) == 0


def test_find_correlation_multiple_correlated():
    x = range(1000)
    noise = np.random.randn(1000)
    y = [a + b for a, b in zip(x, noise)]
    xx = [a + b for a, b in zip(x, noise)]
    z = np.random.randn(1000)
    df = pd.DataFrame(list(zip(x, xx, y, z)), columns=["x", "xx", "y", "z"])
    out = feature_selection.find_correlation(df)
    assert len(out) == 2
    assert "z" not in out


def test_find_correlation_large_n():
    x = range(100000)
    noise = np.random.randn(100000)
    y = [a + b for a, b in zip(x, noise)]
    z = np.random.randn(100000)
    df = pd.DataFrame(list(zip(x, y, z)), columns=["x", "y", "z"])
    out = feature_selection.find_correlation(df)
    assert len(out) == 1
    assert out[0] == ["x"] or ["y"]
    assert out[0] != ["z"]


def test_feature_importance_errors_incorrect_compound_col():
    x = np.random.random(100)
    y = np.random.random(100)
    z = ["pos", "neg"] * 50
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["x", "y", "Metadata_compound"]
    with pytest.raises(ValueError):
        feature_selection.feature_importance(df, "pos", "neg", compound_col="incorrect")


def test_feature_importance_errors_wrong_control_names():
    x = np.random.random(100)
    y = np.random.random(100)
    z = ["pos", "neg"] * 50
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["x", "y", "Metadata_compound"]
    with pytest.raises(ValueError):
        feature_selection.feature_importance(
            df, "pos", "incorrect", compound_col="Metadata_compound"
        )


def test_feature_importance_errors_wrong_control_names2():
    x = np.random.random(100)
    y = np.random.random(100)
    z = ["pos", "neg"] * 50
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["x", "y", "Metadata_compound"]
    with pytest.raises(ValueError):
        feature_selection.feature_importance(
            df, "incorrect", "neg", compound_col="Metadata_compound"
        )


def test_feature_importance_errors_non_dataframe():
    x = np.random.random(100)
    y = np.random.random(100)
    z = ["pos", "neg"] * 50
    df = pd.DataFrame(list(zip(x, y, z)))
    df.columns = ["x", "y", "Metadata_compound"]
    with pytest.raises(ValueError):
        feature_selection.feature_importance(x, "pos", "neg", compound_col="incorrect")


def test_feature_importance_returns_all_feature_columns():
    x, y = make_classification(n_samples=100, n_features=10, n_informative=2)
    x = pd.DataFrame(x)
    x.columns = ["x" + str(i) for i in range(1, 11)]
    x["Metadata_compound"] = ["pos", "neg"] * 50
    out = feature_selection.feature_importance(
        data=x, neg_cmpd="neg", pos_cmpd="pos", compound_col="Metadata_compound"
    )
    assert len(list(out)) == 10


def test_feature_importance_sort():
    x, y = make_classification(n_samples=100, n_features=10, n_informative=2)
    x = pd.DataFrame(x)
    x.columns = ["x" + str(i) for i in range(1, 11)]
    x["Metadata_compound"] = ["pos", "neg"] * 50
    out = feature_selection.feature_importance(
        data=x,
        neg_cmpd="neg",
        pos_cmpd="pos",
        compound_col="Metadata_compound",
        sort=True,
    )
    f_names, importances = list(zip(*out))
    sorted_importances = sorted(list(importances), reverse=True)
    assert sorted_importances == list(importances)


def test_feature_importance_returns_colnames():
    x, y = make_classification(n_samples=100, n_features=10, n_informative=2)
    x = pd.DataFrame(x)
    x.columns = ["x" + str(i) for i in range(1, 11)]
    x["Metadata_compound"] = ["pos", "neg"] * 50
    out = feature_selection.feature_importance(
        data=x, neg_cmpd="neg", pos_cmpd="pos", compound_col="Metadata_compound"
    )
    f_names, importances = list(zip(*out))
    feature_col_names = utils.get_featuredata(x)
    assert list(f_names) == list(feature_col_names)


def test_select_features():
    x, y = make_classification(n_samples=100, n_features=10, n_informative=2)
    x = pd.DataFrame(x)
    x.columns = ["x" + str(i) for i in range(1, 11)]
    x["Metadata_compound"] = ["pos", "neg"] * 50
    out = feature_selection.select_features(
        data=x, neg_cmpd="neg", pos_cmpd="pos", compound_col="Metadata_compound"
    )
    assert isinstance(out, list)
    assert len(out) < len(x.columns.tolist())


def test_find_unwanted():
    """morar.feature_selection.find_unwanted()"""
    colnames = [
        "Nuclei_Intensity_IntegratedIntensityEdge_W1",
        "Nuclei_Intensity_IntegratedIntensity_W1",
        "Nuclei_Intensity_LowerQuartileIntensity_W1",
        "Nuclei_Intensity_MADIntensity_W1",
        "Nuclei_Intensity_MassDisplacement_W1",
        "Nuclei_Intensity_MaxIntensityEdge_W1",
        "Nuclei_Intensity_MaxIntensity_W1",
        "Nuclei_Intensity_MeanIntensityEdge_W1",
        "Nuclei_Intensity_MeanIntensity_W1",
        "Nuclei_Intensity_MedianIntensity_W1",
        "Nuclei_Intensity_MinIntensityEdge_W1",
        "Nuclei_Intensity_MinIntensity_W1",
        "Nuclei_Intensity_StdIntensityEdge_W1",
        "Nuclei_Intensity_StdIntensity_W1",
        "Nuclei_Intensity_UpperQuartileIntensity_W1",
        "Nuclei_Location_CenterMassIntensity_X_W1",
        "Nuclei_Location_CenterMassIntensity_Y_W1",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Nuclei_Location_MaxIntensity_X_W1",
        "Nuclei_Location_MaxIntensity_Y_W1",
        "Nuclei_Number_Object_Number",
        "Nuclei_Texture_AngularSecondMoment_W1_3_0",
        "Nuclei_Texture_AngularSecondMoment_W1_3_135",
        "Nuclei_Texture_AngularSecondMoment_W1_3_45",
        "Nuclei_Texture_AngularSecondMoment_W1_3_90",
    ]
    dat = np.random.randn(10, len(colnames))
    test_df = pd.DataFrame(dat, columns=colnames)
    unwanted = [
        "Nuclei_Location_CenterMassIntensity_X_W1",
        "Nuclei_Location_CenterMassIntensity_Y_W1",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Nuclei_Location_MaxIntensity_X_W1",
        "Nuclei_Location_MaxIntensity_Y_W1",
        "Nuclei_Number_Object_Number",
    ]
    ans = feature_selection.find_unwanted(test_df)
    assert sorted(ans) == sorted(unwanted)


def test_find_unwanted_extra():
    """morar.feature_selection.find_unwanted()"""
    colnames = [
        "Nuclei_Intensity_IntegratedIntensityEdge_W1",
        "Nuclei_Intensity_IntegratedIntensity_W1",
        "Nuclei_Intensity_LowerQuartileIntensity_W1",
        "Nuclei_Intensity_MADIntensity_W1",
        "Nuclei_Intensity_MassDisplacement_W1",
        "Nuclei_Intensity_MaxIntensityEdge_W1",
        "Nuclei_Intensity_MaxIntensity_W1",
        "Nuclei_Intensity_MeanIntensityEdge_W1",
        "Nuclei_Intensity_MeanIntensity_W1",
        "Nuclei_Intensity_MedianIntensity_W1",
        "Nuclei_Intensity_MinIntensityEdge_W1",
        "Nuclei_Intensity_MinIntensity_W1",
        "Nuclei_Intensity_StdIntensityEdge_W1",
        "Nuclei_Intensity_StdIntensity_W1",
        "Nuclei_Intensity_UpperQuartileIntensity_W1",
        "Nuclei_Location_CenterMassIntensity_X_W1",
        "Nuclei_Location_CenterMassIntensity_Y_W1",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Nuclei_Location_MaxIntensity_X_W1",
        "Nuclei_Location_MaxIntensity_Y_W1",
        "Nuclei_Number_Object_Number",
        "Nuclei_Texture_AngularSecondMoment_W1_3_0",
        "Nuclei_Texture_AngularSecondMoment_W1_3_135",
        "Nuclei_Texture_AngularSecondMoment_W1_3_45",
        "Nuclei_Texture_AngularSecondMoment_W1_3_90",
    ]
    dat = np.random.randn(10, len(colnames))
    test_df = pd.DataFrame(dat, columns=colnames)
    ans = feature_selection.find_unwanted(test_df, extra="Std")
    unwanted = [
        "Nuclei_Location_CenterMassIntensity_X_W1",
        "Nuclei_Location_CenterMassIntensity_Y_W1",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Nuclei_Location_MaxIntensity_X_W1",
        "Nuclei_Location_MaxIntensity_Y_W1",
        "Nuclei_Number_Object_Number",
        "Nuclei_Intensity_StdIntensityEdge_W1",
        "Nuclei_Intensity_StdIntensity_W1",
    ]
    assert sorted(ans) == sorted(unwanted)


def test_find_unwanted_extra_error():
    """morar.feature_selection.find_unwanted()"""
    colnames = [
        "Nuclei_Intensity_IntegratedIntensityEdge_W1",
        "Nuclei_Intensity_IntegratedIntensity_W1",
        "Nuclei_Intensity_LowerQuartileIntensity_W1",
        "Nuclei_Intensity_MADIntensity_W1",
        "Nuclei_Intensity_MassDisplacement_W1",
        "Nuclei_Intensity_MaxIntensityEdge_W1",
        "Nuclei_Intensity_MaxIntensity_W1",
        "Nuclei_Intensity_MeanIntensityEdge_W1",
        "Nuclei_Intensity_MeanIntensity_W1",
        "Nuclei_Intensity_MedianIntensity_W1",
        "Nuclei_Intensity_MinIntensityEdge_W1",
        "Nuclei_Intensity_MinIntensity_W1",
        "Nuclei_Intensity_StdIntensityEdge_W1",
        "Nuclei_Intensity_StdIntensity_W1",
        "Nuclei_Intensity_UpperQuartileIntensity_W1",
        "Nuclei_Location_CenterMassIntensity_X_W1",
        "Nuclei_Location_CenterMassIntensity_Y_W1",
        "Nuclei_Location_Center_X",
        "Nuclei_Location_Center_Y",
        "Nuclei_Location_MaxIntensity_X_W1",
        "Nuclei_Location_MaxIntensity_Y_W1",
        "Nuclei_Number_Object_Number",
        "Nuclei_Texture_AngularSecondMoment_W1_3_0",
        "Nuclei_Texture_AngularSecondMoment_W1_3_135",
        "Nuclei_Texture_AngularSecondMoment_W1_3_45",
        "Nuclei_Texture_AngularSecondMoment_W1_3_90",
    ]
    dat = np.random.randn(10, len(colnames))
    test_df = pd.DataFrame(dat, columns=colnames)
    with pytest.raises(TypeError):
        feature_selection.find_unwanted(test_df, extra=5)
