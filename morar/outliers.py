from morar import utils
from morar import stats
import pandas as pd

def get_image_quality(df):
    """
    Returns list of column names from the ImageQuality module that are present
    in df.

    @param df pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(df, "is not a pandas DataFrame")
    colnames = df.columns.tolist()
    return [col for col in colnames if "ImageQuality" in col]


def get_outlier_index(df, method="values", sigma=6):
    """
    Returns index of outlying row(s)

    @param df pandas dataframe
    @param method either 'simple' which is based on hampels robust outlier
                  test on feature values, or 'ImageQualty' which uses the
                  ImageQualty metrics.
    @param sigma number of median absolute deviations away from the sample
                 median to define an outlier. Used only in method="values"
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(df, "is not a pandas DataFrame")
    accepted_methods = ["values", "ImageQuality"]
    if method not in accepted_methods:
        raise ValueError("invalid argument. Options: simple, ImageQuality")
    # TODO ImageQuality: use get_image_quality, and use rules based on out
    #      out-focus images. Return index
    if method == "values":
        feature_cols = utils.get_featuredata(df)
        hampel_out = df[feature_cols].apply(stats.hampel, sigma=sigma)
        hampel_abs = hampel_out.apply(lambda x: sum(abs(x)), axis=1)
        return hampel_abs[hampel_abs > 0].index.tolist()
    if method == "ImageQuality":
        qc_cols = get_image_quality(df)
        df_qc = df[qc_cols]
        # TODO find powerloglogslope, which value is bad
        # TODO find focus_score, which value is bad
        # TODO define cutoff and return index of rows beyond normal cutoff
        raise NotImplementedError("not done this yet, chill out")