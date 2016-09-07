from morar import utils
from morar import stats
import pandas as pd


def get_outlier_index(df, method="values", sigma=6):
    """
    Returns index of outlying row(s)

    Parameters
    ----------
    df : pandas dataframe
        DataFrame

    method : string (default="values")
        either 'simple' which is based on hampels robust outlier
        test on feature values, or 'ImageQualty' which uses the
        ImageQualty metrics - FocusScore and PowerLogLogSlope.

    sigma : int (default=6)
        number of median absolute deviations away from the sample median to
        define an outlier.

    Returns
    -------
    bad_index : list
        list of row index/indices to remove
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("not a pandas DataFrame")
    accepted_methods = ["values", "ImageQuality"]
    if method not in accepted_methods:
        raise ValueError("invalid argument. Options: simple, ImageQuality")
    if method == "values":
        feature_cols = utils.get_featuredata(df)
        hampel_out = df[feature_cols].apply(stats.hampel, sigma=sigma)
        hampel_abs = hampel_out.apply(lambda x: sum(abs(x)), axis=1)
        return hampel_abs[hampel_abs > 0].index.tolist()
    if method == "ImageQuality":
        qc_cols = utils.get_image_quality(df)
        df_qc = df[qc_cols]
        # find bad images with FocusScore
        focus_cols = [col for col in qc_cols if "FocusScore" in col]
        hampel_focus = df[focus_cols].apply(stats.hampel, sigma=sigma)
        focus_sum = hampel_focus.apply(lambda x: sum(x), axis=1)
        focus_bad = focus_sum[focus_sum < 0].index.tolist()
        # find bad images with PowerLogLogSlope
        plls_cols = [col for col in qc_cols if "PowerLogLogSlope" in col]
        hampel_plls = df[plls_cols].apply(stats.hampel, sigma=sigma)
        plls_sum = hampel_focus.apply(lambda x: sum(x), axis=1)
        plls_bad = plls_sum[plls_sum < 0].index.tolist()
        bad_index = list(set(focus_bad + plls_bad))
        return bad_index
