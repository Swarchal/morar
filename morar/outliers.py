import numpy as np
import pandas as pd

from morar import stats, utils

"""
Function(s) for finding outliers. Outliers are normally caused by out-of-focus
images or debris within wells which can cause extreme values upon segmentation.
"""


# TODO: IsolationForest outlier detection


def get_outlier_index(
    data: pd.DataFrame,
    method: str = "values",
    sigma: int | float = 6,
    adjust: bool = True,
    **kwargs
) -> list:
    """
    Returns index of outlying row(s)

    Parameters
    ----------
    data: pandas dataframe
        DataFrame
    method : string (default="values")
        either 'simple' which is based on hampels robust outlier
        test on feature values, or 'ImageQualty' which uses the
        ImageQualty metrics - FocusScore and PowerLogLogSlope.
    sigma : int (default=6)
        number of median absolute deviations away from the sample median to
        define an outlier.
    adjust: boolean (default=True)
        If true will adjust the sigma value to take into account multiple
        measurements. `sigma_adj = sigma * n_feature_columns`
    **kwargs: additional arguments to utils.get_featuredata

    Returns
    -------
    bad_index : list
        list of row index/indices to remove
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("not a pandas DataFrame")
    if method == "values":
        feature_cols = utils.get_featuredata(data, **kwargs)
        # FIXME really crude correction
        if adjust:
            sigma = sigma * len(feature_cols)
        hampel_out = data[feature_cols].apply(stats.hampel, sigma=sigma)
        hampel_abs = hampel_out.apply(lambda x: sum(abs(x)), axis=1)
        return hampel_abs[hampel_abs > 0].index.tolist()
    elif method == "ImageQuality":
        qc_cols = utils.get_image_quality(data)
        # find bad images with FocusScore
        focus_cols = [col for col in qc_cols if "FocusScore" in col]
        hampel_focus = data[focus_cols].apply(stats.hampel, sigma=sigma)
        focus_sum = hampel_focus.apply(np.sum, axis=1)
        focus_bad = focus_sum[focus_sum < 0].index.tolist()
        # find bad images with PowerLogLogSlope
        plls_cols = [col for col in qc_cols if "PowerLogLogSlope" in col]
        hampel_plls = data[plls_cols].apply(stats.hampel, sigma=sigma)
        plls_sum = hampel_plls.apply(np.sum, axis=1)
        plls_bad = plls_sum[plls_sum < 0].index.tolist()
        bad_index = list(set(focus_bad + plls_bad))
        return bad_index
    else:
        raise ValueError("invalid argument. Options: values, ImageQuality")
