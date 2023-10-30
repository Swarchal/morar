"""
Functions used for feature selection.
"""

from morar import utils
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel


def find_correlation(data, threshold=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.

    Parameters
    -----------
    data : pandas DataFrame
        DataFrame
    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value

    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corr_mat = data[utils.get_featuredata(data)].corr()
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


def find_low_var(data, threshold=1e-5):
    """
    Return column names of feature columns with zero or very low variance

    Parameters
    ------------
    data : pandas DataFrame
        DataFrame
    threshold : float
        low variance threshold

    Returns
    -------
    columns : list
        list of columns to remove
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("not a pandas DataFrame")
    var = data[utils.get_featuredata(data)].var(axis=0)
    below_thresh = var[var <= threshold].index.tolist()
    is_nan = utils.is_all_nan(data)
    columns = list(below_thresh) + list(is_nan)
    return columns


def find_replicate_var(data, grouping, sorted_by_var=True):
    """
    Return within replicate variance of featuredata
    """
    variances = []
    feature_cols = utils.get_featuredata(data)
    for _, group in data.groupby(grouping):
        variance = group[feature_cols].var()
        variances.append(variance)
    var_mean = np.nanmean(np.vstack(variances), axis=1)
    feature_var = list(zip(feature_cols, var_mean))
    if sorted_by_var:
        feature_var.sort(key=lambda x: x[1])
    return feature_var


def feature_importance(
    data, neg_cmpd, pos_cmpd, compound_col="Metadata_compound", sort=False
):
    """
    Return features importances, based on separating the positive and negative
    controls in a random forest classifier.

    Parameters
    -----------
    data: pandas DataFrame
        DataFrame
    neg_cmpd : string
        name of negative control in compound_col
    pos_cmpd : string
        name of positive control in compound_col
    compound_col (default="Metadata_compound") : string
        name of column in df that contains compound labels
    sort : boolean (default=False)
        if True will sort the list of features on importance otherwise will
        return them in the original order

    Returns
    --------
    importances : list
        list of lists, feature name and importances
    """
    X, Y = _split_classes(data, neg_cmpd, pos_cmpd, compound_col)
    # create classifier
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X, Y)
    # extract feature importance from model
    importances = clf.feature_importances_
    col_names = X.columns.tolist()
    importances = zip(col_names, importances)
    # convert to list of lists rather than list of tuples
    importances = [list(elem) for elem in importances]
    if sort:
        importances.sort(key=lambda x: x[1], reverse=True)
    return importances


def select_features(data, neg_cmpd, pos_cmpd, compound_col="Metadata_compound", C=0.01):
    """
    Return selected features basd on L1 linear svc.

    Parameters
    -----------
    data : pandas DataFrame
    neg_cmpd : string
        name of negative control in compound_col
    pos_cmpd : string
        name of positive control in compound_col
    compound_col : string
        name of column in data that contains compound labels
    C : float (default=0.01)
        Sparsity, lower the number the fewer features are selected

    Returns
    -------
    selected_features : list
        Selected features
    """
    X, Y = _split_classes(data, neg_cmpd, pos_cmpd, compound_col)
    lin_svc = LinearSVC(C=C, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lin_svc, prefit=True)
    feature_mask = np.array(model.get_support())
    feature_names = np.array(X.columns.tolist())
    selected_features = list(feature_names[feature_mask])
    return selected_features


def _split_classes(data, neg_cmpd, pos_cmpd, compound_col):
    """
    Internal function used to separate featuredata and compound labels for
    classification.

    Parameters
    -----------
    data : pandas DataFrame
    neg_cmpd : string
        name of negative control in compound_col
    pos_cmpd : string
        name of positive control in compound_col
    compound_col : string
        name of column in df that contains compound labels

    Returns
    --------
    classes: list
        [X, Y], where X is the dataframe containing feature columns, and Y
        is the list of integers matching to postive or negative controls.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("is not a pandas DataFrame")
    if compound_col not in data.columns:
        raise ValueError("{} is not a column in data".format(compound_col))
    if neg_cmpd not in data[compound_col].tolist():
        raise ValueError("{} is not in column {}".format(neg_cmpd, compound_col))
    if pos_cmpd not in data[compound_col].tolist():
        raise ValueError("{} is not in column {}".format(pos_cmpd, compound_col))
    # split data into just positive and negative controls
    controls = [neg_cmpd, pos_cmpd]
    df_cntrl = data[data[compound_col].isin(controls)].copy()
    # convert compound labels to integers. pos_cmpd=1, neg_cmpd=0
    cntrl_int = pd.Categorical(df_cntrl[compound_col]).codes.tolist()
    df_cntrl[compound_col] = cntrl_int
    # select just feature data
    X = df_cntrl[utils.get_featuredata(df_cntrl)]
    Y = df_cntrl[compound_col].tolist()
    return [X, Y]


def find_unwanted(data, extra=None):
    """
    Return list of typically unwanted columns such as object number of
    object X,Y position

    Parameters:
    -----------
    data : pd.DataFrame
        data
    extra : string or list of strings
        Any columns containing strings in extra will also be listed

    Returns:
    --------
    List of column names
    """
    to_remove = set()
    colnames = data.columns.tolist()
    unwanted = [
        "Location_Center",
        "Object_Number",
        "ObjectNumber",
        "_Children_",
        "AreaShape_Center_",
        "Parent_",
        "Location_MaxIntensity",
        "Location_Center",
        "AngleBetweenNeighbors",
        "ImageNumber",
        "EulerNumber",
        "_Location_",
    ]
    if extra is not None:
        if isinstance(extra, str):
            unwanted.append(extra)
        elif isinstance(extra, list):
            unwanted.extend(extra)
        else:
            raise TypeError("extra needs to be a list or a string")
    for column in colnames:
        for name in unwanted:
            if name in column:
                to_remove.add(column)
    return list(to_remove)
