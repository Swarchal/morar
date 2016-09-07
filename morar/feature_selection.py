from morar import utils
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

def find_correlation(df, threshold=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove.

    Parameters
    -----------
    df : pandas DataFrame
        DataFrame

    threshold : float
        correlation threshold, will remove one of pairs of features with a
        correlation greater than this value

    Returns
    --------
    select_flat : list
        listof column names to be removed
    """
    corrMatrix = df.corr()
    corrMatrix.loc[:,:] =  np.tril(corrMatrix, k=-1)
    already_in = set()
    result = []
    for col in corrMatrix:
        perfect_corr = corrMatrix[col][corrMatrix[col] > threshold].index.tolist()
        if perfect_corr and col not in already_in:
            already_in.update(set(perfect_corr))
            perfect_corr.append(col)
            result.append(perfect_corr)
    select_nested = [f[1:] for f in result]
    select_flat = [i for j in select_nested for i in j]
    return select_flat


def find_low_var(df, threshold=1e-5):
    """
    Return column names of feature columns with zero or very low variance

    Parameters
    ------------
    df : pandas DataFrame
        DataFrame
    threshold : float
        low variance threshold

    Returns
    -------
    columns : list
        list of columns to remove
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("not a pandas DataFrame")
    var = df[utils.get_featuredata(df)].var(axis=0)
    below_thresh = var[var < threshold].index.tolist()
    is_nan = utils.is_all_nan(df)
    columns = list(below_thresh) + list(is_nan)
    return columns


def feature_importance(df, neg_cmpd, pos_cmpd,
                            compound_col="Metadata_compound", sort=False):
    """
    Return features importances, based on separating the positive and negative
    controls in a random forest classifier.

    Parameters
    -----------
    df: pandas DataFrame
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
    X, Y = split_classes(df, neg_cmpd, pos_cmpd, compound_col)
    # create classifier
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X, Y)
    # extract feature importance from model
    importances = clf.feature_importances_
    col_names = X.columns.tolist()
    importances = list(zip(col_names, importances))
    # convert to list of lists rather than list of tuples
    importances = [list(elem) for elem in importances]
    if sort:
        importances.sort(key=lambda x: x[1], reverse=True)
    return importances


def select_features(df, neg_cmpd, pos_cmpd, compound_col="Metadata_compound",
                    C=0.01):
    """
    Return selected features basd on L1 linear svc.

    Parameters
    -----------
    df : pandas DataFrame

    neg_cmpd : string
        name of negative control in compound_col

    pos_cmpd : string
        name of positive control in compound_col

    compound_col : string
        name of column in df that contains compound labels

    C : float (default=0.01)
        Sparsity, lower the number the fewer features are selected

    Returns
    -------
    selected_features : list
        Selected features
    """
    X, Y = split_classes(df, neg_cmpd, pos_cmpd, compound_col)
    lin_svc = LinearSVC(C=C, penalty="l1", dual=False).fit(X, Y)
    model = SelectFromModel(lin_svc, prefit=True)
    feature_mask = np.array(model.get_support())
    feature_names = np.array(X.columns.tolist())
    selected_features = list(feature_names[feature_mask])
    return selected_features



def split_classes(df, neg_cmpd, pos_cmpd, compound_col):
    """
    Internal function used to separate featuredata and compound labels for
    classification.

    Parameters
    -----------
    df : pandas DataFrame

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
    if not isinstance(df, pd.DataFrame):
        raise ValueError("is not a pandas DataFrame")
    if compound_col not in df.columns:
        raise ValueError("{} is not a column in df".format(compound_col))
    if neg_cmpd not in df[compound_col].tolist():
        raise ValueError("{} is not in column {}".format(neg_cmpd, compound_col))
    if pos_cmpd not in df[compound_col].tolist():
        raise ValueError("{} is not in column {}".format(pos_cmpd, compound_col))
    #split data into just positive and negative controls
    controls = [neg_cmpd, pos_cmpd]
    df_cntrl = df[df[compound_col].isin(controls)].copy()
    # convert compound labels to integers. pos_cmpd=1, neg_cmpd=0
    cntrl_int = pd.Categorical(df_cntrl[compound_col]).codes.tolist()
    df_cntrl[compound_col] = cntrl_int
    # select just feature data
    X = df_cntrl[utils.get_featuredata(df_cntrl)]
    Y = df_cntrl[compound_col].tolist()
    return [X, Y]
