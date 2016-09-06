from morar import utils
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def find_correlation(df, threshold=0.9):
    """
    Given a numeric pd.DataFrame, this will find highly correlated features,
    and return a list of features to remove
    @param df pandas DataFrame
    @param threshold correlation threshold, will remove one of pairs of features
        with a correlation greater than this value
    @return list of column names to be removed
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

    @param df pandas DataFrame
    @oaram threshold low variance threshold
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(df, "is not a pandas DataFrame")
    var = df[utils.get_featuredata(df)].var(axis=0)
    below_thresh = var[var < threshold].index.tolist()
    is_nan = utils.is_all_nan(df)
    columns = list(below_thresh) + list(is_nan)
    return columns


def rf_controls_importances(df, neg_cmpd, pos_cmpd,
                            compound_col="Metadata_compound"):
    """
    Return features importances, based on separating the positive and negative
    controls in a random forest classifier.

    @param pandas DataFrame
    @param neg_cmpd string, name of negative control in compound_col
    @param pos_cmpd string, name of positive control in compound_col
    @param compound_col string, name of column in df that contains compound
                        labels
    @returns feature importances
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError(df, "is not a pandas DataFrame")
    if compound_col not in df.columns:
        raise ValueError(compound_col, "is not a column in", df)
    if neg_cmpd not in df[compound_col].tolist():
        raise ValueError(neg_cmpd, "is not in column", compound_col)
    if pos_cmpd not in df[compound_col].tolist():
        raise ValueError(pos_cmpd, "is not in column", compound_col)
    #split data into just positive and negative controls
    controls = [neg_cmpd, pos_cmpd]
    df_cntrl = df[df[compound_col].isin(controls)]
    # convert compound labels to integers. pos_cmpd=1, neg_cmpd=0
    cntrl_int = pd.Categorical(df_cntrl[compound_col]).codes.tolist()
    df_cntrl[compound_col] = cntrl_int
    # select just feature data
    X = df_cntrl[utils.get_featuredata(df_cntrl)]
    Y = df_cntrl[compound_col].tolist()
    # create classifier
    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X, Y)
    # extract feature importance from model
    importances = clf.feature_importances_
    col_names = X.columns.tolist()
    return zip(col_names, importances)
