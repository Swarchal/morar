"""
a morar DataFrame, just like a pandas dataframe with a few useful extras
"""

import pandas as pd
from morar import utils
from morar import stats
from morar import normalise
from sklearn.decomposition import PCA


def to_morar_df(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return DataFrame(result)
    return wrapper


class DataFrame(pd.DataFrame):

    """
    morar.DataFrame inherits pandas dataframe with a few extra methods
    FIXME: any pandas method that returns a new object is a pandas.DataFrame
           rather than a morar.DataFrame
    """

    def __init__(self, data):
        pd.DataFrame.__init__(self, data)


    @property
    def featuredata(self):
        """return featuredata"""
        featuredata_cols = utils.get_featuredata(self)
        return DataFrame(self[featuredata_cols])


    @property
    def featurecols(self):
        """return of list feature data column names"""
        return utils.get_featuredata(self)


    @property
    def metadata(self):
        """return metadata"""
        metadata_cols = utils.get_metadata(self)
        return DataFrame(self[metadata_cols])


    @property
    def metacols(self):
        """return list of metadata column names"""
        return utils.get_metadata(self)


    def scale_features(self, **kwargs):
        """return dataframe of scaled feature data (via z-score)"""
        return DataFrame(stats.scale_features(self, **kwargs))


    def normalise(self, **kwargs):
        """normalise data via morar.normalise.normalise"""
        return DataFrame(normalise.normalise(self, **kwargs))


    def query(self, string, **kwargs):
        """pass query as in pd.DataFrame.query(string)"""
        pd_data = pd.DataFrame(self)
        result = pd_data.query(string, **kwargs)
        return DataFrame(result)


    def merge(self, right, **kwargs):
        """merge via pandas.DataFrame.merge"""
        pd_data = pd.DataFrame(self)
        result = pd_data.merge(right, *kwargs)
        return DataFrame(result)


    def dropna(self, **kwargs):
        """dropna via pandas.DataFrame.dropna"""
        pd_data = pd.DataFrame(self)
        result = pd_data.dropna(**kwargs)
        return DataFrame(result)


    def drop(self, **kwargs):
        """drop via pandas.DataFrame.drop"""
        pd_data = pd.DataFrame(self)
        result = pd_data.drop(**kwargs)
        return DataFrame(result)


    def pca(self, **kwargs):
        """
        return principal components morar.Dataframe and explained variance

        Returns:
        ---------
        [morar.DataFrame, array]
        morar.DataFrame with calculated principal components and metadata as
        the first element of the list.
        Also returns the explained variance of the principal components as
        calculated by `sklearn.decomposition.PCA.explained_variance_`.
        """
        pca = PCA(**kwargs)
        featuredata = self.featuredata
        metadata = self.metadata
        pca_out = pca.fit_transform(featuredata)
        pc_cols = ["PC" + str(i) for i in range(1, pca_out.shape[1]+1)]
        only_pc = pd.DataFrame(pca_out, columns=pc_cols, index=metadata.index)
        pca_df = DataFrame(pd.concat([only_pc, metadata], axis=1))
        return [pca_df, pca.explained_variance_]

