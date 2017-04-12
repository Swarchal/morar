"""
a morar DataFrame, just like a pandas dataframe with a few useful extras
"""

import pandas as pd
from morar import utils
from morar import stats
from morar import normalise
from sklearn.decomposition import PCA

class DataFrame(pd.DataFrame):

    """
    morar.DataFrame inherits pandas dataframe with a few extra methods
    FIXME: any pandas method that returns a new object is a pandas.DataFrame
           rather than a morar.DataFrame
    """

    def __init__(self, data):
        pd.DataFrame.__init__(self, data)

    @property
    def featuredata(self, **kwargs):
        """return featuredata"""
        featuredata_cols = utils.get_featuredata(self, **kwargs)
        return self[featuredata_cols]

    @property
    def featurecols(self, **kwargs):
        """return of list feature data column names"""
        return utils.get_featuredata(self, **kwargs)

    @property
    def metadata(self, **kwargs):
        """return metadata"""
        metadata_cols = utils.get_metadata(self, **kwargs)
        return self[metadata_cols]

    @property
    def metacols(self, **kwargs):
        """return list of metadata column names"""
        return utils.get_metadata(self, **kwargs)


    def scale_features(self, **kwargs):
        """return dataframe of scaled feature data (via z-score)"""
        return stats.scale_features(self, **kwargs)


    def normalise(self, **kwargs):
        """normalise data via morar.normalise.normalise"""
        return normalise.normalise(self, **kwargs)


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


    def pca(self, var=True, **kwargs):
        """
        return principal components morar.Dataframe and explained variance

        Parameters:
        -----------
        var : bool (default=True)
            whether or not to return the explained variance of the components

        Returns:
        ---------
        morar.DataFrame with calculated principal components and metadata
        If var=True (default), also returns a list, where the second element is
        the exaplained variance of the principal components as calculated by
        `sklearn.decomposition.PCA.explained_variance_`
        """
        pca = PCA(**kwargs)
        featuredata = self.get_featuredata(**kwargs)
        metadata = self.get_metadata(**kwargs)
        pca_out = pca.fit_transform(featuredata)
        pc_cols = ["PC" + str(i) for i in range(1, pca_out.shape[1]+1)]
        only_pc = pd.DataFrame(pca_out, columns=pc_cols, index=metadata.index)
        pca_df = DataFrame(pd.concat([only_pc, metadata], axis=1))
        if var is True:
            return [pca_df, pca.explained_variance_]
        elif var is False:
            return pca_df
        else:
            msg = "expected bool for variance, received {}".format(type(var))
            raise ValueError(msg)

