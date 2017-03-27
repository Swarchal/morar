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
    def get_featuredata(self, **kwargs):
        """return featuredata"""
        featuredata_cols = utils.get_featuredata(self, **kwargs)
        return self[featuredata_cols]

    @property
    def get_featurecols(self, **kwargs):
        """return of list feature data column names"""
        return utils.get_featuredata(self, **kwargs)

    @property
    def get_metadata(self, **kwargs):
        """return metadata"""
        metadata_cols = utils.get_metadata(self, **kwargs)
        return self[metadata_cols]

    @property
    def get_metacols(self, **kwargs):
        """return list of metadata column names"""
        return utils.get_metadata(self, **kwargs)


    def scale_features(self, **kwargs):
        """return dataframe of scaled feature data (via z-score)"""
        return stats.scale_features(self, **kwargs)


    def normalise(self, **kwargs):
        return normalise.normalise(self, **kwargs)


    def pca(self, **kwargs):
        """return principal components dataframe and explained variance"""
        pca = PCA(**kwargs)
        featuredata = self.get_featuredata(**kwargs)
        metadata = self.get_metadata(**kwargs)
        pca_out = pca.fit_transform(featuredata)
        pc_cols = ["PC" + str(i) for i in range(1, pca_out.shape[1]+1)]
        only_pc = pd.DataFrame(pca_out, columns=pc_cols, index=metadata.index)
        pca_df = pd.concat([only_pc, metadata], axis=1)
        return [pca_df, pca.explained_variance_]

