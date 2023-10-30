"""
a morar DataFrame, just like a pandas dataframe with a few useful extras
"""

import pandas as pd
from sklearn.decomposition import PCA

from morar import normalise, stats, utils
from morar.aggregate import aggregate


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

    def __init__(self, data, metadata_string="Metadata_", prefix=True):
        pd.DataFrame.__init__(self, data)
        self.metadata_string = metadata_string
        self.prefix = prefix

    @property
    def _constructor(self):
        return DataFrame

    @property
    def featuredata(self):
        """return featuredata"""
        featuredata_cols = utils.get_featuredata(
            self, self.metadata_string, self.prefix
        )
        return DataFrame(
            self[featuredata_cols],
            metadata_string=self.metadata_string,
            prefix=self.prefix,
        )

    @property
    def featurecols(self):
        """return of list feature data column names"""
        return utils.get_featuredata(self, self.metadata_string, self.prefix)

    @property
    def metadata(self):
        """return metadata"""
        metadata_cols = utils.get_metadata(self, self.metadata_string, self.prefix)
        return DataFrame(
            self[metadata_cols],
            metadata_string=self.metadata_string,
            prefix=self.prefix,
        )

    @property
    def metacols(self):
        """return list of metadata column names"""
        return utils.get_metadata(self, self.metadata_string, self.prefix)

    def scale_features(self):
        """return dataframe of scaled feature data (via z-score)"""
        df = stats.scale_features(
            self, metadata_string=self.metadata_string, prefix=self.prefix
        )
        return DataFrame(df, metadata_string=self.metadata_string, prefix=self.prefix)

    def agg(self, **kwargs):
        """return aggregated dataframe via morar.aggregate.aggregate"""
        agg_df = aggregate(
            self, metadata_string=self.metadata_string, prefix=self.prefix, **kwargs
        )
        return DataFrame(
            agg_df, metadata_string=self.metadata_string, prefix=self.prefix
        )

    def normalise(self, **kwargs):
        """normalise data via morar.normalise.normalise"""
        df = normalise.normalise(
            self, metadata_string=self.metadata_string, prefix=self.prefix**kwargs
        )
        return DataFrame(
            df, metadata_string=self.metadata_string, prefix=self.metadata_prefix
        )

    def query(self, string, **kwargs):
        """pass query as in pd.DataFrame.query(string)"""
        pd_data = pd.DataFrame(self)
        result = pd_data.query(string, **kwargs)
        return DataFrame(
            result, metadata_string=self.metadata_string, prefix=self.prefix
        )

    def merge(self, right, **kwargs):
        """merge via pandas.DataFrame.merge"""
        pd_data = pd.DataFrame(self)
        result = pd_data.merge(right, **kwargs)
        return DataFrame(
            result, metadata_string=self.metadata_string, prefix=self.prefix
        )

    def dropna(self, **kwargs):
        """dropna via pandas.DataFrame.dropna"""
        _check_inplace(**kwargs)
        pandas_df = pd.DataFrame(self)
        result = pandas_df.dropna(**kwargs)
        return DataFrame(
            result, metadata_string=self.metadata_string, prefix=self.prefix
        )

    def drop(self, label, **kwargs):
        """drop via pandas.DataFrame.drop"""
        _check_inplace(**kwargs)
        pandas_df = pd.DataFrame(self)
        result = pandas_df.drop(label, **kwargs)
        return DataFrame(
            result, metadata_string=self.metadata_string, prefix=self.prefix
        )

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
        pc_cols = ["PC" + str(i) for i in range(1, pca_out.shape[1] + 1)]
        only_pc = pd.DataFrame(pca_out, columns=pc_cols, index=metadata.index)
        pca_df = DataFrame(
            pd.concat([only_pc, metadata], axis=1),
            metadata_string=self.metadata_string,
            prefix=self.prefix,
        )
        return [pca_df, pca.explained_variance_]

    def impute(self, method="median", **kwargs):
        """
        Impute missing values by using the feature average.

        Paramters:
        ----------
        method: string (default = "median")
            method with which to calculate the feature average.
            Options = ("mean", "median")
        **kwargs:
            additional methods to be passed to
            sklearn.preprocessing.Imputer

        Returns:
        --------
        pandas.DataFrame
        A dataframe with imputed missing values
        """
        imputed_df = utils.impute(self, method, **kwargs)
        return DataFrame(
            imputed_df,
            metadata_string=self.metadata_string,
            prefix=self.metadata_prefix,
        )


def _check_inplace(**kwargs):
    if "inplace" in kwargs:
        msg = "inplace modifications do not work with morar.DataFrames"
        raise NotImplementedError(msg)
