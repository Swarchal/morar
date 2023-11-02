"""
a morar DataFrame, just like a pandas dataframe with a few useful extras
"""
from typing import Self

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from morar import normalise, positional_correction, stats, utils
from morar.aggregate import aggregate


class DataFrame(pd.DataFrame):

    """
    morar.DataFrame inherits pandas dataframe with a few extra methods
    """

    _metadata = ["metadata_string", "prefix"]
    _internal_names = pd.DataFrame._internal_names + ["metadata_string", "prefix"]
    _internal_names_set = set(_internal_names)

    metadata_string = "Metadata_"
    prefix = True

    def __init__(
        self, *args, metadata_string: str = "Metadata_", prefix: bool = True, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.metadata_string = metadata_string
        self.prefix = prefix

    @property
    def _constructor(self):
        return DataFrame

    @property
    def featuredata(self) -> Self | pd.Series:
        """return featuredata"""
        featuredata_cols = utils.get_featuredata(
            self, self.metadata_string, self.prefix
        )
        return self[featuredata_cols]

    @property
    def featurecols(self) -> list[str]:
        """return of list feature data column names"""
        return utils.get_featuredata(self, self.metadata_string, self.prefix)

    @property
    def metadata(self) -> Self | pd.Series:
        """return metadata"""
        metadata_cols = utils.get_metadata(self, self.metadata_string, self.prefix)
        return self[metadata_cols]

    @property
    def metacols(self) -> list[str]:
        """return list of metadata column names"""
        return utils.get_metadata(self, self.metadata_string, self.prefix)

    def scale_features(self) -> Self:
        """return dataframe of scaled feature data (via z-score)"""
        df = stats.scale_features(
            self, metadata_string=self.metadata_string, prefix=self.prefix
        )
        return DataFrame(df, metadata_string=self.metadata_string, prefix=self.prefix)

    def agg(self, **kwargs) -> Self:
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
            self, metadata_string=self.metadata_string, prefix=self.prefix, **kwargs
        )
        return DataFrame(df, metadata_string=self.metadata_string, prefix=self.prefix)

    def pca(self, **kwargs) -> tuple[Self, np.ndarray]:
        """
        return principal components morar.Dataframe and explained variance

        Parameters:
        ----------
        **kwargs: additional arguments to sklearn.decomposition.PCA()

        Returns:
        ---------
        [morar.DataFrame, array]
        morar.DataFrame with calculated principal components and metadata as
        the first element of the list.
        Also returns the explained variance of the principal components as
        calculated by `sklearn.decomposition.PCA().explained_variance_`.
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
        return pca_df, pca.explained_variance_

    def umap(self, **kwargs) -> Self:
        """
        umap dimensional embededing

        Parameters:
        ----------
        **kwargs: additional arguments to umap.UMAP()

        Returns:
        ---------
        morar.DataFrame
        morar.DataFrame with UMAP dimensions in place to featuredata.
        """
        try:
            from umap import UMAP
        except ImportError:
            raise ImportError("Need to install umap-learn to run umap")
        umap = UMAP(**kwargs)
        featuredata = self.featuredata
        metadata = self.metadata
        umap_out = umap.fit_transform(featuredata)
        umap_df = pd.DataFrame(
            umap_out,
            columns=[f"U{i+1}" for i in range(umap_out.shape[1])],
            index=metadata.index,
        )
        return DataFrame(
            pd.concat([umap_df, metadata], axis=1),
            metadata_string=self.metadata_string,
            predix=self.prefix,
        )

    def impute(self, method: str = "median", **kwargs) -> Self:
        """
        Impute missing values by using the feature average.

        Parameters:
        ----------
        method: string (default = "median")
            method with which to calculate the feature average.
            Options = ("mean", "median")
        **kwargs:
            additional methods to be passed to
            sklearn.preprocessing.Imputer

        Returns:
        --------
        morar.DataFrame
        A dataframe with imputed missing values
        """
        imputed_df = utils.impute(self, method, **kwargs)
        return DataFrame(
            imputed_df,
            metadata_string=self.metadata_string,
            prefix=self.metadata_prefix,
        )

    def whiten(self, centre=True, method="ZCA") -> Self:
        """
        Whiten / spherize feature data y ZCA (aka Mahalanobis whitening).
        Removes linear correlation across features.

        Returns:
        --------
        morar.DataFrame
        same dimensions as input, but feature columns have been whitened.
        """
        df_whitened = normalise.whiten(
            self.data,
            centre=centre,
            method=method,
            metadata_string=self.metadata_string,
            prefix=self.predix,
        )
        return DataFrame(df_whitened, self.metadata_string, self.prefix)

    def median_polish(self, well_col: str, plate_col: str) -> Self:
        """2-way median polish on feature data."""
        df_smoothed = positional_correction.median_smooth_df(
            self.data, fcols=self.featurecols, plate_id_col=plate_col, well_col=well_col
        )
        df_copy = self.data.copy()
        df_copy = df_copy.drop(columns=self.featurecols)
        df_merged = pd.merge(df_copy, df_smoothed, on=[well_col, plate_col], how="left")
        return DataFrame(df_merged, self.metadata_string, self.prefix)
