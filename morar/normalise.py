import multiprocessing
from typing import Callable

import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from pandas.core.groupby.generic import DataFrameGroupBy

from morar import stats, utils

# stop copy warning as not using chained assignment
# pd.options.mode.chained_assignment = None  # default='warn'


def _check_control(
    data: pd.DataFrame,
    plate_id: str,
    compound: str = "Metadata_compound",
    neg_compound: str = "DMSO",
) -> None:
    """
    check each plate contains at least 1 negative control value. Raise an error
    if this is not the case.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    plate_id : string
        column containing plate ID/label
    compound : string (default="Metadata_compound")
        column containing compound name/ID
    neg_compound : string (default="DMSO")
        name of negative control compound in compound col
    """
    for name, group in data.groupby(plate_id):
        group_cmps = group[compound].unique()
        if neg_compound not in group_cmps:
            raise RuntimeError(f"{name} does not contain any negative control values")


def robust_normalise(
    data: pd.DataFrame,
    plate_id: str,
    compound: str = "Metadata_compound",
    neg_compound: str = "DMSO",
    metadata_string: str = "Metadata_",
    prefix: bool = True,
) -> pd.DataFrame:
    """
    Method used in the Carpenter lab. Substract the median feature value for
    each plate negative control from the treatment feature value and divide by
    the median absolute deviation.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    plate_id : string
        column containing plate ID/label
    compound : string (default="Metadata_compound")
        column containing compound name/ID
    neg_compound : string (default="DMSO")
        name of negative control compound in compound col
    **kwargs : additional arguments to utils.get_featuredata/metadata

    Returns
    --------
    df_out : pandas DataFrame
        DataFrame of normalised feature values
    """
    _check_control(data, plate_id, compound, neg_compound)
    f_cols = utils.get_featuredata(data, metadata_string, prefix)
    grouped = data.groupby(plate_id, as_index=False)
    df_out = pd.DataFrame()
    # calculate the average negative control values per plate_id
    for _, group in grouped:
        # find the median and mad dmso value for each plate
        dmso_vals = group[group[compound] == neg_compound]
        dmso_med = dmso_vals[f_cols].median().values
        dmso_mad = dmso_vals[f_cols].apply(stats.mad, axis=0).values
        assert len(dmso_med) == group[f_cols].shape[1]
        # subtract each row of the group by that group's DMSO values
        group[f_cols] = group[f_cols].sub(dmso_med)
        # divide by the MAD of the negative control
        group[f_cols] = group[f_cols].apply(lambda x: (x / dmso_mad) * 1.4826, axis=1)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, group])
    # check we have not lost any rows
    assert data.shape == df_out.shape
    return df_out


def normalise(
    data: pd.DataFrame, plate_id: str, parallel: bool = False, **kwargs
) -> pd.DataFrame:
    """
    Normalise values against negative controls values per plate.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    plate_id : string
        column containing plate ID/label
    compound : string (default="Metadata_compound")
        column containing compound name/ID
    neg_compound : string (default="DMSO")
        name of negative control compound in compound col
    method :string (default="subtract")
        method to normalise against negative control
    **kwargs : additional arguments to utils.get_featuredata/metadata

    Returns
    --------
    df_out : pandas DataFrame
        DataFrame of normalised feature values
    """
    if parallel:
        return p_normalise(data, plate_id, **kwargs)
    else:
        return s_normalise(data, plate_id, **kwargs)


def s_normalise(
    data: pd.DataFrame,
    plate_id: str,
    compound: str = "Metadata_compound",
    neg_compound: str = "DMSO",
    method: str = "subtract",
    metadata_string: str = "Metadata_",
    prefix: bool = True,
) -> pd.DataFrame:
    """
    Normalise values against negative controls values per plate.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame
    plate_id : string
        column containing plate ID/label
    compound : string (default="Metadata_compound")
        column containing compound name/ID
    neg_compound : string (default="DMSO")
        name of negative control compound in compound col
    method :string (default="subtract")
        method to normalise against negative control
    **kwargs : additional arguments to utils.get_featuredata/metadata

    Returns
    --------
    df_out : pandas DataFrame
        DataFrame of normalised feature values
    """
    valid_methods = ["subtract", "divide"]
    if method not in valid_methods:
        raise ValueError(f"Invalid method, options: {valid_methods}")
    # check there are some negative controls on each plate
    _check_control(data, plate_id, compound, neg_compound)
    # identify feature columns
    f_cols = utils.get_featuredata(data, metadata_string, prefix)
    # dataframe for output
    df_out = pd.DataFrame()
    # group by plate
    grouped = data.groupby(plate_id, as_index=False)
    # calculate the average negative control values for each plate
    for _, group in grouped:
        dmso_med_ = group[group[compound] == neg_compound]
        dmso_med = dmso_med_[f_cols].median()
        if method == "subtract":
            group[f_cols] = group[f_cols].sub(dmso_med)
        if method == "divide":
            group[f_cols] = group[f_cols].divide(dmso_med)
        # concatenate group to overall dataframe
        df_out = pd.concat([df_out, group])
    # check we have not lost any rows
    assert data.shape == df_out.shape
    return df_out


def _norm_group(
    group: pd.DataFrame,
    neg_compound: str,
    compound: str,
    f_cols: list[str],
    method: str,
) -> pd.DataFrame:
    """normalisation funcion for use with p_normalise"""
    dmso_med = group[group[compound] == neg_compound][f_cols].median()
    copy = group.copy()
    if method == "subtraction":
        copy[f_cols] = copy[f_cols].sub(dmso_med)
    elif method == "division":
        copy[f_cols] = copy[f_cols].div(dmso_med)
    else:
        raise ValueError(f"{method} not a valid method")
    return copy


def _apply_parallel(
    grouped_df: DataFrameGroupBy,
    func: Callable,
    neg_compound: str,
    compound: str,
    f_cols: list[str],
    n_jobs: int,
    method: str,
) -> pd.DataFrame:
    """internal parallel gubbins for p_normalise"""
    output = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, neg_compound, compound, f_cols, method)
        for _, group in grouped_df
    )
    if output is None:
        output = []
    return pd.concat(output)


def p_normalise(
    data: pd.DataFrame,
    plate_id: str,
    compound: str = "Metadata_compound",
    neg_compound: str = "DMSO",
    n_jobs: int = -1,
    method: str = "subtraction",
    metadata_string: str = "Metadata_",
    prefix: bool = True,
) -> pd.DataFrame:
    """
    parallelised version of normalise, currently only works with subtraction
    normalisation.
    """
    _check_control(data, plate_id, compound, neg_compound)
    if n_jobs < 0:
        # use all available cpu cores
        n_jobs = multiprocessing.cpu_count()
    f_cols = utils.get_featuredata(data, metadata_string, prefix)
    grouped = data.groupby(plate_id, as_index=False)
    return _apply_parallel(
        grouped_df=grouped,
        func=_norm_group,
        neg_compound=neg_compound,
        compound=compound,
        f_cols=f_cols,
        n_jobs=n_jobs,
        method=method,
    )


def whiten(
    data: pd.DataFrame,
    centre=True,
    method="ZCA",
    metadata_string: str = "Metadata_",
    prefix: bool = True,
) -> pd.DataFrame:
    """Whiten/spherize the feature data"""
    Whitener = Spherize(center=centre, method=method)
    df_copy = data.copy()
    fcols = utils.get_featuredata(data, metadata_string, prefix)
    fdata_whitened = Whitener.fit_transform(df_copy[fcols])
    fdata_df = pd.DataFrame(fdata_whitened, columns=fcols, index=data.index)
    # merge with metadata
    df_copy[fcols] = fdata_df
    return df_copy


# BSD 3-Clause License
# Copyright (c) 2019, Broad Institute of MIT and Harvard All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    - Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#    - Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#    - Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
class Spherize:
    """Class to apply a sphering transform (aka whitening) data in the base sklearn
    transform API. Note, this implementation is modified/inspired from the following
    sources:
    1) A custom function written by Juan C. Caicedo
    2) A custom ZCA function at https://github.com/mwv/zca
    3) Notes from Niranj Chandrasekaran (https://github.com/cytomining/pycytominer/issues/90)
    4) The R package "whitening" written by Strimmer et al (http://strimmerlab.org/software/whitening/)
    5) Kessy et al. 2016 "Optimal Whitening and Decorrelation" [1]_

    Attributes
    ----------
    epsilon : float
        fudge factor parameter
    center : bool
        option to center the input X matrix
    method : str
        a string indicating which class of sphering to perform
    """

    def __init__(self, epsilon=1e-6, center=True, method="ZCA"):
        """
        Parameters
        ----------
        epsilon : float, default 1e-6
            fudge factor parameter
        center : bool, default True
            option to center the input X matrix
        method : str, default "ZCA"
            a string indicating which class of sphering to perform
        """
        avail_methods = ["PCA", "ZCA", "PCA-cor", "ZCA-cor"]
        self.epsilon = epsilon
        self.center = center
        assert (
            method in avail_methods
        ), f"Error {method} not supported. Select one of {avail_methods}"
        self.method = method

    def fit(self, X, y=None):
        """Identify the sphering transform given self.X

        Parameters
        ----------
        X : pandas.core.frame.DataFrame
            dataframe to fit sphering transform

        Returns
        -------
        self
            With computed weights attribute
        """
        # Get the mean of the features (columns) and center if specified
        self.mu = X.mean()
        if self.center:
            X = X - self.mu
        # Get the covariance matrix
        C = (1 / X.shape[0]) * np.dot(X.transpose(), X)
        if self.method in ["PCA", "ZCA"]:
            # Get the eigenvalues and eigenvectors of the covariance matrix
            s, U = scipy.linalg.eigh(C)
            # Fix sign ambiguity of eigenvectors
            U = pd.DataFrame(U * np.sign(np.diag(U)))
            # Process the eigenvalues into a diagonal matrix and fix rounding errors
            D = np.diag(1.0 / np.sqrt(s.clip(self.epsilon)))
            # Calculate the sphering matrix
            self.W = np.dot(D, U.transpose())
            # If ZCA, perform additional rotation
            if self.method == "ZCA":
                self.W = np.dot(U, self.W)

        if self.method in ["PCA-cor", "ZCA-cor"]:
            # Get the correlation matrix
            R = np.corrcoef(X.transpose())
            # Get the eigenvalues and eigenvectors of the correlation matrix
            try:
                t, G = scipy.linalg.eigh(R)
            except ValueError:
                raise ValueError(
                    "Divide by zero error, make sure low variance columns are removed"
                )
            # Fix sign ambiguity of eigenvectors
            G = pd.DataFrame(G * np.sign(np.diag(G)))
            # Process the eigenvalues into a diagonal matrix and fix rounding errors
            D = np.diag(1.0 / np.sqrt(t.clip(self.epsilon)))
            # process the covariance diagonal matrix and fix rounding errors
            v = np.diag(1.0 / np.sqrt(np.diag(C).clip(self.epsilon)))
            # Calculate the sphering matrix
            self.W = np.dot(np.dot(D, G.transpose()), v)
            # If ZCA-cor, perform additional rotation
            if self.method == "ZCA-cor":
                self.W = np.dot(G, self.W)
        return self

    def transform(self, X, y=None):
        """Perform the sphering transform

        Parameters
        ----------
        X : pd.core.frame.DataFrame
            Profile dataframe to be transformed using the precompiled weights
        y : None
            Has no effect; only used for consistency in sklearn transform API

        Returns
        -------
        pandas.core.frame.DataFrame
            Spherized dataframe
        """
        return np.dot(X - self.mu, self.W.transpose())

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)
