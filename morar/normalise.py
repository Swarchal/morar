import stats
import pandas as pd
import numpy as np


class Normalise():
    """
    Normalisation functions for pandas dataframes.
    """


    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("input is not a pandas DataFrame")
        self.df = data


    def get_featuredata(self, df, metadata_prefix="Metadata"):
        """
        identifies columns in a dataframe that are not labelled with the
        metadata prefix. Its assumed everything not labelled metadata is
        featuredata
        -----------------------------------------------------------------------
        metadata_prefix: string, prefix for metadata columns
        """
        f_cols = [i for i in self.df.columns if not i.startswith(metadata_prefix)]
        return f_cols


    def check_control(self, df, plate_id, compound="Metadata_compound",
                      neg_compound="DMSO"):
        """ check each plate contains at least one negative control """
        pass


    def normalise(self, plate_id, compound="Metadata_compound",
                  neg_compound="DMSO", method="divide",
                  metadata_prefix="Metadata"):
        """
        Normalise values against DMSO values per plate.
        -----------------------------------------------------------------------
        plate_id: string for column containing plate ID/label
        compound: string for column containing compound name/ID
        neg_compound: string name of negative control compound in compound col
        method: method to normalise against negative control
        metadata_prefix: string, prefix for metadata columns
        """
        valid_methods = ["subtract", "divide"]
        if method not in valid_methods:
            raise ValueError("Invalid method, options: 'subtract', 'divide'")
        # check there are some negative controls on each plate
        self.check_control(self.df, plate_id, compound, neg_compound)
        # identify feature columns
        f_cols = self.get_featuredata(self.df, metadata_prefix)
        df_out = pd.DataFrame()
        # group by plate
        grouped = self.df.groupby(plate_id, as_index=False)
        # calculate the average DMSO values for each plate
        for _, group in grouped:
            # TODO keep metadata columns
            dmso_vals = list(group[group[compound] == neg_compound][f_cols].median().values)
            assert len(dmso_vals) == group[f_cols].shape[1]
            if method == "subtract":
                # TODO fix subtraction calculation
                tmp = group[f_cols].apply(lambda x: x - dmso_vals)
            if method == "divide":
                tmp = group[f_cols].divide(dmso_vals)
            df_out = pd.concat([df_out, tmp])
        # check we have not lost any rows
        assert self.df.shape[1] == df_out.shape[1]
        return df_out


    def robust_normalise(self, plate_id, compound="Metadata_compound",
                         neg_compound="DMSO", metadata_prefix="Metadata"):
        """
        Method used in the Carpenter lab. Substract the median feature value for
        each plate from the treatment feature value and divide by the median
        absolute deviation.
        Returns a pandas DataFrame.
        -----------------------------------------------------------------------
        plate_id: string for column containing plate ID/label
        compound: string for column containing compound name/ID
        neg_compound: string name of negative control compound in compound col
        metadata_prefix: string, prefix for metadata columns
        """
        pass


    def scale_features(self):
        """
        scale and centre features
        """
        pass
