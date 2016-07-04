import logging
import sqlite3
import pandas as pd
import numpy as np
#import sqlalchemy as sql
#import statistics as stats
#import dataframes as df
#from tqdm import tqdm



class TidyData:

    """
    Tidy data, i.e rows per observation, column per measurements
    with measurements from various objects together in a single table
    """


    def __init__(self, path, storage="csv", metadata_prefix="Metadata",
	                plate="Plate", well="Well", sep="_"):

        # load data from csv or sqlite database
        try:
            if storage == "csv":
                self.data = pd.read_csv(path)
            if storage == "sqlite":
                self.db_con = sqlite3.connect(path)
                self.data = pd.read_sql(self.db_con)
        except ValueError:
            logging.error("%s is not a valid format" %storage)
            print("{} is not a valid format, try either 'csv' or 'sqlite'".format(storage))

        # check it's actually returned a dataframe
        assert isinstance(self.data, pd.DataFrame)


        # get and define metadata columns
        self.metadata_prefix = metadata_prefix
        self.metadata_cols = []
        for col in self.data.columns:
            if self.metadata_prefix in col:
                self.metadata_cols.append(col)

        self.plate_col = self.metadata_prefix + sep + plate
        self.well_col = self.metadata_prefix + sep + well

        if self.plate_col not in self.data.columns:
            logging.error("%s not found in columns" % self.plate_col)
            raise ValueError("%s not found in columns" % self.plate_col)
        if self.well_col not in self.data.columns:
            logging.error("%s not found in columns" % self.well_col)
            raise ValueError("%s not found in columns" % self.well_col)

        # get featuredata columns
        self.featuredata_cols = list(set(self.data.columns) - set(self.metadata_cols))
        assert len(list(self.featuredata_cols)) >= 1

        # log setup
        logging.basicConfig(filename="TidyData.log",
                            level=logging.DEBUG,
                            format='%(asctime)s %(levelname)s: %(message)s')
        logging.info('Number of columns found:{}'.format(len(self.data.columns)))
        logging.debug("Columns found:{}".format(list(self.data.columns)))
        logging.debug("Number metadata columns found: %i" % len(self.metadata_cols))
        logging.debug("Metadata columns found: %s" % list(self.metadata_cols))
        logging.info("Number of feature data columns found: %i" % len(self.featuredata_cols))
        logging.debug("Featuredata columns found: %s" % list(self.featuredata_cols))


    def get_numeric_featuredata(self, numeric_only=True):
        """
        Returns list of columns that correspond
        to feature data. i.e not metadata
        """
	# of those columns, get only the numeric columns
        tmp = self.data[self.featuredata_cols].select_dtypes(include=[np.number])
        self.featuredata_cols = tmp.columns
	# check there is at least one column
	assert len(list(self.featuredata_cols)) >= 1
        logging.info("Selected only numeric featuredata")
        logging.debug("Numeric featuredata columns: %s" % list(self.featuredata_cols))
        return list(self.featuredata_cols)



    def get_no_variance_col(self, tolerance=1e-5):
        """
        Returns list of columns that have zero or very low variance.
	Low variance defined as less than `tolerance`, default = 1e-5
        """
        self.zero_var_cols = []
        for col in self.featuredata_cols:
            if np.var(self.data[col]) < tolerance:
                self.zero_var_cols.append(col)
        logging.debug("Columns of zero variance: %s" % self.zero_var_cols)
        logging.info("Number of zero variance columns: %i" % len(self.zero_var_cols))
        return self.zero_var_cols



    def get_all_NA_col(self):
        """"
        Returns list of columns that contain all NA values
        """
        for col in self.data.columns:
            if pd.isnull(col):
                return col


    def  get_any_NA_row(self):
        """"
        Returns row index of rows that contain any NA values
        """
        self.na_rows = self.data[self.data.isnull().any(axis=1)].index.tolist()
        logging.debug("Number of rows containing NA %i" % len(self.na_rows))
	return self.na_rows



    def aggregate_well(self, method="median"):
        """
        Aggregates values down to an image average
            - modifies data in place
        """
        grouped = self.data.groupby([self.plate_col, self.well_col])
        logging.info("Aggregated wells by %s" % method)
        if method == 'median':
            self.data = grouped.median()
        elif method == 'mean':
            self.data = grouped.mean()
        else:
            logging.error("None vald method for aggregate_well selected")
            raise ValueError("%s is not a valid method, try either 'median' or 'mean'" % method)
        logging.debug("Number of rows in aggregated data: %i" % len(self.data.index))



    # TODO make this
    def normalise_to_control(self, unique_plate_col, compound_col="Compound", neg_compound="DMSO"):
        """
        Normalises each featue against the median DMSO value for
        that feature per plate.
        Producing a z-score +/- SDs from the DMSO median
            - TODO modify in place of return new copy?
        """
        plate_grp = self.data.groupby(unique_plate_col)

        def neg_cntl_med():
            pass

        pass


    # TODO test this is working correctly with a test dataframe
    def scale_features(self):
        """
        z-score features, each feature scaled independent.
        """
        def zscore(x):
            (x - np.mean(x)) / np.std(x)
        # zscore numeric featuredata columns
        logging.info("Features scaled via z-score")
        self.data.loc[:, self.featuredata_cols].apply(zscore, axis=0, reduce=False)


    def to_dataframe(self):
        """
        Return a pandas dataframe
        Allows option to use pandas functions
        """
        return pd.DataFrame(self.data)



if __name__ == "__main__":

    Xtest = TidyData('/home/scott/Dropbox/Public/df_cell_subclass.csv')
    Xtest.get_numeric_featuredata()
    Xtest.get_no_variance_col()
    print "%i in full data" % len(Xtest.data.index)
    Xtest.aggregate_well()
    print "%i in agg data" % len(Xtest.data.index)
    print Xtest.scale_features()
    print Xtest.to_dataframe()
