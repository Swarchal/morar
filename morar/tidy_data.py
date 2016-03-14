import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3


class raw_data:

    """
    Un-tidy data as it comes out of cell profiler stored from
    class results_directory
    Functions to transform data into a useable format
    Then make into class tidy_data
    """

    pass
        


class tidy_data:
    """
    Tidy data, i.e rows per observation, column per measurements
    with measurements from various objects together in a single table
    """


    def __init__(self, path, storage = "csv"):
        try:
            if storage == "csv":
                self.data = pd.read_csv(path)
            if storage == "sqlite":
                self.db_con = sqlite3.connect(path)
                self.data = pd.read_sql(self.db_con)
        except ValueError:
            print "%s is not a valid format, try either 'csv' or 'sqlite'" % storage
        assert isinstance(self.data, pd.DataFrame)


    
    def get_metadata(self, metadata_prefix = "Metadata_", name = True):
        """
        Returns list of columns that correspond 
        to metadata.
        """
        self.metadata_prefix = metadata_prefix
        self.metadata_cols = []
        if name:
            for col in self.data.columns:
                if self.metadata_prefix in col:
                    self.metadata_cols.append(col)
        elif name == False:
            pass # TODO return col index


    # TODO: how to handle site repeats? 
    def define_metadata(self, plate = "Plate", well = "Well", sep = "_"):
	"""
	Defines specific metadata columns needed for aggregating
	and normalising functions
	- Plate ID
	- Well ID

	The sep argument is the character used to separate the plate/well string
	from the metadata_prefix that has already been defined in 'get_metadata'
	e.g Metadata_Plate, Metadata_Well
	"""

	# TODO check metadata has already been defined

        self.plate_col = self.metadata_prefix + sep + plate
        self.well_col = self.metadata_prefix + sep + well

        if self.plate_col not in self.data.columns:
            raise ValueError('%s was not found in available columns' % self.plate_col)
        if self.well_col not in self.data.columns:
            raise ValueError('%s was not found in available columns' %self.well_col)


    def get_featuredata(self, name = True, numeric_only = True):
        """
        Returns list of columns that correspond
        to feature data. i.e not metadata
        """
	# get columns that are not metadata
        self.featuredata_cols = list(set(self.data.columns) - set(self.metadata_cols))
	# check there is at least one column
	assert len(list(self.featuredata_cols)) >= 1
        if name:
            if numeric_only:
		# of those columns, get only the numeric columns
                tmp = self.data[self.featuredata_cols].select_dtypes(include=[np.number])
                self.featuredata_cols = tmp.columns
		# check there is at least one column
		assert len(list(self.featuredata_cols)) >= 1
            return list(self.featuredata_cols)
        elif name == False:
            pass # TODO return col index



    def get_no_variance_col(self, name = True, tolerance = 1e-5):
        """
        Returns list of columns that have zero or very low variance.
	Low variance defined as less than `tolerance`, default = 1e-5
        """
        self.zero_var_cols = []
        if name == True:
            for col in self.featuredata_cols:
                if np.var(self.data[col]) < tolerance:
                    self.zero_var_cols.append(col)
        elif name == False:
            pass # TODO return col index

        return self.zero_var_cols



    def get_all_NA_col(self, name = True):
        """"
        Returns list of columns that contain all NA values
        """
        def all_null(x):
            'returns true of column contains all null values'
            return pd.isnull(x)

        if name == True:
            for col in self.data.columns:
                if all_null(col):
                    return col
        elif name == False:
            pass # TODO return col index


    def  get_any_NA_row(self):
        """"
        Returns row index of rows that contain any NA values
        """
        self.na_rows = self.data[self.data.isnull().any(axis=1)].index.tolist()
	return self.na_rows


    # TODO
    def aggregate_well(self, method = "median"):
        """
        Aggregates values down to an image average
            - TODO modify in place or return new copy?
        """
	# pandas group by and median
	# have to define the metadata columns that are needed
        # e.g plate, well, site
        # can do this in define_metadata()
        grouped = self.data.groupby([self.plate_col, self.well_col])
        if method == 'median':
            out = grouped.median()
        elif method == 'mean':
            out = grouped.mean()
        else:
            raise ValueError("%s is not a valid method, try either 'median' or 'mean'" % method)
        return out


    # TODO
    def normalise_to_control(self):
        """
        Normalises each featue against the median DMSO value for
        that feature per plate.
        Producing a z-score +/- SDs from the DMSO median
            - TODO modify in place of return new copy?
        """
        pass


if __name__ == "__main__":

    test = tidy_data('/home/scott/Dropbox/Public/df_cell_subclass.csv')

    test.get_metadata("Metadata")
    test.get_featuredata()
    test.define_metadata()
    print test.plate_col
    print test.well_col
    print test.aggregate_well()
