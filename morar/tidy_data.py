import pandas as pd
import numpy as np
import sqlite3


class RawData:

    """
    Un-tidy data as it comes out of cell profiler stored from
    class results_directory
    Will be a database of a table per csv file:
        Image, Experiment, Objects (multiple)
    With truncated names.

    Functions to transform data into a useable format
    Then make into class tidy_data

    --------------------------------------------------------------------------

    Different datasets have differing number of rows
    Can combine/merge/concat data is same number of rows
    Need to aggregate by ImageNumber
    Then merge together
    Then aggregate by well
    Then pass to tidy_data()

    --------------------------------------------------------------------------
    """

    def __init__(self, path):
        # sqlite database of separate table for each .csv
        # connect to database
        conn = sqlite3.connect(path)
        c = conn.cursor()

        # fetch table names
        c.execute("SELECT name FROM sqlite_master WHERE type='table';")
        self.table_names = cursor.fetchall()
        
        if 'Image' not in self.table_names:
            print "Warning: 'Image' table not found in database"


    # TODO
    def aggregate_imagenumber(self, method = "median"):
        """
        Aggregates each table to a summary value per individual ImageNumber
        This should mean each table then has the same number of rows, and can 
        recursively merge all the tables by ImageNumber go get a single dataset
        """
        # pd.groupby().method()
            # where method is either mean or median
        pass


    # TODO
    def flag_bad_images(self, method = "hampel", **kwargs):
        """
        Identify bad or out-of-focus images from the Image table with
        ImageQuality columns. If no ImageQuality columns are found return
        a warning.
        Return ImageNumber, can be passed to remove_images()
        """
        # identify imageQuality columns in image table
        # method to identify bad images
            # hampel outlier
            # standard deviations
            # certain values over threshold
        pass


    # TODO
    def flag_errors(self):
        """
        Using ModuleError columns, flag images that produced an error.
        Return ImageNumber, can be passed to remove_images()
        """
        # identify error columns in image table
        # return ImageNumber which has sum > 0 for the error cols
        pass


    # TODO
    def remove_images(self, merged = True):
        """
        Given a list of ImageNumbers, this function will remove them from
        either all tables, or from a single merged tables
        """
        # match ImageNumber to row index
        # remove row index
        pass


    # TODO
    def merge_tables(self):
        """
        Merge all tables together by ImageNumber.
        The prefered method is merging the tables after they have been
        aggregated by imagenumber using aggregate_imagenumber()
        Return warning if tables have not been aggregated:
            - will have diff no. of rows per table
            - and duplicate rows for ImageNumber in some tables
        """
        pass


    # TODO
    def clean_cols(self, keep = False, merged = True):
        """
        Remove columns that are not featuredata or metadata.
        E.g file paths, ModuleError etc
        Either from merged table, or from individual tables
        Possibility of moving these columns into a separate table. As they may
        be useful later on. (argument 'keep')
        """
        pass


   









class TidyData:
    """
    Tidy data, i.e rows per observation, column per measurements
    with measurements from various objects together in a single table
    """


    def __init__(self, path, storage = "csv", metadata_prefix = "Metadata",
	    plate = "Plate", well = "Well", sep = "_"):
        
        # load data from csv or sqlite database
        try:
            if storage == "csv":
                self.data = pd.read_csv(path)
            if storage == "sqlite":
                self.db_con = sqlite3.connect(path)
                self.data = pd.read_sql(self.db_con)
        except ValueError:
            print "%s is not a valid format, try either 'csv' or 'sqlite'" % storage
        
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
            raise ValueError("%s not found in columns" % self.plate_col)
        if self.well_col not in self.data.columns:
            raise ValueError("%s not found in columns" % self.well_col)
  
        # get featuredata columns
        self.featuredata_cols = list(set(self.data.columns) - set(self.metadata_cols))
        assert len(list(self.featuredata_cols)) >= 1



    def get_numeric_featuredata(self, numeric_only = True):
        """
        Returns list of columns that correspond
        to feature data. i.e not metadata
        """
	# of those columns, get only the numeric columns
        tmp = self.data[self.featuredata_cols].select_dtypes(include=[np.number])
        self.featuredata_cols = tmp.columns
	# check there is at least one column
	assert len(list(self.featuredata_cols)) >= 1
        return list(self.featuredata_cols)



    def get_no_variance_col(self, tolerance = 1e-5):
        """
        Returns list of columns that have zero or very low variance.
	Low variance defined as less than `tolerance`, default = 1e-5
        """
        self.zero_var_cols = []
        for col in self.featuredata_cols:
            if np.var(self.data[col]) < tolerance:
                self.zero_var_cols.append(col)
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
	return self.na_rows



    def aggregate_well(self, method = "median"):
        """
        Aggregates values down to an image average
            - ??? modify in place or return new copy?
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



    # TODO make this
    def normalise_to_control(self, unique_plate_col, compound_col = "Compound", neg_compound = "DMSO"):
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
            return (x - np.mean(x)) / np.std(x)
        # zscore numeric featuredata columns
        self.data.loc[:, self.featuredata_cols].apply(zscore, axis = 0, reduce = False)


    def to_dataframe(self):
        """
        Return a pandas dataframe
        Allows option to use pandas functions
        """
        return pd.DataFrame(self.data)



    def pd_function(self, *args):
        """
        Apply pandas function to self.data without transforming to dataframe
        Need to keep the self.data as a TidyData class, otherwise cannot use
        other TidyData functions on it.
        If need be can transform back to a TidyData class without passing through
        __init__ again.
        """
        pass

if __name__ == "__main__":

    test = TidyData('/home/scott/Dropbox/Public/df_cell_subclass.csv')
    test.get_numeric_featuredata()
    print test.scale_features()
    x = test.to_dataframe()
    print x.describe()
