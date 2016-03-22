import pandas as pd
import numpy as np
import sqlalchemy as sql
import statistics as stats
import logging
from tqdm import tqdm

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
        engine = sql.create_engine("sqlite:///%s" % path)
        self.connection = engine.connect()
        # log setup
        logging.basicConfig(filename="RawData.log",
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s')
        # fetch table names
        self.table_names = engine.table_names()
        # convert to ascii table names
        self.table_names = [s.encode('ascii') for s in self.table_names]
        logging.info("Table names in database:  %s" % self.table_names)
        if  "Image" not in self.table_names:
            logging.error("'Image' table not found in database")
            ValueError("'Image' table not found in database")


    def aggregate_imagenumber(self, method = "median"):
        """
        Aggregates each table to a summary value per individual ImageNumber
        This should mean each table then has the same number of rows
        """
        logging.info("aggregated by ImageNumber via %s" % method)
        # Experiment table does not contain ImageNumber column
        not_experiment = [n for n in self.table_names if n != 'Experiment']
        for name in tqdm(not_experiment):
            tmp = pd.read_sql_table(name, self.connection)
            logging.debug("%s original row count: %i" % (name,len(tmp.index)))
            grouped = tmp.groupby(["ImageNumber"])
            if method == "median":
                out = grouped.aggregate(np.median)
            elif method == "mean":
                 out = grouped.aggregate(np.mean)
            else:
                ValueError("method has to be 'mean' or 'median'")
            logging.debug("%s aggregate row count: %i" %(name,len(out.index)))
            out.to_sql(name, self.connection, if_exists='replace')
            logging.debug("replaced %s with aggregate data" % name)

    # TODO
    def flag_bad_images(self, method = "hampel", **kwargs):
        """
        Identify bad or out-of-focus images from the Image table with
        ImageQuality columns.
        Return ImageNumber, can be passed to remove_images()
        """
        image = pd.read_sql_table("Image", self.connection,
                index_col = "ImageNumber")

        def subset_col(df, string):
            """
            Returns slice of dataframe, selecting only columns that
            contain 'string'
            """
            out = df[df.columns[df.columns.to_series().str.contains(string)]]
            return out

        df_iq = subset_col(image, "ImageQuality")
        logging.debug("called flag_bad_images with %s" % method)
        if method == "hampel":
            # hampel outlier test on imagequality metrics
            tmp = df_iq.apply(stats.hampel, axis = 0)
            outlier = tmp.abs().sum(axis = 1) # abs of hampel outliers
            lim = np.median(outlier) + 1.5 * stats.iqr(outlier)
            return list(outlier[outlier > lim].index.tolist())
        elif method == "focus":
            # beyond normal range for focus score (low is bad)
            tmp = subset_col(df_iq, "FocusScore_")
            outlier = tmp.apply(stats.u_iqr, axis = 1).sum(axis = 1)
            return list(outlier[outlier != 0].index.tolist())
        elif method == "plls":
            # beyond normal range for powerloglogslope metrics (high is bad)
            tmp = subset_col(df_iq, "PowerLogLogSlope_")
            outlier = tmp.apply(stats.o_iqr, axis = 1).sum(axis = 1)
            return list(outlier[outlier != 0].index.tolist())
        elif method == "correlation":
            # beyond normal range for correlation (high is bad)
            tmp = subset_col(df_iq, "Correlation_")
            outlier = tmp.apply(stats.o_iqr, axis = 1).sum(axis = 1)
            return list(outlier[outlier != 0].index.tolist())
        else:
            logging.error("Non-valid method of '%s' for flag_bad_image"%method)
            ValueError("Invalid method. Options: hampel, focus, plls")




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
            logging.error("%s is not a valid format")
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
        logging.info("Number of columns found: %i" % len(self.data.columns))
        logging.debug("Columns found:  %s" % list(self.data.columns))
        logging.debug("Number metadata columns found: %i" % len(self.metadata_cols))
        logging.debug("Metadata columns found: %s" % list(self.metadata_cols))
        logging.info("Number of feature data columns found: %i" % len(self.featuredata_cols))
        logging.debug("Featuredata columns found: %s" % list(self.featuredata_cols))

        
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
        logging.info("Selected only numeric featuredata")
        logging.debug("Numeric featuredata columns: %s" % list(self.featuredata_cols))
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



    def aggregate_well(self, method = "median"):
        """
        Aggregates values down to an image average
            - ??? modify in place or return new copy?
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
            (x - np.mean(x)) / np.std(x)
        # zscore numeric featuredata columns
        logging.info("Features scaled via z-score")
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
    test.get_no_variance_col()
    print "%i in full data" % len(test.data.index)
    test.aggregate_well()
    print "%i in agg data" % len(test.data.index)
    print test.scale_features()
    x = test.to_dataframe()
    print x.describe()

    x = RawData("/media/datastore_scott/Scott_1/db_test.sqlite")
    x.aggregate_imagenumber()
    print x.flag_bad_images(method = "hampel")
