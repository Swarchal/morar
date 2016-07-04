import pandas as pd
import numpy as np
import sqlalchemy as sql
import statistics as stats
import dataframes as df
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
        logging.info("Table names in database: %s" % self.table_names)
        if  "Image" not in self.table_names:
            logging.error("'Image' table not found in database")
            ValueError("'Image' table not found in database")


    def aggregate_imagenumber(self, method="median"):
        """
        Aggregates each table to a summary value per individual ImageNumber
        This should mean each table then has the same number of rows
        """
        logging.info("aggregated by ImageNumber via {}".format(method))
        # Experiment table does not contain ImageNumber column
        not_experiment = [n for n in self.table_names if n != 'Experiment']
        for name in tqdm(not_experiment):
            tmp = pd.read_sql_table(name, self.connection)
            grouped = tmp.groupby(["ImageNumber"])
            if method == "median":
                out = grouped.aggregate(np.median)
            elif method == "mean":
                out = grouped.aggregate(np.mean)
            else:
                ValueError("method has to be 'mean' or 'median'")
            logging.debug("%s aggregate row count: %i" %(name, len(out.index)))
            out.to_sql(name, self.connection, if_exists='replace')
            logging.debug("replaced %s with aggregate data" % name)

    # TODO arguments to pass to hampel, i.e sigma
    def flag_bad_images(self, method="hampel", **kwargs):
        """
        Identify bad or out-of-focus images from the Image table with
        ImageQuality columns.
        Return ImageNumber, can be passed to remove_images()
        """
        image = pd.read_sql_table("Image", self.connection,
                                  index_col="ImageNumber")

        df_iq = df.subset_col(image, "ImageQuality")
        logging.debug("called flag_bad_images with %s" % method)
        if method == "hampel":
            # hampel outlier test on imagequality metrics
            tmp = df_iq.apply(stats.hampel, axis=0)
            outlier = tmp.abs().sum(axis=1) # abs of hampel outliers
            lim = np.median(outlier) + 1.5 * stats.iqr(outlier)
            return list(outlier[outlier > lim].index.tolist())
        elif method == "focus":
            # beyond normal range for focus score (low is bad)
            tmp = df.subset_col(df_iq, "FocusScore_")
            outlier = tmp.apply(stats.u_iqr, axis=1).sum(axis=1)
            return list(outlier[outlier != 0].index.tolist())
        elif method == "plls":
            # beyond normal range for powerloglogslope metrics (high is bad)
            tmp = df.subset_col(df_iq, "PowerLogLogSlope_")
            outlier = tmp.apply(stats.o_iqr, axis=1).sum(axis=1)
            return list(outlier[outlier != 0].index.tolist())
        elif method == "correlation":
            # beyond normal range for correlation (high is bad)
            tmp = df.subset_col(df_iq, "Correlation_")
            outlier = tmp.apply(stats.o_iqr, axis=1).sum(axis=1)
            return list(outlier[outlier != 0].index.tolist())
        else:
            logging.error("Non-valid method of '%s' for flag_bad_image" %method)
            ValueError("Invalid method. Options: hampel, focus, plls")



    def flag_errors(self):
        """
        Using ModuleError columns, flag images that produced an error.
        Return ImageNumber, can be passed to remove_images()
        """
        image = pd.read_sql_table("Image", self.connection,
                                  index_col="ImageNumber")
    df_err = df.subset_col(image, "ModuleError_")
    errors = df_err.sum(axis=1)
    logging.info("%i errors found in ModuleErrors" % sum(errors))
    return list(errors[errors > 0].index.tolist())



    def remove_images(self, images, merged=True):
        """
        Given a list of ImageNumbers, this function will remove them from
        either all tables, or from single merged tables
        """
        # match ImageNumber to row index
        # remove row index
        pass


    # metadata and featuredata columns
    # how to know which is metadata and which is featuredata
    # metadata can be selected with str.contains("Metadata_")
    # some featuredata is not useful
    # can use those from measurements module names:
        # either get these manually and store somewhere
        # or exclude known nuisance measurements
        # leaning towards storing known measurements



if __name__ == "__main__":
    x = RawData("/mnt/datastore/Scott_1/db_test.sqlite")
    x.aggregate_imagenumber()
    print x.flag_bad_images(method="hampel")
    print x.flag_errors()
