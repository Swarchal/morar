import os
import re
import pandas as pd
from sqlalchemy import create_engine, Table, ForeignKey, Metadata
from sqlalchemy.ext.declarative import declarative_base

# TODO: variable database name

class results_directory(file_path, truncate = True):
    
    """
    Directory containing the .csv from a CellProfiler run
    ------------------------------------------------------

    - create_db(): creates an sqlite database in the results directory

    - to_db(): loads the csv files in the directory and writes them as
               tables to the sqlite database created by create_db()
    """

    def __init__(self, truncate):
	self.path = file_path
	# full name of csv files
	full_paths = [i for i in os.listdir(file_path) if i.endswith(".csv")]
	self.full_paths = full_paths
        if truncate == True:
	    # trim between _ and .csv
	    p = re.compile(ur'(?<=_)(.*)(?=.csv)')
	    csv_files = []
	    for csv in csv_files:
	        csv_files.append(re.search(p, csv)
	    self.csv_files = csv_file
        else:
            self.csv_files = full_pathss

    def create_db(self):
	engine = create_engine('sqlite:///database.sqlite')


    def to_db(self):
	for x in enumerate(self.full_path)s:
	    tmp_file = pd.read_csv(self.full_path[x], chunk = 1000)
	    tmp_file.to_sql(self.csv_files[x], engine)
