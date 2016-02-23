import os
import re
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

# TODO: variable database name

# TODO: truncate removes everything but final string before .csv
#       currently rubbish if more than one underscore

class results_directory:
    
    """
    Directory containing the .csv from a CellProfiler run
    ------------------------------------------------------

    - create_db(): creates an sqlite database in the results directory

    - to_db(): loads the csv files in the directory and writes them as
               tables to the sqlite database created by create_db()
    """

    def __init__(self, file_path, truncate = True):
	self.path = file_path
	# full name of csv files
	full_paths = [i for i in os.listdir(file_path) if i.endswith(".csv")]
	self.full_paths = full_paths

        if truncate == True:
	    # trim between _ and .csv
	    p = re.compile(ur'(?<=_)(.*)(?=.csv)')
	    csv_files = []

	    for csv in full_paths:
	        csv_files.append(re.search(p, csv).group())
            
            self.csv_files = csv_files

        else:
            self.csv_files = full_paths
        
        
    def create_db(self):
	self.engine = create_engine('sqlite:///database.sqlite')

    def to_db(self):
	for x in xrange(len(self.full_paths)):
            f = os.path.join(self.path, self.full_paths[x])
	    tmp_file = pd.read_csv(f)
	    tmp_file.to_sql(self.csv_files[x], self.engine)


