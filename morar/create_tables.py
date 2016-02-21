import os
import re
import pandas as pd
from sqlalchemy import create_engine, Table, ForeignKey, Metadata
from sqlalchemy.ext.declarative import declarative_base

# TODO: variable database name

class results_directory(file_path):
    
    """
    Directory containing the .csv from a CellProfiler run
    ------------------------------------------------------

    - list_csv(): lists all .csv files in the directory
		  - truncate trims the prefix and the .csv from the file name

    - create_db(): creates an sqlite database in the results directory

    - write_to_db(): loads the csv files using load_csv() and writes
		     them as tables to the sqlite database created by
		     create_db()
    """

    def __init__(self):
	self.path = file_path


    def list_csv(self, truncate = True):
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
	    self.csv_files = full_paths


    def create_db(self):
	engine = create_engine('sqlite:///database.sqlite')


    def write_to_db(self, truncate = True):
	for x in self.full_paths:
	    tmp_file = pd.read_csv(x, chunk = 1000)
	    tmp_file.to_sql(self.csv_files, engine)

