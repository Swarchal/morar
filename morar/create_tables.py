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

    - load_csv(): reads csv files as pandas DataFrame, method used by
		  write_to_db()
		  - truncate trims name down to just the object name

    - create_db(): creates an sqlite database in the results directory
		   - name is the argument to name the database

    - write_to_db(): loads the csv files using load_csv() and writes
		     them as tables to the sqlite database created by
		     create_db()
    """

    def __init__(self):
	self.path = file_path


    def list_csv(self):
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


    def create_db(self, name):
	engine = create_engine('sqlite:///database.sqlite')


    def write_to_db(self, truncate = True):
	for x in self.full_paths:
	    tmp_file = pd.read_csv(x, chunk = 1000)
	    tmp_file.to_sql(self.csv_files, engine)

