import os
import re
import pandas as pd
from sqlalchemy import create_enginee

# TODO: variable database name


class results_directory:
    
    """
    Directory containing the .csv from a CellProfiler run
    ------------------------------------------------------

    - create_db(): creates an sqlite database in the results directory

    - to_db(): loads the csv files in the directory and writes them as
               tables to the sqlite database created by create_db()

    - truncate: argument will name files from text between the last underscore
      and .csv. e.g 'date_plate_cell.csv' will return 'cell'
    """

    def __init__(self, file_path, truncate = True):
        # path of directory
	self.path = file_path
	# full name of csv files
	full_paths = [i for i in os.listdir(file_path) if i.endswith(".csv")]
	self.full_paths = full_paths

        if truncate == True:
	    # trim between _ and .csv
	    p = re.compile(ur'[^_][^_]+(?=.csv)')
	    csv_files = []

	    for csv in full_paths:
	        csv_files.append(re.search(p, csv).group())
            self.csv_files = csv_files
        else:
            self.csv_files = full_paths
        
    # create sqlite database in cwd    
    def create_db(self):
	self.engine = create_engine('sqlite:///database.sqlite')
    
    # write csv files to database
    def to_db(self):
	for x in xrange(len(self.full_paths)):
            f = os.path.join(self.path, self.full_paths[x])
	    tmp_file = pd.read_csv(f, iterator = True, chunksize = 1000)
            all_file = pd.concat(tmp_file)
	    all_file.to_sql(name = self.csv_files[x], con = self.engine,
                    flavor ='sqlite', index = False, if_exists = 'replace',
                    chunksize = 1000)

