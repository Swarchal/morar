import pandas as pd

def subset_col(df, string):
    """
    Returns slice of dataframe, selecting only columns that
    contain 'string'
    """
    out = df[df.columns[df.columns.to_series().str.contains(string)]]
    return out                                                          
