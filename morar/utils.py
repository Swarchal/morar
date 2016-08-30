def get_featuredata(df, metadata_prefix="Metadata"):
    """
    identifies columns in a dataframe that are not labelled with the
    metadata prefix. Its assumed everything not labelled metadata is
    featuredata

    @param df dataframe
    @param metadata_prefix string, prefix for metadata columns
    @return list of column names
    """
    f_cols = [i for i in df.columns if not i.startswith(metadata_prefix)]
    return f_cols


def get_metadata(df, metadata_prefix="Metadata"):
    """
    identifies column in a dataframe that are labelled with the metadata_prefix

    @param df pandas DataFrame
    @param metadata_prefix metadata prefix in column name
    @return list of column names
    """
    m_cols = [i for i in df.columns if i.startswith(metadata_prefix)]
    return m_cols