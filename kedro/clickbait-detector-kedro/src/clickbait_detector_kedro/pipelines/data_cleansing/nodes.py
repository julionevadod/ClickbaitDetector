import pandas as pd
import numpy as np

def deduplicate_data(df: pd.DataFrame, deduplication_key: list[str]) -> pd.DataFrame:
    """Deduplicate data in df based on deduplication_key

    :param df: Data to be deduplicated
    :type df: pd.DataFrame
    :param deduplication_key: Keys upon which deduplication occurs
    :type deduplication_key: list[str]
    :return: Deduplicated data
    :rtype: pd.DataFrame
    """
    return df[
        ~df.duplicated(
            subset=deduplication_key,
            keep="first"
        )
    ].copy()

def datatype_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """Convert label 'clickbait' from text to numeric

    :param df: Dataframe containing truthClass column
    :type df: pd.DataFrame
    :return: Data with numerical label
    :rtype: pd.DataFrame
    """
    df_converted = df.copy()
    df_converted["truthClass"] = np.where(
        df_converted["truthClass"] == "clickbait",
        1,
        0
    )
    return df_converted
