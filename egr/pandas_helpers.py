from typing import List, Optional

import pandas as pd


def mean_df(
    df_list: List[pd.DataFrame], columns: Optional[List[str]] = None
) -> pd.DataFrame:
    if len(df_list) == 0:
        return pd.DataFrame()

    df = sum(df_list) / len(df_list)
    if columns == None:
        return df
    return pd.DataFrame(df, columns=[columns])


def element_wise_mean(dfs: List[pd.DataFrame], cols: List[str]) -> pd.DataFrame:
    tables = [df.to_numpy() for df in dfs]
    a = sum(tables) / len(tables)
    return pd.DataFrame(a, columns=cols)
