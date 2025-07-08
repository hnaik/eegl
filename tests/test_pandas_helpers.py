import pandas as pd

from egr.pandas_helpers import mean_df


def test_mean_df():
    df_1 = pd.DataFrame(data={'A': [1, 2, 3, 4], 'B': [2, 4, 6, 8]})
    df_2 = pd.DataFrame(data={'A': [1, 1, 1, 1], 'B': [2, 2, 2, 2]})

    df_expected = pd.DataFrame(
        data={'A': [1.0, 1.5, 2.0, 2.5], 'B': [2.0, 3.0, 4.0, 5.0]}
    )
    df_actual = mean_df([df_1, df_2])

    pd.testing.assert_frame_equal(df_actual, df_expected)


def test_mean_df_columns():
    df_1 = pd.DataFrame(data={'A': [1, 2, 3, 4], 'B': [2, 4, 6, 8]})
    df_2 = pd.DataFrame(data={'A': [1, 1, 1, 1], 'B': [2, 2, 2, 2]})

    df_expected = pd.DataFrame(
        data={
            'A-changed': [1.0, 1.5, 2.0, 2.5],
            'B-changed': [2.0, 3.0, 4.0, 5.0],
        }
    )
    df_actual = mean_df([df_1, df_2], columns=['A-changed', 'B-changed'])

    # pd.testing.assert_frame_equal(df_actual, df_expected)
    print(df_actual)
