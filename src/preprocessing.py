import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

"""
The cyclic feature generation was taken from
https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b
"""


def all_preproc_steps(
        df, uk, shift, non_nan_percentage,
        col_to_be_lagged, val_ratio):
    """Apply all preprocessing steps. For description see individual functions.

    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test.
    """
    df = preproc1_names_and_index(df, uk)
    df = preproc2_nans(df, uk, non_nan_percentage)
    df = preproc3_featgen(df, shift, col_to_be_lagged)
    df = preproc4_train_val_test_split(
        df, uk, f'power_next_{shift}', val_ratio)
    return df


def preproc1_names_and_index(df, uk):
    """Change the name of the target column to "power" and index to "time"
    Make index a datetime object

    Parameters:
    -----------
    df: pd.DataFrame
      data
    uk: bool
      Is the input df from the uk dataset?

    Returns:
    -------
    pd.DataFrame
    """
    if uk:
        df = df.rename(columns={"Power (kW)": "power",
                       '# Date and time': 'time'})
    else:
        df = df.rename(columns={"active_power_total": "power", 'Time': 'time'})

    df = df.set_index(["time"])
    df.index = pd.to_datetime(df.index)

    return df


def preproc2_nans(df, uk, non_nan_percentage):
    """Drop columns if too many nans. Interpolate for inside NaNs,
    fill rest with next valid value

    Parameters:
    -----------
    df: pd.DataFrame
      data
    uk: bool
      Is the input df from the uk dataset?
    non_nan_percentage: float
      Require non_nan_percentage % many non-NaN values for a column to remain.

    Returns:
    -------
    pd.DataFrame
    """
    if uk:
        # Mostly NaNs until 2016-01-21
        df = df["2016-01-21":]

    # Require 70% non-NaN values for column to remain
    df = df.dropna(thresh=df.shape[0]*non_nan_percentage/100, axis=1)
    df = df.interpolate(limit_area='inside')  # Interpolate for inside NaNs
    df = df.fillna(df.bfill()).fillna(df.ffill())  # outside NaNs

    return df


def generate_cyclic_features(df, col_name, period, start_num=0):
    """Specifically encode the cyclic structure of hours and days

    Parameters:
    ----------
    df: pd.DataFrame
    col_name: string
      cyclic feature to be dealt with
    period: int
      how long until the feature repeats itsself e.g. after 24h

    Returns:
    --------
    pd.DataFrame
    """

    # TODO: How does it work?
    kwargs = {
        f'sin_{col_name}': lambda x: np.sin(2*np.pi*(
            df[col_name] - start_num)/period),
        f'cos_{col_name}': lambda x: np.cos(2*np.pi*(
            df[col_name] - start_num)/period)
    }

    return df.assign(**kwargs).drop(columns=[col_name])


def preproc3_featgen(df, shift, col_to_be_lagged):
    """Generate interesting features in order to make time-series data
    a supervised problem. Encode time dependencies through lags.

    Parameters:
    -----------
    df: pd.DataFrame
      data
    shift: int ∈ {1,6,144}
      Number of time steps the target column gets shifted
    col_to_be_lagged: list
      Columns of which lagged values will be added to df

    Returns:
    -------
    pd.DataFrame
    """

    # Generate shifted target column y s.t. features_(t)
    # are in one row with y_(t+shift)
    df[f"power_next_{shift}"] = df["power"].shift(-shift)

    # Add lagged data to dataframe, lags depend on time horizon of prediction
    lags = {1: [1, 2, 3, 6, 12, 24], 6: [1, 2, 6, 12, 24, 48],
            144: [1, 2, 6, 12, 24, 48, 100, 144]}[shift]
    for lag in lags:
        df[f"power_lag{lag}"] = df["power"].shift(lag)
        for col_name in col_to_be_lagged:
            df[f"{col_name}_lag_{lag}"] = df[f"{col_name}"].shift(lag)

    # Include month and hour of each row
    df = (
        df
        .assign(hour=df.index.hour)
        .assign(month=df.index.month)
    )
    df = generate_cyclic_features(df, 'hour', 24, 0)
    df = generate_cyclic_features(df, 'month', 12, 1)

    df = df.dropna(axis=0)  # Drop nans introduced by lags and shift
    return df


def preproc4_train_val_test_split(df, uk, target_col, val_ratio):
    """Split data in train, validation and test set

    Parameters:
    -----------
    df: pd.DataFrame
    uk: bool
      Is df from the uk wind farm?
    target_col: string
      Column that is to be predicted
    val_ratio: float
      ratio of the training set that is used for validation

    Returns:
    -------
    6x pd.DataFrame,...,pd.DataFrame
      X_train, X_val, X_test, y_train, y_val, y_test
    """

    # feature label split
    y = df[[target_col]]
    X = df.drop(columns=[target_col])

    # Split data in train and test according to given benchmark
    if uk:
        X_train, X_test, y_train, y_test = X[:'2020-07-01'], \
            X['2020-07-01':], y[:'2020-07-01'], y['2020-07-01':]
    else:
        X_train, X_test, y_train, y_test = X[:'2014-05-18'], \
            X['2014-05-18':], y[:'2014-05-18'], y['2014-05-18':]

    # Split data in train and validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                      test_size=val_ratio,
                                                      shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test
