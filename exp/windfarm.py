import pandas as pd
import numpy as np
from itables import init_notebook_mode
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error

init_notebook_mode(all_interactive=True)

def load_data():
    """ Load the data of the second turbine of Kelmarsh wind farm
        Return: Dataframe
    """    
    kwf2_paths = [
            "./UK/Kelmarsh_SCADA_2016_3082/Turbine_Data_Kelmarsh_2_2016-01-03_-_2017-01-01_229.csv",
            "./UK/Kelmarsh_SCADA_2017_3083/Turbine_Data_Kelmarsh_2_2017-01-01_-_2018-01-01_229.csv",
            "./UK/Kelmarsh_SCADA_2018_3084/Turbine_Data_Kelmarsh_2_2018-01-01_-_2019-01-01_229.csv",
            "./UK/Kelmarsh_SCADA_2019_3085/Turbine_Data_Kelmarsh_2_2019-01-01_-_2020-01-01_229.csv",
            "./UK/Kelmarsh_SCADA_2020_3086/Turbine_Data_Kelmarsh_2_2020-01-01_-_2021-01-01_229.csv",
            "./UK/Kelmarsh_SCADA_2021_3087/Turbine_Data_Kelmarsh_2_2021-01-01_-_2021-07-01_229.csv"
        ]

    kwf2 = pd.DataFrame()

    for path in kwf2_paths:
        
        df = pd.read_csv(
            path,
            skiprows =9
            )
        kwf2 = pd.concat([kwf2, df])
    
    return kwf2



def preprocessing(df,non_nan_percentage,uk):
  # Rename and set index
  if uk:
    # Mostly NaNs until 2016-01-21
    df = df["2016-01-21":]

    df = df.rename(columns={"Power (kW)":"power",'# Date and time':'time'})
  else: 
     df = df.rename(columns={"active_power_total" :"power",'Time':'time'})
     
  df = df.set_index(["time"])
  df.index = pd.to_datetime(df.index)

  # Require 70% non-NaN values for column to remain
  df = df.dropna(thresh=df.shape[0]*non_nan_percentage/100,axis=1)

 
  
  # Interpolate for inside NaNs
  df = df.interpolate(limit_area='inside')
  
  # use backfill and forwardfill for the nans in the beginning/end
  df = df.fillna(df.bfill()).fillna(df.ffill())

  return df


# Caputre cyclic structure of time features
def generate_cyclic_features(df, col_name, period,start_num=0):
  kwargs = {
    f'sin_{col_name}': lambda x: np.sin(2*np.pi*(df[col_name] -start_num)/period),
    f'cos_{col_name}': lambda x: np.cos(2*np.pi*(df[col_name] -start_num)/period)

  }
  return df.assign(**kwargs).drop(columns=[col_name])


def generate_features(df,shift,weather_col_names):
  '''Generate the features for time series prediction
  
  '''
  # Generate shifted y
  df[f"power_next_{shift}"] = df["power"].shift(-shift)

  # Add lagged data to dataframe
  lags = {1:[1,2,3,6,12,24],6:[1,2,6,12,24,48],144:[1,2,6,12,24,48,100,144]}[shift]
  for lag in lags:
    df[f"power_lag{lag}"] = df["power"].shift(lag)
    for col_name in weather_col_names:
      df[f"{col_name}_lag_{lag}"] = df[f"{col_name}"].shift(lag)


  df = (
  df
  .assign(hour = df.index.hour)
  .assign(month = df.index.month)
  )

  df = generate_cyclic_features(df, 'hour', 24,0)
  df = generate_cyclic_features(df, 'month', 12,1)

  # TODO: Check whether doint the right thing

  no_col_before = df.shape[0]
  print("Shape before dropna in feature generation",df.shape)
  df = df.dropna(axis=0)
  print("Shape after dropna in feature generation",df.shape)
  print("Number of dropped rows",-df.shape[0]+no_col_before)

  return df

# Split data set in X and y
def feature_label_split(df, target_col):
    y = df[[target_col]]
    X = df.drop(columns=[target_col])
    return X, y

from sklearn.model_selection import train_test_split

# Split data in train, validation and test
def train_val_test_split(df, target_col, val_ratio,uk):
    X, y = feature_label_split(df, target_col)
    if uk:
      X_train, X_test, y_train, y_test = X[:'2020-07-01'], X['2020-07-01':], y[:'2020-07-01'], y['2020-07-01':]
    else:
       X_train, X_test, y_train, y_test = X[:'2014-04-13'], X['2014-04-13':], y[:'2014-04-13'], y['2014-04-13':]
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_ratio, shuffle=False)
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_data(scaler,X_train, X_val, X_test, y_train, y_val, y_test):

  X_train_arr = scaler.fit_transform(X_train)
  X_val_arr = scaler.transform(X_val)
  X_test_arr = scaler.transform(X_test)

  y_train_arr = scaler.fit_transform(y_train)
  y_val_arr = scaler.transform(y_val)
  y_test_arr = scaler.transform(y_test)
  return X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr

def train_model(type,X_train_arr,X_val_arr,y_train_arr,y_val_arr):
        reg = {"xgboost":xgb.XGBRegressor(n_estimators=1000)}[type]
        reg.fit(X_train_arr, y_train_arr,
                eval_set=[(X_train_arr,y_train_arr),(X_val_arr,y_val_arr)],
                early_stopping_rounds=50,
                verbose=False) # Change verbose to True if you want to see it train
        return reg