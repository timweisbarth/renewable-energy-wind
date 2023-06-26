from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Scale data according to scaler
def scale_data(scaler,X_train, X_val, X_test, y_train, y_val, y_test):

  X_train_arr = scaler.fit_transform(X_train)
  X_val_arr = scaler.transform(X_val)
  X_test_arr = scaler.transform(X_test)

  y_train_arr = scaler.fit_transform(y_train)
  y_val_arr = scaler.transform(y_val)
  y_test_arr = scaler.transform(y_test)
  return X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr

# train model according to type
def train_model(type,X_train_arr,X_val_arr,y_train_arr,y_val_arr):
  reg = {"xgboost":xgb.XGBRegressor(n_estimators=1000)}[type]
  reg.fit(X_train_arr, y_train_arr,
          eval_set=[(X_train_arr,y_train_arr),(X_val_arr,y_val_arr)],
          early_stopping_rounds=50,
          verbose=False) # Change verbose to True if you want to see it train
  return reg

# Inverse scaling
def inverse_scaler(reg, scaler, X_test_arr,y_test_arr):
  predictions = scaler.inverse_transform(reg.predict(X_test_arr).reshape(-1,1))
  truths = scaler.inverse_transform(y_test_arr)  
  return predictions, truths

def model_metrics(predictions,truths):
  rmse = mean_squared_error(y_true=truths,
                   y_pred=predictions,
                   squared=False)

  mae = mean_absolute_error(y_true=truths,
                   y_pred=predictions)
  
  return rmse,mae