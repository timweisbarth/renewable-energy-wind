import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import mean_squared_error as mse

import os
os.environ["OPENBLAS_NUM_THREADS"] = "4"



def scale_data(scaler, X_train, X_val, X_test, y_train, y_val, y_test):
    """ Scales the data according to given scaler

     Parameters:
    -----------
    scaler: sklearn.preprocessing
        e.g. MinMaxScaler
    X_train, X_val, X_test, y_train, y_val, y_test: pd.DataFrame
        Training, evaluation and test sets

    Returns:
    -------
    tuple of ndarrays:
        Scaled training, evaluation and test sets
    """

    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train)
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)

    return X_train_arr, X_val_arr, X_test_arr, \
        y_train_arr, y_val_arr, y_test_arr


def train_model(model_name, X_train, X_val, y_train, y_val):
    """ Trains model chosen by type

     Parameters:
    -----------
    model_name: string
        Name of model choice
    X_train, X_val, y_train, y_val
        Training and evaluation sets

    Returns:
    -------
    tuple
        Scaled training, evaluation and test sets
    """

    params = {"xgboost": {"n_estimators": 1000, "early_stopping_rounds": 50},
              "linreg":{},
              "auto":{}
    }
    params = params[model_name]

    os.environ["OPENBLAS_NUM_THREADS"] = "4"

    #if type == "auto":
    #    # Instantiate the regressor
    #    reg = AutoSklearnRegressor(time_left_for_this_task=120, # run auto-sklearn for at most 2min
    #                       per_run_time_limit=30, # spend at most 30 sec for each model training
    #                       metric=mse,
    #                       n_jobs=4,  # This will use 4 cores
    #                       memory_limit=2048
    #                       )
    #
    #    # Fit the model
    #    reg.fit(X_train, y_train)
    #    # Summary statistics
    #    print(reg.sprint_statistics())
    #    # Detailed information of all models found
    #    print(reg.show_models())
        

    if model_name == "linreg":
        model = LinearRegression(**params)
        model.fit(np.concatenate([X_train, X_val]),np.concatenate([y_train,y_val]))
    if model_name == "xgboost":
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False)
    
    return model


def predict_and_inv_scaler(reg, dataset_name, scaler, X_test_arr, y_test_arr):
    """Returns model's predictions on test data and inverts the scaling

    Parameters:
    -----------
    reg:
        Learnt model choice
    dataset_name: string
      From which farm is the dataset?
    scaler: sklearn.preprocessing
        e.g. MinMaxScaler
    X_test_arr, y_test_arr: ndarrays
        Test set

    Returns:
    -------
    tuple:
        prediction of reg on test set and y_test (= truth)
    """

    predictions = scaler.inverse_transform(reg.predict(X_test_arr).reshape(-1, 1))
    truths = scaler.inverse_transform(y_test_arr)

    # Cut unrealistic predictions by min/max value that occured in dataset
    # 2080kW for kwf, 930kW for ueps, 840kW for uebb
    power_max = 2080 if dataset_name == "kwf" else \
                930 if dataset_name == "ueps" else 840
    
    predictions = [0 if pred < 0 
                   else 
                   power_max if pred > power_max 
                   else 
                   float(pred) 
                   for pred in predictions
                   ]
    
    predictions = np.array(predictions)

    return predictions, truths


def model_metrics(predictions, truths):
    """Computes the RMSE and MAE

    Parameters:
    -----------
    predictions: ndarray
        predictions of the model on test set
    truths: ndarray
        y_test

    Returns:
    -------
    tuple:
        Rmse and mae
    """

    rmse = mean_squared_error(y_true=truths,
                              y_pred=predictions,
                              squared=False)

    mae = mean_absolute_error(y_true=truths,
                              y_pred=predictions)

    return rmse, mae
