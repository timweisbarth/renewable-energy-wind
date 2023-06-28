import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import re
from sklearn.ensemble import RandomForestRegressor


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

    params = {"xgboost": {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.01},
              "linreg": {},
              "auto": {}
              }
    

    if model_name == "linreg":

        params = params[model_name]
        model = LinearRegression(**params)
        model.fit(np.concatenate([X_train, X_val]),
                  np.concatenate([y_train, y_val]))
    
    if model_name == "xgboost_HPO":

        # Define the parameter grid for HPO
        param_grid = {
            'max_depth': [3, 5, 6, 7],
            'learning_rate': [0.1, 0.02],
            'n_estimators': [350, 500, 1000, 1500],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0, 0.1, 0.5]
        }

        
        #  Create a XGBRegressor instance
        xgbr = xgb.XGBRegressor()

        # Instantiate the grid search
        grid_search = RandomizedSearchCV(estimator=xgbr, 
                                   param_distributions=param_grid, 
                                   n_iter = 20,
                                   verbose=10,
                                   scoring='neg_mean_squared_error',
                                   cv=2, 
                                   n_jobs=-1)

        # Fit the grid search to the data
        grid_search.fit(np.concatenate([X_train, X_val]),
                        np.concatenate([y_train, y_val]))

        # Print and prepare best params for model fitting
        best_params = grid_search.best_params_
        params["xgboost"] = best_params
        print(best_params)
    
    if re.match(r"^xgboost\w*", model_name):

        params = params["xgboost"]
        model = xgb.XGBRegressor(**params, early_stopping_rounds = 50)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_val, y_val)],
                  verbose=True)
        # Get the default parameters
        default_parameters = model.get_params()

        # Print the default parameters
        for param_name, value in default_parameters.items():
            print(f"{param_name} : {value}")
    
    if model_name == "rf":

        # Create a Random Forest Regressor object with the desired parameters
        model = RandomForestRegressor(n_estimators=200, max_depth=6)
        
        # Fit the model to the training data
        model.fit(np.concatenate([X_train, X_val]),
                np.concatenate([y_train, y_val]).ravel())
        

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

    predictions = scaler.inverse_transform(
        reg.predict(X_test_arr).reshape(-1, 1))
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
