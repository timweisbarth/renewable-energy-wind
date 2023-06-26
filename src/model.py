import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def train_model(type, X_train_arr, X_val_arr, y_train_arr, y_val_arr):
    """ Trains model chosen by type

     Parameters:
    -----------
    type: string
        Name of model choice
    X_train, X_val, y_train, y_val
        Training and evaluation sets

    Returns:
    -------
    tuple
        Scaled training, evaluation and test sets
    """

    params = {"xgboost": {"n_estimators": 1000, "early_stopping_rounds": 50}}
    params = params[type]

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_arr, y_train_arr,
              eval_set=[(X_train_arr, y_train_arr), (X_val_arr, y_val_arr)],
              verbose=False)
    
    return model


def inverse_scaler(reg, scaler, X_test_arr, y_test_arr):
    """Inverts the scaling introduced by scaler on the test data and the
    predictions

    Parameters:
    -----------
    reg:
        Name of model choice
    scaler: sklearn.preprocessing
        e.g. MinMaxScaler
    X_test_arr, y_test_arr: ndarrays
        Test set

    Returns:
    -------
    tuple:
        Scaled training, evaluation and test sets
    """

    predictions = scaler.inverse_transform(
        reg.predict(X_test_arr).reshape(-1, 1))
    truths = scaler.inverse_transform(y_test_arr)
    
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
