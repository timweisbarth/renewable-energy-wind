import preprocessing as pp
import model as m


def pipeline(df, models_names, dataset_name, shifts, non_nan_percentage,
             col_to_be_lagged, val_ratio, scalers):
    """ Wrapper for pipeline_worker

     Parameters:
    -----------
    see pipeline_worker for documentation

    Returns:
    -------
    list of dictionaries
        Each dictionary contains key properties of each time horizon model

    """

    models = []
    for i, shift in enumerate(shifts):
        model = pipeline_worker(df,
                                models_names[i],
                                dataset_name,
                                shifts[i],
                                non_nan_percentage,
                                col_to_be_lagged,
                                val_ratio,
                                scalers[i])
        models.append(model)
    return models


def pipeline_worker(df, model_name, dataset_name, shift, non_nan_percentage,
                    col_to_be_lagged, val_ratio, scaler):
    """ Executes the  ML-pipeline as specified below

     Parameters:
    -----------
    df: pd.DataFrame
      data
    model_name: string
        Name of the model to be used
    dataset_name: string
      From which farm is the dataset?
    shift: int ∈ {1,6,144}
      Number of time steps the target column gets shifted e.g. 1 hour = 6 * 10min
    non_nan_percentage: int ∈ [0,100]
        Require non_nan_percentage % many non-NaN values for a column to remain
    col_to_be_lagged: list
        Columns of which lagged values will be added to df
    val_ratio: float ∈ [0,1]
        |Training set| * val_ratio = |Validation set|
    scaler: sklearn.preprocessing
        e.g. MinMaxScaler

    Returns:
    -------
    dictionary
        Contains key properties of the evaluated model
    """

    X_train, X_val, X_test, y_train, y_val, y_test = \
        pp.all_preproc_steps(df=df,
                             dataset_name=dataset_name,
                             shift=shift,
                             non_nan_percentage=non_nan_percentage,
                             col_to_be_lagged=col_to_be_lagged,
                             val_ratio=val_ratio)

    X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr = \
        m.scale_data(scaler,
                     X_train,
                     X_val,
                     X_test,
                     y_train,
                     y_val,
                     y_test)

    reg = m.train_model(model_name, X_train_arr,
                        X_val_arr, y_train_arr, y_val_arr)

    predictions, truths = m.predict_and_inv_scaler(reg, dataset_name, scaler, X_test_arr, y_test_arr)
    

    rmse, mae = m.model_metrics(predictions, truths)

    horizon = {1: "10min horizon", 6: "1 hour horizon",
            144: "1 day horizon"}[shift]

    print(f"Finished training on {dataset_name} for {horizon}")

    return {
        "horizon": horizon, "rmse": rmse, "mae": mae,
        "X_test": X_test, "predictions": predictions, "truths": truths
    }
