import preprocessing as pp
import model as m


def pipeline(df, uk, shifts, non_nan_percentage, col_to_be_lagged, val_ratio, scalers):
  models = []
  for i,shift in enumerate(shifts):
    model = pipeline_helper(df,
                            uk, 
                            shift, 
                            non_nan_percentage, 
                            col_to_be_lagged, 
                            val_ratio, 
                            scalers[i])
    models.append(model)
  return models


def pipeline_helper(df, uk, shift, non_nan_percentage, col_to_be_lagged, val_ratio, scaler):
  X_train, X_val, X_test, y_train, y_val, y_test = \
    pp.all_preproc_steps(df=df,
                         uk=uk,
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


  reg = m.train_model("xgboost", X_train_arr, X_val_arr, y_train_arr, y_val_arr)
  predictions, truths = m.inverse_scaler(reg,scaler, X_test_arr, y_test_arr)
  rmse, mae = m.model_metrics(predictions, truths)

  name = {1:"10min horizon", 6:"1 hour horizon", 144:"1 day horizon"}[shift]

  print(f"Finished training model {name}")

  return {"name":name, "rmse":rmse, "mae":mae,
          "X_test":X_test,"predictions":predictions, "truths":truths
          }

