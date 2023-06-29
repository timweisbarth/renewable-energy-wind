# Kelmarsh:</br>

n_est = 1000, earlystopping=50
| Model Name              |    RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|--------:|-----------------:|---------:|----------------:|
| Kelmarsh 10min horizon  | 142.219 |          145.603 |  89.3621 |         91.5538 |


{'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.01}
| Model Name              |    RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|--------:|-----------------:|---------:|----------------:|
| Kelmarsh 10min horizon  | 142.483 |          145.603 |  90.5201 |         91.5538 |


n_iter = 20 HP search:
{'subsample': 1.0, 'reg_lambda': 0.1, 'reg_alpha': 0.5, 'n_estimators': 350, 'max_depth': 3, 'learning_rate': 0.02, 'gamma': 0.1, 'colsample_bytree': 1.0}
| Model Name              |    RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|--------:|-----------------:|---------:|----------------:|
| Kelmarsh 10min horizon  | 147.612 |          145.603 |  91.9744 |         91.5538 |


n_iter = 20 HP search:
{'subsample': 1.0, 'reg_lambda': 0.1, 'reg_alpha': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.02, 'gamma': 0.1, 'colsample_bytree': 0.8}
| Model Name              |    RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|--------:|-----------------:|---------:|----------------:|
| Kelmarsh 1 day horizon  | 596.683 |          623.023 | 476.414  |        510.71   |

n_iter = 20 HP search:
{'subsample': 1.0, 'reg_lambda': 0.5, 'reg_alpha': 0.5, 'n_estimators': 350, 'max_depth': 3, 'learning_rate': 0.02, 'gamma': 0.1, 'colsample_bytree': 1.0}
| Model Name              |    RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|--------:|-----------------:|---------:|----------------:|
| Kelmarsh 1 hour horizon | 246.713 |          263.749 | 169.369  |        183.286  |

# Beberide: </br>

### Ordinary least squares regression
| Model Name              |     RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|---------:|-----------------:|---------:|----------------:|
| Beberide 10min horizon  |  52.4074 |          55.4172 |  34.5282 |         36.245  |
| Beberide 1 hour horizon | 110.986  |         119.25   |  79.5638 |         81.9437 |
| Beberide 1 day horizon  | 178.003  |         196.742  | 130.404  |        151.508  |


### xgboost
 {'subsample': 0.8, 'reg_lambda': 0.5, 'reg_alpha': 0.1, 'n_estimators': 1000, 'max_depth': 3, 'learning_rate': 0.01, 'gamma': 0. 'colsample_bytree': 0.8}
 | Model Name              |     RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
 |:------------------------|---------:|-----------------:|---------:|----------------:|
 | Beberide 10min horizon  |  51.5927 |          55.4172 |  33.4749 |         36.245  |
 
{'n_estimators': 1000}
| Model Name              |     RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|---------:|-----------------:|---------:|----------------:|
| Beberide 10min horizon  |  53.7562 |          55.4172 |  35.2344 |         36.245  |

### Random Forest
n_estimators=200, max_depth=6
| Model Name              |     RMSE |   Benchmark_RMSE |      MAE |   Benchmark_MAE |
|:------------------------|---------:|-----------------:|---------:|----------------:|
| Beberide 10min horizon  |  53.4398 |          55.4172 |  34.5853 |         36.245  |