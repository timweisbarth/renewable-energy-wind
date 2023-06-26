import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np


def plot_models(name_of_farm, models):
    """Plots the predictions and vs. y_test of the models

    Parameters:
    -----------
    name_of_farm: string
        which of the farms is plotted?
    models: list of dictionaries
        Each list entry contains a dictionary of the key properties
        of a model

    Returns:
    -------
    NoneType
    """
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))

    for i, model in enumerate(models):
        ax[i].plot(model["X_test"].index, model["truths"], label='Truth')
        ax[i].plot(model["X_test"].index,
                   model["predictions"], label='Prediction')
        ax[i].set_xlabel('dates')   # Set x-axis label
        ax[i].set_ylabel('power [kW]')   # Set y-axis label
        ax[i].set_title(model["name"])  # Set title for each subplot
        ax[i].legend()  # Display the legend

    plt.tight_layout()  # To prevent overlapping of subplots
    fig.suptitle(
        f"{name_of_farm} wind farm: Comparison of power predictions",
        fontsize=13, y=1.02)
    plt.show()


def plot_metrics(name_of_farm, models):
    """ASCII style table of the performance metrics of the models

    Parameters:
    -----------
    name_of_farm: string
        which of the farms is plotted?
    models: list of dictionaries
        Each list entry contains a dictionary of the key properties
        of each model

    Returns:
    -------
    NoneType
    """

    # Benchmarks as provided by task
    benchmarks = {
        "Kelmarsh": [
            {"rmse": 145.602811, "mae": 91.553768},
            {"rmse": 263.749456, "mae": 183.285915},
            {"rmse": 623.023208, "mae": 510.709616},
        ],
        "Beberide": [
            {"rmse": 55.417200, "mae": 36.244988},
            {"rmse": 119.249865, "mae": 81.943719},
            {"rmse": 196.741606, "mae": 151.508157}
        ],
        "Pedra do Sal": [
            {"rmse": np.nan, "mae": np.nan},
            {"rmse": np.nan, "mae": np.nan},
            {"rmse": np.nan, "mae": np.nan}
        ]
    }[name_of_farm]

    # Extract the relevant details for the table
    table_data = [
        [
            name_of_farm + " " + model["name"],
            model["rmse"],
            bm["rmse"],
            model["mae"],
            bm["mae"]
        ] for (model, bm) in zip(models, benchmarks)
    ]

    headers = ['Model Name', 'RMSE', 'Benchmark_RMSE', 'MAE', 'Benchmark_MAE']

    # Create the table
    table = tabulate(table_data, headers, tablefmt="pipe")

    print(table)
