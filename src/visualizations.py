import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_models(models):
  fig, ax = plt.subplots(3,1, figsize=(10,10))

  for i, model in enumerate(models):
      ax[i].plot(model["X_test"].index, model["truths"], label='Truth')
      ax[i].plot(model["X_test"].index, model["predictions"], label='Prediction')
      ax[i].set_xlabel('dates')   # Set x-axis label
      ax[i].set_ylabel('power [kW]')   # Set y-axis label
      ax[i].set_title(model["name"])  # Set title for each subplot
      ax[i].legend()  # Display the legend

  plt.tight_layout()  # To prevent overlapping of subplots
  plt.show()

def plot_metrics(models):
 
  # Extract the relevant details for the table
  table_data = [[model["name"], model["rmse"], model["mae"]] for model in models]

  # Define the headers
  headers = ['Model Name', 'RMSE', 'MAE']
  
  # Create the table
  table = tabulate(table_data, headers, tablefmt="pipe")

  print(table)