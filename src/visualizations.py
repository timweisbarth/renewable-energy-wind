import matplotlib.pyplot as plt
from tabulate import tabulate

def plot_models(name_of_farm, models):
  fig, ax = plt.subplots(3,1, figsize=(10,10))

  for i, model in enumerate(models):
      ax[i].plot(model["X_test"].index, model["truths"], label='Truth')
      ax[i].plot(model["X_test"].index, model["predictions"], label='Prediction')
      ax[i].set_xlabel('dates')   # Set x-axis label
      ax[i].set_ylabel('power [kW]')   # Set y-axis label
      ax[i].set_title(model["name"])  # Set title for each subplot
      ax[i].legend()  # Display the legend

  plt.tight_layout()  # To prevent overlapping of subplots
  fig.suptitle(f"{name_of_farm} wind farm: Comparison of power predictions", fontsize=13, y=1.02) 
  plt.show()

def plot_metrics(name_of_farm, models):
 
  # Extract the relevant details for the table
  table_data = [[name_of_farm + " " + model["name"], model["rmse"], model["mae"]] for model in models]

  # Define the headers
  headers = ['Model Name', 'RMSE', 'MAE']
  
  # Create the table
  table = tabulate(table_data, headers, tablefmt="pipe")

  print(table)