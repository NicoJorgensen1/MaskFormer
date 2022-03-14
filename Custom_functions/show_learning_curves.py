# Import libraries
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted


# Define a function to compute the moving average of an input array or list
def mov_avg_array(inp_array, mov_of_last_n_elements=4, output_last_n_elements=1):                       # Define a function to compute the moving average of an array or a list
    assert output_last_n_elements <= mov_of_last_n_elements, "The moving average can't be outputted for more values than it is being calculated for"
    if mov_of_last_n_elements > len(inp_array): mov_of_last_n_elements = len(inp_array)                 # If the list/array isn't as long as the wanted moving-average value, the n is lowered
    used_array_part = inp_array[-mov_of_last_n_elements:]                                               # Extract the last mov_of_last_n_elements from the list to compute the moving average for
    used_array_cumsum = np.cumsum(used_array_part)                                                      # Compute the cumulated sum for the used array part
    used_array_mov_avg = np.divide(used_array_cumsum, np.arange(1,1+mov_of_last_n_elements))            # Compute the moving average of the used array part
    return used_array_mov_avg[-output_last_n_elements:]                                                 # Output the last output_last_n_elements of the moving average array 


# Define a function to load the metrics.json in each output directory
def load_json_metrics(config):
    metrics_list = [os.path.join(config["OUTPUT_DIR"], x) for x                                         # Iterate through the files in the output dir of the config file ...
        in os.listdir(config["OUTPUT_DIR"]) if x.startswith("metrics") and x.endswith(".json")]         # ... and extract the metrics.json filenames
    metrics_list = natsorted(metrics_list)                                                              # Perform natural sorting on the list of metrics files to assure that their epoch number are sorted
    metrics = {"epoch_num": list()}
    for epoch_idx, metric_file in enumerate(metrics_list):
        for it_idx, line in enumerate(open(metric_file)):
            vals = json.loads(line)
            for key in vals:
                if key not in list(metrics.keys()): metrics[key] = list()
                metrics[key].append(vals[key])
        metrics["epoch_num"].extend(np.repeat(a=epoch_idx+1, repeats=it_idx+1).tolist())
    for key in metrics.keys():                                                                          # Looping through all key values in the training metrics dictionary
        if "loss" not in key.lower(): continue                                                          # If the key is not a loss-key, skip to the next key
        key_val, mov_avg_val = list(), list()                                                           # Initiate lists to store the actual values and the moving-average computed values
        for item in metrics[key]:                                                                       # Loop through each item in the dict[key]->value list
            key_val.append(item)                                                                        # Append the actual item value to the key_val list
            mov_avg_val.append(mov_avg_array(inp_array=key_val, mov_of_last_n_elements=10, output_last_n_elements=1).item())    # Compute the next mov_avg val for the last 10 elements
        metrics[key] = mov_avg_val                                                                      # Assign the newly computed moving average of the dict[key]->values to the dictionary
    return metrics                                                                                      # Return the moving average value dictionary


# Function to display learning curves
def show_history(config, FLAGS):                                                                        # Define a function to visualize the learning curves
    # train_history = load_json_metrics(config=config)                                                    # Load the metrics into the history dictionary
    history = load_json_metrics(config=config)
    # val_history 
    loss_total = [key for key in history.keys() if "total_loss" in key.lower()]                         # Find all keys with loss_ce
    loss_ce = [key for key in history.keys() if "loss_ce" in key.lower()]                               # Find all keys with loss_ce
    loss_dice = [key for key in history.keys() if "loss_dice" in key.lower()]                           # Find all keys with loss_dice
    loss_mask = [key for key in history.keys() if "loss_mask" in key.lower()]                           # Find all keys with loss_mask
    learn_rate = [key for key in history.keys() if "lr" in key.lower()]                                 # Find all keys with loss_mask
    hist_keys = [loss_total, learn_rate, loss_ce, loss_dice, loss_mask]                                 # Combine the key-lists into a list of lists
    ax_titles = ["total_loss", "learning_rate", "loss_ce", "loss_dice", "loss_mask"]                    # Create titles for the axes
    colors = ["blue", "red", "black", "green", "magenta", "cyan", "yellow"]                             # Colors for the line plots
    fig = plt.figure(figsize=(17,8))                                                                    # Create the figure
    n_rows, n_cols, ax_count = 2, (2,3), 0                                                              # Initiate values for the number of rows and columns
    if FLAGS.use_per_pixel_baseline==True: n_rows = 1                                                   # If we train with the per_pixel_classification method, only CCE loss is calculated in the total_loss metric
    for row in range(n_rows):                                                                           # Loop through all rows
        for col in range(n_cols[row]):                                                                  # Loop through all columns in the current row
            plt.subplot(n_rows, n_cols[row], 1+row*n_cols[row]+col)                                     # Create a new subplot
            plt.xlabel(xlabel="Epoch #")                                                                # Set correct xlabel
            plt.ylabel(ylabel=ax_titles[ax_count].replace("_", " "))                                    # Set correct ylabel
            plt.grid(True)                                                                              # Activate the grid on the plot
            plt.xlim(left=0, right=np.max(history["epoch_num"]))                                        # Set correct xlim
            plt.title(label=ax_titles[ax_count].replace("_", " ").capitalize())                         # Set plot title
            y_top_val = 0                                                                               # Initiate a value to determine the y_max value of the plot
            for kk, key in enumerate(hist_keys[ax_count]):                                              # Looping through all keys in the history dict that will be shown on the current subplot axes
                if np.max(history[key]) > y_top_val:                                                    # If the maximum value in the array is larger than the current y_top_val ...
                    y_top_val = np.ceil(np.max(history[key])*10)/10                                     # ... y_top_val is updated and rounded to the nearest 0.1
                plt.plot(np.linspace(start=np.min(history["epoch_num"])-1, stop=np.max(history["epoch_num"])+1, num=len(history["epoch_num"])), history[key], color=colors[kk], linestyle="-", marker=".")   # Plot the data
            plt.legend([key for key in hist_keys[ax_count]], framealpha=0.5)                            # Create a legend for the subplot with the history keys displayed
            ax_count += 1                                                                               # Increase the subplot counter
        if y_top_val <= 0.05 and "lr" not in key.lower(): plt.ylim(bottom=-0.05, top=0.05)              # If the max y-value is super low, the limits are changed
        else: plt.ylim(top=y_top_val)                                                                   # Set the final, updated y_top_value as the y-top-limit on the current subplot axes
        if "lr" in key.lower():                                                                         # If we are plotting the learning rate ...
            plt.ylim(bottom=np.min(history[key]) / 2, top=np.ceil(np.max(history[key])*100)/100)        # ... the y_limits are rounded to the nearest 0.01
            plt.yscale('log')                                                                           # ... the y_scale will be logarithmic
    try: fig.savefig(os.path.join(config.OUTPUT_DIR, "Learning_curves.jpg"), bbox_inches="tight")       # Try and save the figure in the OUTPUR_DIR ...
    except: pass                                                                                        # ... otherwise simply skip saving the figure
    fig.tight_layout()                                                                                  # Make the figure tight_layout, which assures the subplots will be better spaced together
    fig.show() if FLAGS.display_images==True else plt.close(fig)                                        # If the user chose to not display the figure, the figure is closed
    return fig                                                                                          # The figure handle is returned

