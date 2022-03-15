# Import libraries
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from natsort import natsorted
from visualize_vitrolife_batch import extractNumbersFromString

# config = cfg
# config.OUTPUT_DIR = "/mnt/c/Users/Nico-/Documents/Python_Projects/MaskFormer/output_vitrolife_13_25_14MAR2022"


# Define a function to compute the moving average of an input array or list
def mov_avg_array(inp_array, mov_of_last_n_elements=4, output_last_n_elements=1):                       # Define a function to compute the moving average of an array or a list
    assert output_last_n_elements <= mov_of_last_n_elements, "The moving average can't be outputted for more values than it is being calculated for"
    if mov_of_last_n_elements > len(inp_array): mov_of_last_n_elements = len(inp_array)                 # If the list/array isn't as long as the wanted moving-average value, the n is lowered
    used_array_part = inp_array[-mov_of_last_n_elements:]                                               # Extract the last mov_of_last_n_elements from the list to compute the moving average for
    used_array_cumsum = np.cumsum(used_array_part)                                                      # Compute the cumulated sum for the used array part
    used_array_mov_avg = np.divide(used_array_cumsum, np.arange(1,1+mov_of_last_n_elements))            # Compute the moving average of the used array part
    return used_array_mov_avg[-output_last_n_elements:]                                                 # Output the last output_last_n_elements of the moving average array 


# Define a function to load the metrics.json in each output directory
def load_json_metrics(config, data_split="train"):
    metrics_list = natsorted([x for x in os.listdir(config["OUTPUT_DIR"]) if x.startswith("{:s}_metrics".format(data_split))    # Loop through all files in the output directory ...
            and x.endswith(".json") and not np.isnan(extractNumbersFromString(x))])                     # ... and gather all the split_metrics_x.json files, where x=epoch_number and split=run_mode
    metrics = {"epoch_num": list()}                                                                     # Initiate the dictionary to store all the files
    for epoch_idx, metric_file in enumerate(metrics_list):
        for it_idx, line in enumerate(open(os.path.join(config.OUTPUT_DIR, metric_file))):
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
            mov_avg_val.append(mov_avg_array(inp_array=key_val, mov_of_last_n_elements=25, output_last_n_elements=1).item())    # Compute the next mov_avg val for the last 10 elements
        metrics[key] = mov_avg_val                                                                      # Assign the newly computed moving average of the dict[key]->values to the dictionary
    return metrics                                                                                      # Return the moving average value dictionary

# metrics = load_json_metrics(config=config)



# Function to display learning curves
def show_history(config, FLAGS, metrics_train, metrics_eval, history=None):                             # Define a function to visualize the learning curves
    # Mean intersection-over-union averaged across classes (mIoU)
    # Frequency Weighted IoU (fwIoU)
    # Mean pixel accuracy averaged across classes (mACC)
    # Pixel Accuracy (pACC)


    train_history = load_json_metrics(config=config, data_split="train")                                # Load the metrics into the history dictionary
    val_history = load_json_metrics(config=config, data_split="val")                                    # Load the metrics into the history dictionary
    if history == None: history = {}                                                                    # Initiate the history dictionary that will be used
    for key in train_history: history["train_"+key] = train_history[key]                                # Give all training loss metrics the prefix "train"
    for key in val_history: history["val_"+key] = val_history[key]                                      # Give all validation loss metrics the prefix "val"
    for key in metrics_train.keys():                                                                    # Iterate over all keys in the history
        if "train_"+key not in history: history["train_"+key] = list()                                  # If the given key doesn't exist add the key with ...
        if "val_"+key not in history: history["val_"+key] = list()                                      # ... the data_split mode prefix as an empty list
        history["train_"+key].append(metrics_train[key])                                                # Append the current key-value from the metrics_train
        history["val_"+key].append(metrics_eval[key])                                                   # Append the current key-value from the metrics_val
    loss_total = [key for key in history.keys() if "total_loss" in key.lower()]                         # Find all keys with loss_ce
    learn_rate = [key for key in history.keys() if "lr" in key.lower() and "val" not in key.lower()]    # Find the training learning rate
    loss_ce = [key for key in history.keys() if "loss_ce" in key.lower()]                               # Find all keys with loss_ce
    loss_dice = [key for key in history.keys() if "loss_dice" in key.lower()]                           # Find all keys with loss_dice
    loss_mask = [key for key in history.keys() if "loss_mask" in key.lower()]                           # Find all keys with loss_mask
    m_f_IoU = [key for key in history.keys() if key.endswith("IoU")]                                    # Find the mIoU and fIoU keys
    IoU_class = [key for key in history.keys() if "IoU-" in key and not key.endswith("IoU")]            # Find the class specific IoU keys
    acc_keys = [key for key in history.keys() if key.endswith("ACC")]                                   # Find the keys with the pixel accuracy
    acc_class = [key for key in history.keys() if "ACC-" in key and not key.endswith("ACC")]            # Find the class specific pixel accuracies
    hist_keys = [loss_total, m_f_IoU, loss_ce, loss_dice, loss_mask, acc_keys, learn_rate, acc_class, IoU_class]    # Combine the key-lists into a list of lists
    for key_idx, hist_key_list in enumerate(hist_keys):                                                 # Iterate over all the found key-lists
        keys_to_remove = [key for key in hist_key_list if "background" in key.lower() or not np.isnan(extractNumbersFromString(key))]   # If the key_name contains "background" or a number ...
        [hist_keys[key_idx].remove(key) for key in keys_to_remove]                                      # ..., i.e. the [0,1,2,3,4] losses (what ever those are) then remove that key from the list
    ax_titles = ["Total_loss", "mIoU and fIoU", "Loss_CE", "Loss_dice", "Loss_mask",                    # Create titles and ylabels ...
                "mACC and pACC", "Pixel accuracy per class", "Learning_rate", "IoU per class"]          # ... for the axes
    colors = ["blue", "red", "black", "green", "magenta", "cyan", "yellow", "deeppink", "purple",       # Create colors for ... 
                "peru", "darkgrey", "gold", "springgreen", "orange", "crimson", "lawngreen"]            # ... the line plots
    fig = plt.figure(figsize=(16,18.5))                                                                 # Create the figure
    n_rows, n_cols, ax_count = 4, (2,3,2,2), 0                                                          # Initiate values for the number of rows and columns
    if FLAGS.use_per_pixel_baseline==True: n_rows, n_cols = 3, (2,2,2)                                  # If we train with the per_pixel_classification method, only CCE loss is calculated in the total_loss metric
    for row in range(n_rows):                                                                           # Loop through all rows
        for col in range(n_cols[row]):                                                                  # Loop through all columns in the current row
            plt.subplot(n_rows, n_cols[row], 1+row*n_cols[row]+col)                                     # Create a new subplot
            plt.xlabel(xlabel="Epoch #")                                                                # Set correct xlabel
            plt.ylabel(ylabel=ax_titles[ax_count].replace("_", " "))                                    # Set correct ylabel
            plt.grid(True)                                                                              # Activate the grid on the plot
            plt.xlim(left=0, right=np.max(history["train_epoch_num"]))                                  # Set correct xlim
            plt.title(label=ax_titles[ax_count].replace("_", " "))                                      # Set plot title
            y_top_val = 0                                                                               # Initiate a value to determine the y_max value of the plot
            for kk, key in enumerate(sorted(hist_keys[ax_count], key=str.lower)):                       # Looping through all keys in the history dict that will be shown on the current subplot axes
                if np.max(history[key]) > y_top_val:                                                    # If the maximum value in the array is larger than the current y_top_val ...
                    y_top_val = np.ceil(np.max(history[key])*10)/10                                     # ... y_top_val is updated and rounded to the nearest 0.1
                plt.plot(np.linspace(start=np.min(history["train_epoch_num"])-(0 if any(["IoU" in key or "ACC" in key for key in hist_keys[ax_count]]) else 1), stop=np.max(history["train_epoch_num"]),   # Plot the data, using a linspace argument for the x-axis ...
                        num=len(history[key])), history[key], color=colors[kk], linestyle="-", marker=".")  # ... and the data itself for the y-axis
            plt.legend(sorted([key.replace("_"," ") for key in hist_keys[ax_count]], key=str.lower),    # Create a legend for the subplot with ...
                    framealpha=0.5, loc="best" if len(hist_keys[ax_count])<4 else "upper left")         # ... the history keys displayed
            ax_count += 1                                                                               # Increase the subplot counter
        if y_top_val <= 0.05 and "lr" not in key.lower(): plt.ylim(bottom=-0.05, top=0.05)              # If the max y-value is super low, the limits are changed
        else: plt.ylim(top=y_top_val)                                                                   # Set the final, updated y_top_value as the y-top-limit on the current subplot axes
        if "lr" in key.lower():                                                                         # If we are plotting the learning rate ...
            plt.ylim(bottom=np.min(history[key])*0.9, top=np.max(history[key])*1.075)                   # ... the y_limits are changed
            plt.yscale('log')                                                                           # ... the y_scale will be logarithmic
    try: fig.savefig(os.path.join(config.OUTPUT_DIR, "Learning_curves.jpg"), bbox_inches="tight")       # Try and save the figure in the OUTPUR_DIR ...
    except: pass                                                                                        # ... otherwise simply skip saving the figure
    # fig.tight_layout()                                                                                  # Make the figure tight_layout, which assures the subplots will be better spaced together
    fig.show() if FLAGS.display_images==True else plt.close(fig)                                        # If the user chose to not display the figure, the figure is closed
    return history                                                                                      # The history dictionary is returned


history = show_history(config=cfg, FLAGS=FLAGS, metrics_train=eval_train_results["sem_seg"], metrics_eval=eval_val_results["sem_seg"])




