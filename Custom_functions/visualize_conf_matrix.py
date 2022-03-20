import numpy as np
import os
import matplotlib.pyplot as plt
from copy import deepcopy
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Function to normalize a confusion matrix
def normalizeConfMatrix(conf_matrix):
    conf_matrix_normal = deepcopy(conf_matrix)                                                      # First make a deep copy of the confusion matrix
    sum_rows = np.sum(conf_matrix_normal, axis=1)                                                   # Then compute the sum for each row of the matrix
    conf_matrix_normal = np.divide(conf_matrix_normal.T, sum_rows).T                                # Then divide all elements in each row by the sum of each row
    return conf_matrix_normal                                                                       # Return the normalized confusion matrix


# Function to plot confusion matrixes
def plot_confusion_matrix(labels, conf_train=None, conf_val=None, conf_test=None, save_folder=None):
    conf_matrixes = {}                                                                              # Initiate the dictionary to store the confusion matrixes in
    if conf_train is not None:                                                                      # If the training confusion matrix is not None ...
        conf_matrixes["Train"] = conf_train                                                         # ... it will be added to the dictionary
    if conf_val is not None:                                                                        # If the validation confusion matrix is not None ...
        conf_matrixes["Val"] = conf_val                                                             # ... it will be added to the dictionary
    if conf_test is not None:                                                                       # If the test confusion matrix is not None ...
        conf_matrixes["Test"] = conf_test                                                           # ... it will be added to the dictionary
    assert len(conf_matrixes) >= 1, "At least one confusion matrix must be plotted"                 # Assure at least one confusion matrix will be plotted
    fontdict_title = {'fontsize': 15}                                                               # Fontdict for the title of each axes subplot
    tit_ext = ""                                                                                    # As default the confusion matrix image will get no title prefix
    fmt = 'd'                                                                                       # As default the display format will be as integers
    val_max = int(255)                                                                              # As default the colorbar max value will be 255
    n_rows, n_cols = 2, len(conf_matrixes)                                                          # The figure will have 2 rows and a column pr confusion matrix
    fig, axs = plt.subplots(figsize=(int(np.ceil(n_cols*5.5)), n_rows*5), nrows=2, ncols=len(conf_matrixes))    # Create the figure
    for row_fig in range(n_rows):                                                                   # Iterate over both rows of the figure
        for kk, (split, conf_matrix) in enumerate(conf_matrixes.items()):                           # Iterate over all confusion matrixes in the conf_matrixes dictionary
            if row_fig+1 >= n_rows:                                                                 # If we are plotting on the second row ...
                conf_matrix = normalizeConfMatrix(conf_matrix)                                      # ... the confusion matrix must be normalized along rows (i.e. each true label)
                tit_ext = " normalized"                                                             # ... the title will get the prefix "normalized"
                fmt = '.2f'                                                                         # ... the numbered display format will be of decimals
                val_max = float(1)                                                                  # ... the colorbars max value will be a 1
            ax = axs[row_fig,kk]                                                                    # Set ax as the current axes
            im = ax.imshow(conf_matrix, cmap="jet", vmin=0, vmax=val_max)                           # Display the image
            ax.set(xticks=np.arange(0, len(labels)), yticks=np.arange(0, len(labels)),              # Set the tick parameters ...
                xticklabels=labels, yticklabels=labels, xlabel="Predicted labels", ylabel="True labels")    # ... including tick labels
            ax.set_title("{:s}".format(split+tit_ext+" confusion matrix"), fontdict_title)          # Set the title for the confusion matrix
            ax.tick_params(axis='both', labelrotation = 45, width=1, pad=0, labelsize=10)           # https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.axes.Axes.tick_params.html
            axe = make_axes_locatable(ax)                                                           # Extracts the current axes
            cax = axe.append_axes('right', size='5%', pad=0.05)                                     # Padding with some extra space next to the axes
            fig.colorbar(im, cax=cax, orientation='vertical')                                       # Creates the colorbar
            if np.unique(labels).shape[0] <= 10:                                                    # If there is less than ten classes in the dataset...
                treshVal = np.max(conf_matrix)/2                                                    # Treshold value for the color on the text => given as half the of the maximum value in the confusion matrix
                for row_cmatrix in range(conf_matrix.shape[0]):                                     # Loops over all rows in the matrix
                    for col in range(conf_matrix.shape[1]):                                         # Loops over all columns in the matrix
                        ax.text(col, row_cmatrix, format(conf_matrix[row_cmatrix, col], fmt),       # A matrix is indexed [rows, cols], while the cartesian coordinate system is [x, y].
                        ha="center", va="center",                                                   # Horizontal and vertical alignment of the text + center and fontsize
                        color="black" if conf_matrix[row_cmatrix, col] > treshVal else "white")     # If confMatrix[row, col] > treshVal the color is white, otherwise the text color is black
    if save_folder is not None: fig.savefig(os.path.join(save_folder, "Confusion matrixes.jpg"), bbox_inches="tight")
    fig.tight_layout()
    return fig

# conf_train=np.random.randint(low=0, high=255, size=(6,6)).astype(np.uint8)
# conf_val=np.random.randint(low=0, high=255, size=(6,6)).astype(np.uint8)
# conf_test=np.random.randint(low=0, high=255, size=(6,6)).astype(np.uint8)
# labels = ["Background", "Well", "Zona", "PV space", "Cell", "PN"]
# fig = plot_confusion_matrix(labels=labels, conf_train=conf_train, conf_val=conf_val, conf_test=conf_test)
# fig.show()

