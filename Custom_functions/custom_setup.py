# Import other libraries and custom functions
from registerMaskformerDataset_paths import register_paths
register_paths()
from GPU_memory_ranked_assigning import assign_free_gpus                    # Function to assign the running process to a specified number of GPUs ranked by memory availability
from register_vitrolife_dataset import register_vitrolife_data_and_metadata_func    # Import function to register the vitrolife datasets in Detectron2 
from create_custom_config import createVitrolifeConfiguration               # Function to create the custom configuration used for the training with Vitrolife dataset

### CUSTOM FUNCTIONS
# Alter the FLAGS input arguments
def changeFLAGS(FLAGS):
    if FLAGS.num_gpus != FLAGS.gpus_used: FLAGS.num_gpus = FLAGS.gpus_used  # As there are two input arguments where the number of GPUs can be assigned, the gpus_used argument is superior
    if "vitrolife" in FLAGS.dataset_name.lower() : FLAGS.num_gpus = 1       # Working with the Vitrolife dataset can only be done using a single GPU for some weird reason...
    if FLAGS.eval_only != FLAGS.inference_only: FLAGS.eval_only = FLAGS.inference_only  # As there are two inputs where "eval_only" can be set, inference_only is the superior
    return FLAGS

# Define the main function used to send input arguments. Just return the altered FLAGS arguments as a namespace variable
def main(FLAGS):
    return changeFLAGS(FLAGS)

# Assign to GPU, register dataset and create the config used
def main_setup(FLAGS):
    FLAGS = main(FLAGS=FLAGS)                                                   # Get the FLAGS input arguments
    assign_free_gpus(max_gpus=FLAGS.num_gpus)                                   # Assigning the running script to the selected amount of GPU's with the largest memory available
    register_vitrolife_data_and_metadata_func(debugging=FLAGS.debugging)        # Register the vitrolife dataset
    cfg = createVitrolifeConfiguration(FLAGS=FLAGS)                             # Create the custom configuration used to e.g. build the model
    print(__name__)
    return FLAGS, cfg