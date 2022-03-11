# Add the MaskFormer directory to PATH
def register_paths():
    import os                                                                   # Used to navigate the folder structure in the current os
    import sys                                                                  # Used to control the PATH variable
    MaskFormer_dir = os.path.join("/mnt", "c", "Users", "Nico-", "Documents", "Python_Projects", "MaskFormer")                                                              # Home WSL
    if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("C:\\", MaskFormer_dir.split(os.path.sep, 1)[1])                                                    # Home windows computer
    if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Repositories", "MaskFormer")  # Larac server
    if not os.path.isdir(MaskFormer_dir): MaskFormer_dir = os.path.join("/mnt", "home_shared", MaskFormer_dir.split(os.path.sep, 2)[2])                                     # Balder server
    assert os.path.isdir(MaskFormer_dir), "The MaskFormer directory doesn't exist in the chosen location"
    sys.path.append(MaskFormer_dir)                                             # Add MaskFormer directory to PATH
    sys.path.append(os.path.join(MaskFormer_dir, "Custom_functions"))           # Add Custom_functions directory to PATH
    sys.path.append(os.path.join(MaskFormer_dir, "tools"))                      # Add the tools directory to PATH

    # Add the environmental variable DETECTRON2_DATASETS
    dataset_dir = os.path.join("/mnt", "c", "Users", "Nico-", "OneDrive - Aarhus Universitet", "Biomedicinsk Teknologi", "5. semester", "Speciale", "Datasets")             # Home WSL
    if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("C:\\", dataset_dir.split(os.path.sep,1)[1])                                                              # Home windows computer
    if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/home", "neal", "Panoptic_segmentation_using_deep_neural_networks", "Datasets")                          # Larac server
    if not os.path.isdir(dataset_dir): dataset_dir = os.path.join("/mnt", "home_shared", dataset_dir.split(os.path.sep, 2)[2])                                              # Balder server
    assert os.path.isdir(dataset_dir), "The dataset directory doesn't exist in the chosen location"
    os.environ["DETECTRON2_DATASETS"] = dataset_dir
