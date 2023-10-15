import torch.nn.functional as F
import torch.nn as nn
def get_out_channels(dataset_name):
    if dataset_name == "flood_net":
        return 2
    elif dataset_name == "Liveability":
        return 1

def get_loss_function(dataset_name, class_weights=None):
    if dataset_name == "flood_net":
        return nn.CrossEntropyLoss(weight=class_weights)
    return nn.MSELoss()