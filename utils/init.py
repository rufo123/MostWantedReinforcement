"""
Module containing weight initialization functions for PyTorch models.
"""

from torch.nn import init


def weights_init_xavier(layer):
    """
    Initializes the weights of the layer using the Xavier initialization method.

    Args:
        layer: The layer whose weights need to be initialized.
    """
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.xavier_uniform_(layer.weight)
        init.zeros_(layer.bias)


def init_orthogonal_head(layer):
    """
    Initializes the weights of the layer using the orthogonal initialization method for the model 
        head.

    Args:
        layer: The layer whose weights need to be initialized.
    """
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(layer.weight, 0.01)
        init.zeros_(layer.bias)


def init_orthogonal_features(layer):
    """
    Initializes the weights of the layer using the orthogonal initialization method for the model
        features.

    Args:
        layer: The layer whose weights need to be initialized.
    """
    classname = layer.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        init.orthogonal_(layer.weight, 2 ** 0.5)
        init.zeros_(layer.bias)
