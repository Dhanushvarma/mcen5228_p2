"""
Utility functions for model initialization and parameter counting.
"""

import torch
from torch.nn import init


def init_weights(net, init_type="normal", gain=0.02):
    """
    Initialize network weights.
    
    Args:
        net: Neural network model
        init_type: Type of initialization ('normal', 'xavier', 'kaiming', 'orthogonal')
        gain: Scaling factor for weight initialization
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    f"Initialization method [{init_type}] is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print(f"Initializing network with {init_type} initialization")
    net.apply(init_func)


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Total number of trainable parameters
    """
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params = 1
            for s in p.size():
                num_params *= s
            total_params += num_params
    return total_params