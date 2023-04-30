"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx2pytorch import ConvertModel
import torchinfo

torch.manual_seed(8)
np.random.seed(8)

# Load the ONNX model
onnx_model = onnx.load('model/model.onnx')

input_shape = [dim.dim_value for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]
output_shape = [dim.dim_value for dim in onnx_model.graph.output[0].type.tensor_type.shape.dim]

# Convert onnx model to pytorch model
model = ConvertModel(onnx_model)

# Look at model architecture before making changes
print("Before")
torchinfo.summary(model, (input_shape[1:]), batch_dim = 0, col_names = ("input_size", "output_size"), verbose = 1)
# print(model)

for layer in model.modules():
    # Initialize the convolutions layer with uniform xavier and initialize all biases with zeros
    if isinstance(layer, nn.Conv2d):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)
        
    # Initialize the linear layer with a normal distribution (mean=0.0, std=1.0) 
    # and initialize all biases with zeros
    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        nn.init.zeros_(layer.bias)

# Add batch normalization layer after every conv2d layer
def add_bn_after_conv2d(model):
    for name, layer in model.named_children():
        if isinstance(layer, nn.Conv2d):
            # Get the number of output channels of the convolutional layer
            out_channels = layer.out_channels
            # Add a batch normalization layer after the convolutional layer
            bn_layer = nn.BatchNorm2d(num_features=out_channels)
            # Insert the batch normalization layer after the convolutional layer
            model._modules[name] = nn.Sequential(layer, bn_layer)
        else:
            # Recursively apply the function to child modules
            add_bn_after_conv2d(layer)
    return model

model = add_bn_after_conv2d(model)
     
# Look at model architecture after making changes  
print("After----------------------")
torchinfo.summary(model, (input_shape[1:]), batch_dim = 0, col_names = ("input_size", "output_size"), verbose = 1)
# print(model)

