"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The wights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn


torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)
print(f"y is : {y}\n")

# now write your custom layer
# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CustomGroupedConv2D, self).__init__()

        # store the arguments as class attributes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

        # calculate the number of output channels per group
        assert out_channels % groups == 0, "out_channels must be divisible by groups"
        self.out_channels_per_group = out_channels // groups

        # create the convolution layers in a ModuleList
        self.convs = nn.ModuleList()
        for i in range(groups):
            self.convs.append(nn.Conv2d(in_channels, self.out_channels_per_group, kernel_size, stride, padding, dilation, 1, bias))

    def forward(self, x):
        # split the input tensor along the channel dimension
        x_splits = torch.split(x, self.in_channels // self.groups, dim=1)

        # apply each convolution layer to its corresponding input tensor
        conv_outputs = []
        for i in range(self.groups):
            conv_outputs.append(self.convs[i](x_splits[i]))

        # concatenate the output tensors along the channel dimension
        y = torch.cat(conv_outputs, dim=1)

        return y


# Custom Grouped Conv2D Layer
grouped_layer_custom = CustomGroupedConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=16, bias=True)

# Copy the weights and bias from the original layer to the custom layer
group_num = grouped_layer_custom.groups
for i in range(group_num):
    grouped_layer_custom.convs[i].weight.data = w_torch.data[i*(w_torch.shape[0]//group_num):(i+1)*(w_torch.shape[0]//group_num), :, :, :]
    grouped_layer_custom.convs[i].bias.data = b_torch.data[i*(b_torch.shape[0]//group_num):(i+1)*(b_torch.shape[0]//group_num)]

y_custom = grouped_layer_custom(x)

# Test the output
print(f"y_custom is : {y_custom}\n")

# Print the shapes of the outputs
print(f"Shape of y is: {y.shape}")
print(f"Shape of y_custom is: {y_custom.shape}\n")

# Print if they are equal or not
print(f"Are they equal? : {torch.allclose(y, y_custom)}") 
print(f"Are they equal with 1e-3 tolerance? : {torch.allclose(y, y_custom, rtol=1e-3, atol=1e-3)}\n") 

# Print the difference
print(f"Difference between them is : {torch.sum(y_custom - y)}")