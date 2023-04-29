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
print(y)

class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(CustomGroupedConv2D, self).__init__()
        self.groups = groups
        self.conv_list = nn.ModuleList()    # empty list to store the convolutions
        in_channels_per_group = in_channels // groups    # calculate the number of input channels per group
        out_channels_per_group = out_channels // groups    # calculate the number of output channels per group

        # split the weights and biases between the groups
        # assign weights and biases from grouped_layer to grouped_layer_custom
        for i in range(grouped_layer_custom.groups):
            # split the weight and bias tensors for each group
            weight_i = w_torch[i*(64//grouped_layer_custom.groups): (i+1)*(64//grouped_layer_custom.groups)]
            bias_i = b_torch[i*(128//grouped_layer_custom.groups): (i+1)*(128//grouped_layer_custom.groups)]
            
            grouped_layer_custom.conv_list[i].weight = nn.Parameter(weight_i.clone())
            grouped_layer_custom.conv_list[i].bias = nn.Parameter(bias_i.clone())

        
    def forward(self, x):
        # split the input tensor along the channel dimension
        x_list = torch.chunk(x, self.groups, dim=1)
        out_list = []

        # apply convolution to each group and concatenate the results
        for i in range(self.groups):
            out_list.append(self.conv_list[i](x_list[i]))
        out = torch.cat(out_list, dim=1)
        return out
# create an instance of CustomGroupedConv2D
grouped_layer_custom = CustomGroupedConv2D(64, 128, kernel_size=3, stride=1, padding=1, groups=16, bias=True)


# assign weights and biases from grouped_layer to grouped_layer_custom
grouped_layer_custom.conv_list[0].weight = nn.Parameter(grouped_layer.weight.data.clone())
grouped_layer_custom.conv_list[0].bias = nn.Parameter(grouped_layer.bias.data.clone())


y_custom = grouped_layer_custom(x)



#
print(y_custom.shape)
print(y.shape)
print(torch.allclose(y, y_custom))  # expect True

        
