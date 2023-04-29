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

import torch.nn as nn

class CustomGroupedConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', n=1):
        super(CustomGroupedConv2D, self).__init__()
        self.n = n
        self.convs = nn.ModuleList()
        for i in range(n):
            self.convs.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels//n, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=1, bias=bias, padding_mode=padding_mode))

    def forward(self, x):
        outputs = []
        for i in range(self.n):
            outputs.append(self.convs[i](x[:, i*(x.shape[1]//self.n):(i+1)*(x.shape[1]//self.n), :, :]))
        return torch.cat(outputs, dim=1)

# özelleştirilmiş gruplanmış 2D evrişim katmanı
grouped_layer_custom = CustomGroupedConv2D(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=16, bias=True, padding_mode='zeros', n=16)

# orijinal ağırlıkların kopyalanması ve bölünmesi
for i in range(16):
    grouped_layer_custom.convs[i].weight.data = w_torch.data[i*(w_torch.shape[0]//16):(i+1)*(w_torch.shape[0]//16), :, :, :]
    grouped_layer_custom.convs[i].bias.data = b_torch.data[i*(b_torch.shape[0]//16):(i+1)*(b_torch.shape[0]//16)]
y2 = grouped_layer_custom(x)

# çıktı tensörü
print(y2)
print(y2.shape)
print(y.shape)
print(torch.allclose(y, y2)) 