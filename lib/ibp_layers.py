'''
Define layers that propagate interval bounds
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class IBPLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(IBPLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(
            torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # initialize weights
        torch.nn.init.xavier_uniform_(self.weight)
        if bias is not None:
            torch.nn.init.uniform_(self.bias)

    def forward(self, x_tuple):
        x, x_ub, x_lb = x_tuple
        # run a normal forward pass
        z = F.linear(x, self.weight, self.bias)

        # IBP
        if x_ub is not None and x_lb is not None:
            mu = (x_ub + x_lb) / 2
            r = (x_ub - x_lb) / 2
            mu = F.linear(mu, self.weight, self.bias)
            r = F.linear(r, self.weight.abs())
            z_lb = mu - r
            z_ub = mu + r
        else:
            z_lb = None
            z_ub = None

        return z, z_ub, z_lb


class IBPConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(IBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride, (0, 0), self.dilation,
                            self.groups)
        return F.conv2d(input, weight, bias, self.stride, self.padding,
                        self.dilation, self.groups)

    def forward(self, x_tuple):
        x, x_ub, x_lb = x_tuple
        # normal forward
        z = self.conv2d_forward(x, self.weight, self.bias)

        # IBP
        if x_ub is not None and x_lb is not None:
            mu = (x_ub + x_lb) / 2
            r = (x_ub - x_lb) / 2
            mu = self.conv2d_forward(mu, self.weight, self.bias)
            r = self.conv2d_forward(r, self.weight.abs(), None)
            z_lb = mu - r
            z_ub = mu + r
        else:
            z_lb = None
            z_ub = None

        return z, z_ub, z_lb


class IBPReLU(nn.ReLU):
    def __init__(self, inplace=False):
        super(IBPReLU, self).__init__()
        self.inplace = inplace

    def forward(self, x_tuple):
        x, x_ub, x_lb = x_tuple
        x = F.relu(x, inplace=self.inplace)
        if x_ub is not None and x_lb is not None:
            x_ub = F.relu(x_ub, inplace=self.inplace)
            x_lb = F.relu(x_lb, inplace=self.inplace)
        return x, x_ub, x_lb
