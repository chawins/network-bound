'''
Define layers that translate custom uncertainty sets to interval bound
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ibp_layers import IBPConv


class CustomLinear(nn.Linear):
    """Template"""

    def __init__(self, input_features, output_features, bias=True):
        super(CustomLinear, self).__init__(
            input_features, output_features, bias=bias)

    def forward(self, x, params):
        """
        x: torch.tensor
            input with size = (batch_size, self.weight.size(1))
        params['input_bound']: tuple
            lower and upper bounds of the input, (lb, ub). Set to None if do
            not want to bound input
        """

        z = F.linear(x, self.weight, self.bias)

        # TODO: main code should go here
        # Check LpLinear for example
        z_lb = None
        z_ub = None

        # intersect with bound that comes from the input domain
        input_bound = params['input_bound']
        if input_bound:
            x_ub = torch.zeros_like() + input_bound[1]
            x_lb = torch.zeros_like() + input_bound[0]
            mu = (x_ub + x_lb) / 2
            r = (x_ub - x_lb) / 2
            mu = F.linear(mu, self.weight, self.bias)
            r = F.linear(r, self.weight.abs())
            z_ub = torch.min(z_ub, mu + r)
            z_lb = torch.max(z_lb, mu - r)

        return z, z_ub, z_lb


# ============================================================================ #


class LpLinear(nn.Linear):
    """
    Implement linear layer that bounds its output in an interval given an Lp
    constraint on the perturbation of the input (i.e. norm(delta, p) <= eps)
    """

    def __init__(self, input_features, output_features, bias=True):
        super(LpLinear, self).__init__(
            input_features, output_features, bias=bias)

    def forward(self, x, params):
        """
        Return output of a linear layer when given input <x> and the interval
        bounds under an Lp-norm constraint on the perturbation,
        z = W(x + delta) + b where norm(delta, p) <= eps

        x: torch.tensor
            input with size = (batch_size, self.weight.size(1))
        params['p']: float
            p in p-norm that defines the uncertainty set
        params['epsilon']: float
            size of the uncertainty set (i.e. norm(delta, p) <= eps)
        params['input_bound']: tuple
            lower and upper bounds of the input, (lb, ub). Set to None if do
            not want to bound input
        """

        # nominal output
        z = F.linear(x, self.weight, self.bias)

        # get parameters
        p = params['p']
        eps = params['epsilon']
        q = p / (p - 1)

        # calculate lower and upper bounds (dual norm)
        w_norm = self.weight.norm(q, dim=1)
        z_ub = z + eps * w_norm
        z_lb = z - eps * w_norm

        # intersect with bound that comes from the input domain
        input_bound = params['input_bound']
        if input_bound:
            x_ub = torch.zeros_like(x) + input_bound[1]
            x_lb = torch.zeros_like(x) + input_bound[0]
            mu = (x_ub + x_lb) / 2
            r = (x_ub - x_lb) / 2
            mu = F.linear(mu, self.weight, self.bias)
            r = F.linear(r, self.weight.abs())
            z_ub = torch.min(z_ub, mu + r)
            z_lb = torch.max(z_lb, mu - r)

        return z, z_ub, z_lb


class LpConv(IBPConv):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(LpConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def forward(self, x, params):
        # nominal output
        z = self.conv2d_forward(x, self.weight, self.bias)

        # get parameters
        p = params['p']
        eps = params['epsilon']
        q = p / (p - 1)

        # calculate lower and upper bounds (dual norm)
        # TODO: figure out how to do this for conv layer
        w_norm = self.weight.norm(q, dim=1)
        z_ub = z + eps * w_norm
        z_lb = z - eps * w_norm

        # intersect with bound that comes from the input domain
        input_bound = params['input_bound']
        if input_bound:
            x_ub = torch.zeros_like() + input_bound[1]
            x_lb = torch.zeros_like() + input_bound[0]
            mu = (x_ub + x_lb) / 2
            r = (x_ub - x_lb) / 2
            mu = self.conv2d_forward(mu, self.weight, self.bias)
            r = self.conv2d_forward(r, self.weight.abs())
            z_ub = torch.min(z_ub, mu + r)
            z_lb = torch.max(z_lb, mu - r)

        return z, z_ub, z_lb
