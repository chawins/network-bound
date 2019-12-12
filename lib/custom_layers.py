'''
Define layers that translate custom uncertainty sets to interval bound
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cvxopt import matrix, solvers
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


class IntLinear(nn.Linear):
    """
    Implement linear layer that bounds its output in an interval given an Lp
    constraint on the perturbation of the input (i.e. norm(delta, p) <= eps)
    """

    def __init__(self, input_features, output_features, bias=True):
        super(IntLinear, self).__init__(
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
        eps = params['epsilon']
        input_bound = params['input_bound']
        if input_bound:
            x_ub = torch.clamp(x + eps, input_bound[0], input_bound[1])
            x_lb = torch.clamp(x - eps, input_bound[0], input_bound[1])
        else:
            x_ub = x + eps
            x_lb = x - eps

        mu = (x_ub + x_lb) / 2
        r = (x_ub - x_lb) / 2
        mu = F.linear(mu, self.weight, self.bias)
        r = F.linear(r, self.weight.abs())
        z_lb = mu - r
        z_ub = mu + r

        return z, z_ub, z_lb


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
        if p == float("inf"):
            q = 1
        elif p > 1:
            q = p / (p - 1)
        elif p == 1:
            q = float("inf")

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


class EllipsLinear(nn.Linear):
    """
    Implement linear layer that bounds its output in an interval given an ellipsoidal
    constraint on the perturbation of the input (i.e. ||Q delta||_2 <= eps)
    """

    def __init__(self, input_features, output_features, bias=True):
        super(EllipsLinear, self).__init__(
            input_features, output_features, bias=bias)

    def forward(self, x, params):
        """
        Return output of a linear layer when given input <x> and the interval
        bounds under an ellipsoidal constraint on the perturbation,
        z = W(x + delta) + b where ||Q delta||_2 <= eps

        x: torch.tensor
            input with size = (batch_size, self.weight.size(1))
        params['Q']: torch.tensor
            Q that maps ellipsoidal uncertainty set to a epsilon ball
        params['epsilon']: float
            size of the uncertainty set (i.e. ||Q delta||_2 <= eps)
        params['input_bound']: tuple
            lower and upper bounds of the input, (lb, ub). Set to None if do
            not want to bound input
        """

        # nominal output
        z = F.linear(x, self.weight, self.bias)

        # get parameters
        Q = params['Q']
        eps = params['epsilon']

        # calculate lower and upper bounds (dual norm)
        w_norm = F.linear(self.weight, Q.inverse()).norm(2, dim=1)
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


class CardRelaxLinear(nn.Linear):
    """
    Implement linear layer that bounds its output in an interval given a
    cardinality constraint intersected with feasibility constraint on the
    perturbation of the input (i.e. ||delta||_0 <= eps & 0 <= x + delta <= 1)
    """

    def __init__(self, input_features, output_features, bias=True):
        super(CardRelaxLinear, self).__init__(
            input_features, output_features, bias=bias)

    def forward(self, x, params):
        """
        Return output of a linear layer when given input <x> and the interval
        bounds under a cardinality constraint intersected with feasibility
        constraint on the perturbation,
        z = W(x + delta) + b where ||delta||_0 <= eps & 0 <= x + delta <= 1
        x: torch.tensor
            input with size = (batch_size, self.weight.size(1))
        params['epsilon']: integer
            size of the uncertainty set (i.e. ||delta||_0 <= eps)
        params['input_bound']: tuple
            lower and upper bounds of the input, (lb, ub). Set to None if do
            not want to bound input
        """

        # nominal output
        z = F.linear(x, self.weight, self.bias)

        # get parameters
        eps = params['epsilon']

        # calculate lower and upper bounds (dual norm)
        w_norm = self.weight.norm(float('inf'), dim=1)
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


class CardLinear(nn.Linear):
    """
    Implement linear layer that bounds its output in an interval given a
    cardinality constraint intersected with feasibility constraint on the
    perturbation of the input (i.e. ||delta||_0 <= eps & 0 <= x + delta <= 1)
    by solving linear programs
    """

    def __init__(self, input_features, output_features, bias=True):
        super(CardLinear, self).__init__(
            input_features, output_features, bias=bias)

    def forward(self, x, params):
        """
        Return output of a linear layer when given input <x> and the interval
        bounds under a cardinality constraint intersected with feasibility
        constraint on the perturbation,
        z = W(x + delta) + b where ||delta||_0 <= eps & 0 <= x + delta <= 1
        x: torch.tensor
            input with size = (batch_size, self.weight.size(1))
        params['epsilon']: integer
            size of the uncertainty set (i.e. ||delta||_0 <= eps)
        params['input_bound']: tuple
            lower and upper bounds of the input, (lb, ub). Set to None if do
            not want to bound input
        """

        # nominal output
        z = F.linear(x, self.weight, self.bias)

        # get parameters
        eps = params['epsilon']

        # calculate lower and upper bounds
        n = np.shape(x)[1]  # get size of x
        row = np.shape(x)[0]
        z_lb = torch.zeros((row, n), device='cuda')
        z_ub = torch.zeros((row, n), device='cuda')
        solvers.options['show_progress'] = False

        for rn in range(1, row + 1):
            zr = z[rn - 1, :]
            xr = x[rn - 1, :]

            zr_lb = torch.zeros(n, device='cuda')
            for i in range(1, n + 1):
                Wi = self.weight[i - 1, :]
                # coefficients for objective function
                c = np.zeros(2 + 2 * n)
                c[0] = 1
                c = matrix(c)
                # coefficients for constraints expressed as a matrix
                A = np.zeros((2 + 4 * n, 2 + 2 * n))
                A[0][0] = -1
                A[0][1] = eps
                A[0][2:n + 2] = 1
                A[1][0] = -1
                A[2:n + 2][1] = -1
                temp = np.zeros((n, n))
                np.fill_diagonal(temp, -1)
                A[2:n + 2, 2:n + 2] = temp
                A[n + 2:2 * n + 2, 2:n + 2] = temp
                A[2 * n + 2:3 * n + 2, n + 2:2 * n + 2] = temp
                np.fill_diagonal(temp, 1)
                A[3 * n + 2:4 * n + 2, n + 2:2 * n + 2] = temp
                np.fill_diagonal(temp, Wi.detach().cpu().numpy())
                A[2:n + 2, n + 2:2 * n + 2] = temp
                A = matrix(A)
                # values of right-hand-side
                b = np.zeros(2 + 4 * n)
                b[0] = -1 * zr[i - 1]
                b[2 * n + 2:3 * n + 2] = xr.cpu()
                b[3 * n + 2:4 * n + 2] = 1 - xr.cpu()
                b = matrix(b)
                sol = solvers.lp(c, A, b)
                zr_lb[i - 1] = sol['primal objective']

            zr_ub = torch.zeros(n, device='cuda')
            for i in range(1, n + 1):
                Wi = self.weight[:, i - 1]
                # coefficients for objective function
                c = np.zeros(2 + 2 * n)
                c[0] = 1
                c = matrix(c)
                # coefficients for constraints expressed as a matrix
                A = np.zeros((2 + 4 * n, 2 + 2 * n))
                A[0][0] = -1
                A[0][1] = eps
                A[0][2:n + 2] = 1
                A[1][0] = -1
                A[2:n + 2][1] = -1
                temp = np.zeros((n, n))
                np.fill_diagonal(temp, -1)
                A[2:n + 2, 2:n + 2] = temp
                A[n + 2:2 * n + 2, 2:n + 2] = temp
                A[2 * n + 2:3 * n + 2, n + 2:2 * n + 2] = temp
                np.fill_diagonal(temp, 1)
                A[3 * n + 2:4 * n + 2, n + 2:2 * n + 2] = temp
                np.fill_diagonal(temp, -1 * Wi.detach().cpu().numpy())
                A[2:n + 2, n + 2:2 * n + 2] = temp
                A = matrix(A)
                # values of right-hand-side
                b = np.zeros(2 + 4 * n)
                b[0] = zr[i - 1]
                b[2 * n + 2:3 * n + 2] = xr.cpu()
                b[3 * n + 2:4 * n + 2] = 1 - xr.cpu()
                b = matrix(b)
                b = matrix(b)
                sol = solvers.lp(c, A, b)
                zr_ub[i - 1] = -1 * sol['primal objective']

            z_lb[rn - 1, :] = zr_lb
            z_ub[rn - 1, :] = zr_ub

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


class PermS1Linear(nn.Linear):

    def __init__(self, input_features, output_features, bias=True):
        super(PermS1Linear, self).__init__(
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

        # lb = torch.zeros(784)
        # ub = torch.zeros(784)
        # n = list(W.size())[0]
        # for i in range(0, n):
        #     Wi = torch.narrow(W, 0, i, 1)
        #     min_val = torch.mm(Wi, x)
        #     max_val = torch.mm(Wi, x)
        #     x_0 = x[0]
        #     for j in range(1, 2):  # bc we dont need to check switching 0th with 0th
        #         x_j = x[j]
        #         xj = x.clone()
        #         xj[0] = x_j
        #         xj[j] = x_0
        #         val = torch.mm(Wi, xj)
        #         min_val = torch.min(min_val, val)
        #         max_val = torch.max(max_val, val)
        #     lb[i] = min_val
        #     ub[i] = max_val

        # (batch_size, j, outdim)
        W1zj = x[:, 1:].unsqueeze(-1) @ self.weight[:, 0].unsqueeze(0)
        Wjzj = x[:, 1:].unsqueeze(-1) * self.weight[:, 1:].transpose(0, 1)
        Wjz1 = (self.weight[:, 1:].unsqueeze(-1) @
                x[:, 0].unsqueeze(0)).permute(2, 1, 0)
        W1z1 = x[:, 0].unsqueeze(-1) * self.weight[:, 0].unsqueeze(0)
        diff = W1zj - Wjzj + Wjz1

        z_lb = z + self.bias - W1z1 + diff.min(1)[0]
        z_ub = z + self.bias - W1z1 + diff.max(1)[0]

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
