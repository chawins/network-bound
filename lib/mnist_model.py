'''
Define MNIST models
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):

    def __init__(self, num_classes=10):
        super(BasicModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=6, stride=2, padding=3)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ============================================================================ #


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


class IBPLastLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(IBPLastLinear, self).__init__()
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
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        super(IBPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias, padding_mode)

    def conv2d_forward(self, input, weight, bias):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            (0, 0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

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


class IBPMedium(nn.Module):

    def __init__(self, num_classes=10):
        super(IBPMedium, self).__init__()
        self.conv1 = IBPConv(1, 32, kernel_size=3, stride=1, padding=0)
        self.relu1 = IBPReLU(inplace=True)
        self.conv2 = IBPConv(32, 32, kernel_size=4, stride=2, padding=0)
        self.relu2 = IBPReLU(inplace=True)
        self.conv3 = IBPConv(32, 64, kernel_size=3, stride=1, padding=0)
        self.relu3 = IBPReLU(inplace=True)
        self.conv4 = IBPConv(64, 64, kernel_size=4, stride=2, padding=0)
        self.relu4 = IBPReLU(inplace=True)
        self.fc1 = IBPLinear(1024, 512)
        self.relu5 = IBPReLU(inplace=True)
        self.fc2 = IBPLinear(512, 512)
        self.relu6 = IBPReLU(inplace=True)
        self.fc3 = IBPLinear(512, num_classes)

        for m in self.modules():
            if isinstance(m, IBPConv):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_tuple):
        x_tuple = self.conv1(x_tuple)
        x_tuple = self.relu1(x_tuple)

        x_tuple = self.conv2(x_tuple)
        x_tuple = self.relu2(x_tuple)

        x_tuple = self.conv3(x_tuple)
        x_tuple = self.relu3(x_tuple)

        x_tuple = self.conv4(x_tuple)
        x_tuple = self.relu4(x_tuple)

        x, x_ub, x_lb = x_tuple
        x = x.view(x.size(0), -1)
        if x_ub is not None and x_lb is not None:
            x_ub = x_ub.view(x_ub.size(0), -1)
            x_lb = x_lb.view(x_lb.size(0), -1)
        x_tuple = (x, x_ub, x_lb)

        x_tuple = self.fc1(x_tuple)
        x_tuple = self.relu5(x_tuple)

        x_tuple = self.fc2(x_tuple)
        x_tuple = self.relu6(x_tuple)

        z_tuple = self.fc3(x_tuple)

        return z_tuple


class IBPBasic(nn.Module):

    def __init__(self, num_classes=10):
        super(IBPBasic, self).__init__()
        self.fc1 = IBPLinear(784, 512)
        self.relu1 = IBPReLU(inplace=True)
        self.fc2 = IBPLinear(512, 512)
        self.relu2 = IBPReLU(inplace=True)
        self.fc3 = IBPLinear(512, num_classes)

        for m in self.modules():
            if isinstance(m, IBPConv):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x_tuple):
        x, x_ub, x_lb = x_tuple
        x = x.view(x.size(0), -1)
        if x_ub is not None and x_lb is not None:
            x_ub = x_ub.view(x_ub.size(0), -1)
            x_lb = x_lb.view(x_lb.size(0), -1)
        x_tuple = (x, x_ub, x_lb)

        x_tuple = self.fc1(x_tuple)
        x_tuple = self.relu1(x_tuple)

        x_tuple = self.fc2(x_tuple)
        x_tuple = self.relu2(x_tuple)

        z_tuple = self.fc3(x_tuple)

        return z_tuple
