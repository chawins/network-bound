'''
Define MNIST models
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.ibp_layers import IBPConv, IBPLinear, IBPReLU


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


class IBPMedium(nn.Module):

    def __init__(self, num_classes=10):
        super(IBPMedium, self).__init__()
        self.num_classes = num_classes
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

    def get_bound(self, x_tuple, targets):
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

        # last layer
        x, x_ub, x_lb = x_tuple
        batch_size = x.size(0)
        eye = torch.eye(self.num_classes, device=x.device)
        e = eye[targets]
        diff = torch.zeros(batch_size, device=x.device) + 1e5
        for i in range(self.num_classes):
            # minimize c^T x
            c = (e - eye[i]) @ self.fc3.weight
            z = (c * ((c > 0).float() * x_lb + (c <= 0).float() * x_ub)).sum(1)
            z += (e - eye[i]) @ self.fc3.bias
            idx = np.where((z < diff).detach().cpu().numpy())[0]
            diff[idx] = z[idx]
        return diff


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


class IBPMediumCustom(nn.Module):
    """Same as IBPMedium but first layer can take customized uncertainty set"""

    def __init__(self, first_layer, params, num_classes=10):
        super(IBPMediumCustom, self).__init__()
        self.num_classes = num_classes
        self.params = params
        self.conv1 = first_layer(1, 32, kernel_size=3, stride=1, padding=0)
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
            if isinstance(m, first_layer):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, IBPConv):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, params=None):
        if params:
            x_tuple = self.conv1(x, params)
        else:
            x_tuple = self.conv1(x, self.params)
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

    def get_bound(self, x, targets, params=None):
        if params:
            x_tuple = self.conv1(x, params)
        else:
            x_tuple = self.conv1(x, self.params)
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

        # last layer
        x, x_ub, x_lb = x_tuple
        batch_size = x.size(0)
        eye = torch.eye(self.num_classes, device=x.device)
        e = eye[targets]
        diff = torch.zeros(batch_size, device=x.device) + 1e5
        for i in range(self.num_classes):
            # minimize c^T x
            c = (e - eye[i]) @ self.fc3.weight
            z = (c * ((c > 0).float() * x_lb + (c <= 0).float() * x_ub)).sum(1)
            z += (e - eye[i]) @ self.fc3.bias
            idx = np.where((z < diff).detach().cpu().numpy())[0]
            diff[idx] = z[idx]
        return diff


class IBPSmallCustom(nn.Module):
    """Same as IBPMedium but first layer can take customized uncertainty set"""

    def __init__(self, first_layer, params, num_classes=10):
        super(IBPSmallCustom, self).__init__()
        self.num_classes = num_classes
        self.params = params
        self.fc1 = first_layer(784, 2000)
        self.relu1 = IBPReLU(inplace=True)
        self.fc2 = IBPLinear(2000, 2000)
        self.relu2 = IBPReLU(inplace=True)
        self.fc3 = IBPLinear(2000, 512)
        self.relu3 = IBPReLU(inplace=True)
        self.fc4 = IBPLinear(512, num_classes)

        for m in self.modules():
            if isinstance(m, first_layer):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, IBPLinear):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x, params=None):
        x = x.view(x.size(0), -1)
        if params:
            x_tuple = self.fc1(x, params)
        else:
            x_tuple = self.fc1(x, self.params)
        x_tuple = self.relu1(x_tuple)

        x_tuple = self.fc2(x_tuple)
        x_tuple = self.relu2(x_tuple)

        x_tuple = self.fc3(x_tuple)
        x_tuple = self.relu3(x_tuple)

        z_tuple = self.fc4(x_tuple)
        return z_tuple

    def get_bound(self, x, targets, params=None):
        x = x.view(x.size(0), -1)
        if params:
            x_tuple = self.fc1(x, params)
        else:
            x_tuple = self.fc1(x, self.params)
        x_tuple = self.relu1(x_tuple)

        x_tuple = self.fc2(x_tuple)
        x_tuple = self.relu2(x_tuple)

        x_tuple = self.fc3(x_tuple)
        x_tuple = self.relu3(x_tuple)

        # last layer
        x, x_ub, x_lb = x_tuple
        batch_size = x.size(0)
        eye = torch.eye(self.num_classes, device=x.device)
        e = eye[targets]
        diff = torch.zeros(batch_size, device=x.device) + 1e5
        for i in range(self.num_classes):
            # minimize c^T x = z_label - z_i
            c = (e - eye[i]) @ self.fc4.weight
            z = (c * ((c > 0).float() * x_lb + (c <= 0).float() * x_ub)).sum(1)
            z += (e - eye[i]) @ self.fc4.bias
            idx = np.where((z < diff).detach().cpu().numpy())[0]
            diff[idx] = z[idx]
        return diff
