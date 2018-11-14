import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class equalized_conv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding):
        super(equalized_conv2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, stride, padding, bias=True)
        nn.init.kaiming_normal_(self.conv.weight, a=calculate_gain('conv2d'))
        nn.init.constant_(self.conv.bias, val=0.)
        # conv_w = self.conv.weight.data.clone()
        self.scale = self.conv.weight.data.pow(2.).mean().sqrt()
        self.conv.weight.data.copy_(self.conv.weight.data / self.scale)
    def forward(self, x):
        return self.conv(x.mul(self.scale))

class equalized_deconv2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, stride, padding):
        super(equalized_deconv2d, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_features, out_features, kernel_size, stride, padding, bias=True)
        nn.init.kaiming_normal(self.deconv.weight, a=calculate_gain('conv2d'))
        nn.init.constant_(self.deconv.bias, val=0.)
        # deconv_w = self.deconv.weight.data.clone()
        self.scale = self.deconv.weight.data.pow(2.).mean().sqrt()
        self.deconv.weight.data.copy_(self.deconv.weight.data/self.scale)
    def forward(self, x):
        return self.deconv(x.mul(self.scale))

class equalized_linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(equalized_linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal(self.linear.weight, a=calculate_gain('linear'))
        nn.init.constant_(self.linear.bias, val=0.)
        # linear_w = self.linear.weight.data.clone()
        self.scale = self.linear.weight.data.pow(2.).mean().sqrt()
        self.linear.weight.data.copy_(self.linear.weight.data/self.scale)

    def forward(self, x):
        return self.linear(x.mul(self.scale))
