import math
import torch
import torch.nn as nn
from layers import *

#----------------------------------------------------------------------------
# Auxiliary functions.
# reference: https://github.com/nashory/pggan-pytorch/blob/master/network.py

class conv_block(layers, in_features, out_features, kernel_size, stride, padding, pixel_norm):
    layers.append(EqualizedConv2d(in_features, out_features, kernel_size, stride, padding))
    layers.append(nn.LeakyReLU(0.2))
    if pixel_norm:
        layers.append(PixelwiseNorm())
    return layers

def deepcopy_module(module, target):
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name == target:
            new_module.add_module(name, m)                          # make new structure and,
            new_module[-1].load_state_dict(m.state_dict())         # copy weights
    return new_module

def soft_copy_param(target_link, source_link, tau):
    ''' soft-copy parameters of a link to another link. '''
    target_params = dict(target_link.named_parameters())
    for param_name, param in source_link.named_parameters():
        target_params[param_name].data = target_params[param_name].data.mul(1.0-tau)
        target_params[param_name].data = target_params[param_name].data.add(param.data.mul(tau))

def get_module_names(model):
    names = []
    for key, val in model.state_dict().iteritems():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names

#----------------------------------------------------------------------------
# Generator.
# reference 1: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L144
# reference 2: https://github.com/nashory/pggan-pytorch/blob/master/network.py

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self, nc=3, nz=512, size=256).__init__()
        self.nc = nc # number of channels of the generated image
        self.nz = nz # dimension of the input noise
        self.size = size # the final size of the generated image
        self.stages = int(math.log2(self.size/4)) + 1 # the total number of stages (7 when size=256)
        self.nf = lambda stage: min(int(8192 / (2.0 ** stage)), 512) # the number of channels in a particular stage
        self.layer_name = None
        self.module_names = []
        self.model = self.get_init_G()
    def get_init_G(self):
        model = nn.Sequential()
        model.add_module('stage_1', self.first_block())
        model.add_module('to_rgb', self.to_rgb_block(self.nf(1)))
        self.module_names = get_module_names(model)
        return model
    def first_block(self):
        layers = []
        ndim = self.nf(1)
        layers.append(pixelwise_norm_layer()) # normalize latent vectors before feeding them to the network
        layers = conv_block(layers, in_features=self.nz, out_features=ndim, kernel_size=4, stride=1, padding=3, pixel_norm=True)
        layers = conv_block(layers, in_features=ndim, out_features=ndim, kernel_size=3, stride=1, padding=1, pixel_norm=True)
        return  nn.Sequential(*layers)
    def to_rgb_block(self, ndim):
        return EqualizedConv2d(in_features=ndim, out_features=self.nc, kernel_size=1, stride=1, padding=0)
    def intermediate_block(self, stage):
        assert stage > 1, 'For intermediate blocks, stage should be larger than 1!'
        assert stage <= self.stages, 'Exceeding the maximum stage number!'
        layer_name = 'stage_{}'.format(stage)
        layers = []
        layers.append(Upsample())
        layers = conv_block(layers, in_features=self.nf(stage-1), out_features=self.nf(stage), kernel_size=3, stride=1, padding=1, pixel_norm=True)
        layers = conv_block(layers, in_features=self.nf(stage), out_features=self.nf(stage), kernel_size=3, stride=1, padding=1, pixel_norm=True)
        return  nn.Sequential(*layers), layer_name
    def grow_network(self, stage):
    def flush_network(self):
    def freeze_layers(self):
    def forward(self, x):
        assert (len(x.size()) != 2) | (len(x.size()) != 4), 'Invalid input size!'
        if len(x.size() == 2):
            x = x.view(x.size(0), x.size(1), 1, 1)
        return self.model(x)

#----------------------------------------------------------------------------
# Discriminator.
