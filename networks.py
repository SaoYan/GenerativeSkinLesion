import math
import torch
import torch.nn as nn
import torchvision.models as models
from layers_ import *

#----------------------------------------------------------------------------
# helper functions

def conv_block(layers, in_features, out_features, kernel_size, stride, padding, pixel_norm):
    layers.append(EqualizedConv2d(in_features, out_features, kernel_size, stride, padding))
    layers.append(nn.LeakyReLU(0.2))
    if pixel_norm:
        layers.append(PixelwiseNorm())
    return layers

def deconv_block(layers, in_features, out_features, kernel_size, stride, padding, pixel_norm):
    layers.append(EqualizedDeconv2d(in_features, out_features, kernel_size, stride, padding))
    layers.append(nn.LeakyReLU(0.2))
    if pixel_norm:
        layers.append(PixelwiseNorm())
    return layers

def deepcopy_layers(module, layer_name):
    # copy the layer with name in "layer_name"
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name in layer_name:
            new_module.add_module(name, m)                 # construct new structure
            new_module[-1].load_state_dict(m.state_dict()) # copy weights
    return new_module

def deepcopy_exclude(module, exclude_name):
    # copy all the layers EXCEPT "layer_name"
    new_module = nn.Sequential()
    for name, m in module.named_children():
        if name not in exclude_name:
            new_module.add_module(name, m)                 # construct new structure
            new_module[-1].load_state_dict(m.state_dict()) # copy weights
    return new_module


#----------------------------------------------------------------------------
# generator

class Generator(nn.Module):
    def __init__(self, nc=3, nz=512, size=256, cond=False, num_classes=7):
        super(Generator, self).__init__()
        self.nc = nc # number of channels of the generated image
        self.nz = nz # dimension of the input noise
        self.size = size # the final size of the generated image
        self.cond = cond # true: conditional gan
        self.num_classes = num_classes
        self.stages = int(math.log2(self.size/4)) + 1 # the total number of stages (7 when size=256)
        self.current_stage = 1
        self.nf = lambda stage: min(int(8192 / (2.0 ** stage)), self.nz) # the number of channels in a particular stage
        self.module_names = []
        self.model = self.get_init_G()
        # one-hot embedding
        self.embedding = nn.Embedding(num_embeddings=self.num_classes, embedding_dim=self.num_classes)
        nn.init.eye_(self.embedding.weight)
        self.embedding.weight.requires_grad_(False)
    def get_init_G(self):
        model = nn.Sequential()
        model.add_module('stage_{}'.format(self.current_stage), self.first_block())
        model.add_module('to_rgb', self.to_rgb_block(self.nf(self.current_stage)))
        return model
    def first_block(self):
        layers = []
        ndim = self.nf(self.current_stage)
        if not self.cond:
            layers.append(PixelwiseNorm()) # normalize latent vectors before feeding them to the network
        layers = deconv_block(layers, in_features=self.nz+self.num_classes if self.cond else self.nz,
            out_features=ndim, kernel_size=4, stride=1, padding=0, pixel_norm=True)
        layers = conv_block(layers, in_features=ndim, out_features=ndim, kernel_size=3, stride=1, padding=1, pixel_norm=True)
        return  nn.Sequential(*layers)
    def to_rgb_block(self, ndim):
        return EqualizedConv2d(in_features=ndim, out_features=self.nc, kernel_size=1, stride=1, padding=0)
    def intermediate_block(self, stage):
        assert stage > 1, 'For intermediate blocks, stage should be larger than 1!'
        assert stage <= self.stages, 'Exceeding the maximum stage number!'
        layers = []
        layers.append(Upsample())
        layers = conv_block(layers, in_features=self.nf(stage-1), out_features=self.nf(stage), kernel_size=3, stride=1, padding=1, pixel_norm=True)
        layers = conv_block(layers, in_features=self.nf(stage), out_features=self.nf(stage), kernel_size=3, stride=1, padding=1, pixel_norm=True)
        return  nn.Sequential(*layers)
    def grow_network(self):
        self.current_stage += 1
        assert self.current_stage <= self.stages, 'Exceeding the maximum stage number!'
        print('\ngrowing Generator...\n')
        # copy the trained layers except "to_rgb"
        new_model = deepcopy_exclude(self.model, ['to_rgb'])
        # old block (used for fade in)
        old_block = nn.Sequential()
        old_to_rgb = deepcopy_layers(self.model, ['to_rgb'])
        old_block.add_module('old_to_rgb', old_to_rgb[-1])
        old_block.add_module('old_upsample', Upsample())
        # new block to be faded in
        new_block = nn.Sequential()
        inter_block = self.intermediate_block(self.current_stage)
        new_block.add_module('new_block', inter_block)
        new_block.add_module('new_to_rgb', self.to_rgb_block(self.nf(self.current_stage)))
        # add fade in layer
        new_model.add_module('concat_block', ConcatTable(old_block, new_block))
        new_model.add_module('fadein', Fadein())
        del self.model
        self.model = new_model
    def flush_network(self):
        # once the fade in is finished, remove the old block and preserve the new block
        print('\nflushing Generator...\n')
        new_block = deepcopy_layers(self.model.concat_block.layer2, ['new_block'])
        new_to_rgb = deepcopy_layers(self.model.concat_block.layer2, ['new_to_rgb'])
        # copy the previous trained layers (before ConcatTable and Fadein)
        new_model = nn.Sequential()
        new_model = deepcopy_exclude(self.model, ['concat_block', 'fadein'])
        # preserve the new block
        layer_name = 'stage_{}'.format(self.current_stage)
        new_model.add_module(layer_name, new_block[-1])
        new_model.add_module('to_rgb', new_to_rgb[-1])
        del self.model
        self.model = new_model
    def forward(self, x, labels=None):
        assert len(x.size()) == 2 or len(x.size()) == 4, 'Invalid input size!'
        if len(x.size()) == 2:
            x = x.view(x.size(0), x.size(1), 1, 1)
        input = x
        if self.cond:
            assert labels is not None, 'Missing labels for conditional GAN!'
            cond = self.embedding(labels.long().view(-1)) # N --> N x C
            cond = cond.view(cond.size(0), cond.size(1), 1, 1).repeat(1, 1, x.size(2), x.size(3))
            cond = cond.float().detach()
            input = torch.cat((cond, x), dim=1)
        return self.model(input)


#----------------------------------------------------------------------------
# discriminator

class Discriminator(nn.Module):
    def __init__(self, nc=3, nz=512, size=256, cond=False, num_classes=7):
        super(Discriminator, self).__init__()
        self.nc = nc # number of channels of the input
        self.nz = nz # dimension of G's input noise
        self.size = size # the size of the input image
        self.cond = cond # true: conditional gan
        self.num_classes = num_classes
        self.stages = int(math.log2(self.size/4)) + 1 # the total number of stages (7 when size=256)
        self.current_stage = self.stages
        self.nf = lambda stage: min(int(8192 / (2.0 ** stage)), self.nz) # the number of channels in a particular stage
        self.module_names = []
        self.model = self.get_init_D()
        self.dis = EqualizedLinear(in_features=self.nf(8-self.stages), out_features=1) # discriminate fake and real
        self.cls = None # classifier
        if self.cond:
            self.cls = EqualizedLinear(in_features=self.nf(8-self.stages), out_features=num_classes)
    def get_init_D(self):
        model = nn.Sequential()
        model.add_module('from_rgb', self.from_rgb_block(self.nf(8-self.current_stage)))
        model.add_module('stage_{}'.format(self.current_stage), self.last_block())
        return model
    def last_block(self):
        layers = []
        ndim = self.nf(8-self.stages)
        layers.append(MinibatchStddev()) # add minibatch stddev only at the last stage
        layers = conv_block(layers, in_features=ndim+1, out_features=ndim, kernel_size=3, stride=1, padding=1, pixel_norm=False)
        layers = conv_block(layers, in_features=ndim, out_features=ndim, kernel_size=4, stride=1, padding=0, pixel_norm=False)
        return nn.Sequential(*layers)
    def from_rgb_block(self, ndim):
        layers = []
        layers = conv_block(layers, in_features=self.nc, out_features=ndim, kernel_size=1, stride=1, padding=0, pixel_norm=False)
        return nn.Sequential(*layers)
    def intermediate_block(self, stage):
        assert stage >= 1, 'Stage number cannot be smaller than 1!'
        assert stage < self.stages, 'For intermediate layers, stage number should be smaller than {}!'.format(self.stages)
        layers = []
        layers = conv_block(layers, in_features=self.nf(8-stage), out_features=self.nf(8-stage), kernel_size=3, stride=1, padding=1, pixel_norm=False)
        layers = conv_block(layers, in_features=self.nf(8-stage), out_features=self.nf(7-stage), kernel_size=3, stride=1, padding=1, pixel_norm=False)
        layers.append(nn.AvgPool2d(kernel_size=2))
        return  nn.Sequential(*layers)
    def grow_network(self):
        self.current_stage -= 1
        assert self.current_stage >= 1, 'Stage number cannot be smaller than 1!'
        print('\ngrowing Discriminator...\n')
        # old block (used for fade in)
        old_block = nn.Sequential()
        old_from_rgb = deepcopy_layers(self.model, ['from_rgb'])
        old_block.add_module('old_downsample', nn.AvgPool2d(kernel_size=2))
        old_block.add_module('old_from_rgb', old_from_rgb[-1])
        # new block to be faded in
        new_block = nn.Sequential()
        inter_block = self.intermediate_block(self.current_stage)
        new_block.add_module('new_from_rgb', self.from_rgb_block(self.nf(8-self.current_stage)))
        new_block.add_module('new_block', inter_block)
        # add fade in layer
        new_model = nn.Sequential()
        new_model.add_module('concat_block', ConcatTable(old_block, new_block))
        new_model.add_module('fadein', Fadein())
        # copy the trained layers except "to_rgb"
        for name, module in self.model.named_children():
            if name != 'from_rgb':
                new_model.add_module(name, module)
                new_model[-1].load_state_dict(module.state_dict())
        del self.model
        self.model = new_model
    def flush_network(self):
        # once the fade in is finished, remove the old block and preserve the new block
        print('\nflushing Discriminator...\n')
        new_block = deepcopy_layers(self.model.concat_block.layer2, ['new_block'])
        new_from_rgb = deepcopy_layers(self.model.concat_block.layer2, ['new_from_rgb'])
        # preserve the new block
        new_model = nn.Sequential()
        layer_name = 'stage_{}'.format(self.current_stage)
        new_model.add_module('from_rgb', new_from_rgb[-1])
        new_model.add_module(layer_name, new_block[-1])
        # copy the previous trained layers (before ConcatTable and Fadein)
        for name, module in self.model.named_children():
            if name != 'concat_block' and name != 'fadein':
                new_model.add_module(name, module)
                new_model[-1].load_state_dict(module.state_dict())
        del self.model
        self.model = new_model
    def forward(self, x):
        assert len(x.size()) == 4, 'Invalid input size!'
        x = self.model(x)
        dis = self.dis(x)
        cls = None
        if self.cond:
            cls = self.cls(x)
        return dis, cls
