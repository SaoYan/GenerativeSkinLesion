import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from networks import Generator, Discriminator
from layers import *
import torchvision.transforms as transforms
from data import preprocess_data_gan,  ISIC_GAN
from PIL import Image

if __name__ == "__main__":
    D = Discriminator()
    # D = nn.DataParallel(D, device_ids=[0,1]).to('cuda')
    opt_D = optim.Adam(D.parameters(), lr=0.001, betas=(0,0.99), eps=1e-8, weight_decay=0.)
    opt_D.step()
    print(opt_D.state)
    print('\n\n\n\n')

    # D.grow_network()
    # opt_D = optim.Adam(D.parameters(), lr=0.001, betas=(0,0.99), eps=1e-8, weight_decay=0.)
    # print(opt_D.state_dict()['param_groups'])
    # print('\n\n\n\n')
    #
    # D.flush_network()
    # opt_D = optim.Adam(D.parameters(), lr=0.001, betas=(0,0.99), eps=1e-8, weight_decay=0.)
    # print(opt_D.state_dict()['param_groups'])
    # print('\n\n\n\n')
