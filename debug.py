import torch
import torch.nn as nn
import torch.nn.functional as F
from networks import Generator, Discriminator
from layers import *
import torchvision.transforms as transforms
from data import preprocess_data_gan,  ISIC2017_GAN
from PIL import Image

if __name__ == "__main__":
    D = Discriminator()
    D = nn.DataParallel(D, device_ids=[0,1]).to('cuda')
    print(D)
    print('\n\n\n\n')
    D.module.grow_network()
    print(D)
    print('\n\n\n\n')
    # D.module.flush_network()
    # print(D)
    # print('\n\n\n\n')
    
