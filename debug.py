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

    T = D.module.model.concat_block.to('cuda')
    F = D.module.model.fadein.to('cuda')
    transform = transforms.Compose([
        transforms.Resize((8,8), interpolation=Image.NEAREST),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    dataset = ISIC2017_GAN('train_gan.csv', shuffle=True, rotate=True, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=8)
    for i, data in enumerate(dataloader, 0):
        real_data = data
        real_data = real_data.mul(2.).sub(1.).to('cuda') # [0,1] --> [-1,1]
        pred = D(real_data)
        print(pred.size())
        print(i)
