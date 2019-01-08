import os
import math
import cv2
import numpy as np
import torch
from networks import Generator

class ImageGenerator:
    def __init__(self, arg, device):
        self.device = device
        # network
        self.nc = arg.nc
        self.nz = arg.nz
        self.init_size = arg.init_size
        self.size = arg.size
        self.G = Generator(nc=self.nc, nz=self.nz, size=self.size)
        # pre-growing
        total_stages = int(math.log2(self.size/self.init_size)) + 1
        for i in range(total_stages-1):
            self.G.grow_network()
            self.G.flush_network()
        for param in self.G.parameters():
            param.requires_grad_(False)
        # load checkpoint
        checkpoint = torch.load('checkpoint.tar')
        self.G.load_state_dict(checkpoint['G_EMA_state_dict'])
        self.G = self.G.to(self.device)
        self.G.eval()
    def normalize_tensor(self, tensor):
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)
    def generate(self, num):
        with torch.no_grad():
            for n in range(num):
                z = torch.FloatTensor(1, 512).normal_(0.0, 1.0).to(self.device)
                generate_data = self.G.forward(z)
                image = generate_data.to('cpu').squeeze()
                image = self.normalize_tensor(image)
                image = np.transpose(image.numpy(), (1,2,0))
                image *= 255
                image = image.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                filename = os.path.join('Images_Gen', 'ISIC_gen_{:07d}.jpg'.format(n))
                cv2.imwrite(filename, image)
                print(filename)
