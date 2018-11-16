import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.autograd as autograd
from PIL import Image
from networks import Generator, Discriminator
from data import preprocess_data_gan,  ISIC2017_GAN
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1,3,4]

parser = argparse.ArgumentParser(description="PGAN-Skin-Lesion")

parser.add_argument("--preprocess", action='store_true')
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--unit_epoch", type=int, default=30)
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')

opt = parser.parse_args()

#----------------------------------------------------------------------------
# Trainer
# reference: https://github.com/nashory/pggan-pytorch/blob/master/trainer.py

class trainer:
    def __init__(self, nc=3, nz=512, final_size=256):
        self.nc = nc # number of channels of the generated image
        self.nz = nz # dimension of the input noise
        self.final_size = final_size # the final size of the generated image
        self.current_size = 4
        self.writer = SummaryWriter(opt.outf)
        self.num_aug = 5
        self.init_trainer()
    def init_trainer(self):
        # networks
        self.G = Generator(nc=self.nc, nz=self.nz, size=self.final_size)
        self.D = Discriminator(nc=self.nc, size=self.final_size)
        # move to GPU
        self.G = nn.DataParallel(self.G, device_ids=device_ids).to(device)
        self.D = nn.DataParallel(self.D, device_ids=device_ids).to(device)
        # optimizers
        self.opt_G = optim.Adam(self.G.parameters()), lr=opt.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
        self.opt_D = optim.Adam(self.D.parameters()), lr=opt.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
        # data loader
        self.transform = transforms.Compose([
            transforms.Resize((self.current_size,self.current_size), interpolation=Image.NEAREST),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        self.dataset = ISIC2017_GAN('train_gan.csv', shuffle=True, rotate=True, transform=self.transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    def update_trainer(self, stage, inter_epoch):
        if stage == 1:
            assert inter_epoch < opt.unit_epoch, 'Invalid epoch number!'
            G_alpha = 0
            D_alpha = 0
        else:
            total_stages = int(math.log2(self.final_size/4)) + 1
            assert stage <= total_stages, 'Invalid stage number!'
            assert inter_epoch < opt.unit_epoch*3, 'Invalid epoch number!'
            # modify data loader (new image resolution)
            if inter_epoch == 0:
                self.current_size *= 2
                self.transform = transforms.Compose([
                    transforms.Resize((self.current_size,self.current_size), interpolation=Image.NEAREST),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ])
                self.dataset = ISIC2017_GAN('train_gan.csv', shuffle=True, rotate=True, transform=self.transform)
                self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
            # grow networks
            delta = 1. / 29.
            if inter_epoch == 0:
                self.G.module.grow_network()
                self.D.module.grow_network()
            # fade in G
            if inter_epoch < opt.unit_epoch:
                self.G.module.model.fadein.update_alpha(delta)
            # fade in D (30 epochs)
            elif inter_epoch < opt.unit_epoch*2:
                if inter_epoch == opt.unit_epoch:
                    self.G.module.flush_network()
                self.D.module.model.fadein.update_alpha(delta)
            # stablization (30 epochs)
            elif inter_epoch < opt.unit_epoch*3:
                if inter_epoch == opt.unit_epoch*2:
                    self.D.module.flush_network()
            # archive alpha
            G_alpha = self.G.module.model.fadein.get_alpha()
            D_alpha = self.D.module.model.fadein.get_alpha()
        return [G_alpha, D_alpha]
    def update_network(self, real_data):
        # clear grad cache
        self.G.train(); self.D.train()
        self.G.zero_grad(); self.D.zero_grad()
        self.opt_G.zero_grad(); self.opt_D.zero_grad()
        # D loss - real data
        pred_real = self.D(real_data).mean()
        pred_real.backward(torch.FloatTensor([-1]))
        # D loss - fake data
        z = torch.FloatTensor(real_data.size(0), self.nz).normal_(0.0, 1.0).to(device)
        fake_data = self.G(z)
        pred_fake = self.D(fake_data.detach()).mean()
        pred_fake.backward(torch.FloatTensor([1]))
        # D loss - gradient penalty
        gp = self.gradient_penalty(real_data, fake_data.detach())
        gp.backward()
        # update D
        D_loss = pred_fake.item() - pred_real.item() + gp.item()
        Wasserstein_Dist = pred_real.item() - pred_fake.item()
        self.opt_D.step()
        # G loss
        z = torch.FloatTensor(real_data.size(0), self.nz).normal_(0.0, 1.0).to(device)
        fake_data = self.G(z)
        pred_fake = self.D(fake_data.detach()).mean()
        pred_fake.backward(torch.FloatTensor([-1]))
        G_loss = -pred_fake.item()
        self.opt_D.step()
        return [G_loss, D_loss, Wasserstein_Dist]
    def gradient_penalty(self, real_data, fake_data):
        # reference: https://github.com/caogang/wgan-gp/blob/master/gan_mnist.py#L129
        LAMBDA = 10.
        alpha = torch.rand(real_data.size(0)).expand(real_data.size()).to(device)
        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        disc_interpolates = self.D(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True, retain_graph=True)[0]
        gradient_penalty = gradients.norm(2, dim=1).sub(1.).pow(2.).mean()
        return gradient_penalty * LAMBDA
    def train(self):
        global_step = 0
        global_epoch = 0
        total_stages = int(math.log2(self.final_size/4)) + 1
        for stage in range(1, total_stages+1):
            M = opt.unit_epoch if stage == 1 else opt.unit_epoch * 3
            for epoch in range(M):
                G_alpha, D_alpha = self.update_trainer(stage, epoch)
                self.writer.add_scalar('archive/G_alpha', G_alpha, global_epoch)
                self.writer.add_scalar('archive/D_alpha', D_alpha, global_epoch)
                for aug in range(self.num_aug):
                    for i, data in enumerate(self.dataloader, 0):
                        real_data = data
                        real_data = real_data.to(device)
                        G_loss, D_loss, Wasserstein_Dist = self.update_network(real_data)
                        if i % 10 == 0:
                            self.writer.add_scalar('train/G_loss', G_loss, global_step)
                            self.writer.add_scalar('train/D_loss', D_loss, global_step)
                            self.writer.add_scalar('train/Wasserstein_Dist', Wasserstein_Dist, global_step)
                            print("[stage {}/{}][epoch {}/{}][aug {}/{}][iter {}/{}] G_loss {} D_loss {} W_Dist {}" \
                                .format(stage, total_stages, epoch+1, M, aug+1, self.num_aug, i, len(self.dataloader), G_loss, D_loss, Wasserstein_Dist))
                        global_step += 1
                global_epoch += 1

#----------------------------------------------------------------------------
# main function
# perform training
if __name__ == "__main__":
    gan_trainer = trainer(nc=3, nz=512, final_size=256)
    gan_trainer.train()
