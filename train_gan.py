import os
import math
import argparse
import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torchvision import transforms, utils, datasets
from PIL import Image
from tensorboardX import SummaryWriter
from networks import Generator, Discriminator
from data import preprocess_data_gan, ISIC_GAN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1,2,3]

parser = argparse.ArgumentParser(description="PGAN-Skin-Lesion")

parser.add_argument("--preprocess", action='store_true')

parser.add_argument("--nc", type=int, default=3, help="number of channels of the generated image")
parser.add_argument("--nz", type=int, default=512, help="dimension of the input noise")
parser.add_argument("--size", type=int, default=256, help="the final size of the generated image")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--unit_epoch", type=int, default=50)
parser.add_argument("--num_aug", type=int, default=10, help="times of data augmentation (num_aug times through the dataset is one actual epoch)")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')

opt = parser.parse_args()

#----------------------------------------------------------------------------
# Trainer

def __worker_init_fn__():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

class trainer:
    def __init__(self):
        print("\ninitializing trainer ...\n")
        self.current_size = 4
        self.writer = SummaryWriter(opt.outf)
        self.init_trainer()
        print("\ndone\n")
    def init_trainer(self):
        # networks
        self.G = Generator(nc=opt.nc, nz=opt.nz, size=opt.size)
        self.D = Discriminator(nc=opt.nc, nz=opt.nz, size=opt.size)
        self.G_EMA = copy.deepcopy(self.G)
        # move to GPU
        self.G = nn.DataParallel(self.G, device_ids=device_ids).to(device)
        self.D = nn.DataParallel(self.D, device_ids=device_ids).to(device)
        self.G_EMA = self.G_EMA.to('cpu') # keep this model on CPU to save GPU memory
        for param in self.G_EMA.parameters():
            param.requires_grad_(False) # turn off grad because G_EMA will only be used for inference
        # optimizers
        self.opt_G = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
        self.opt_D = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
        # data loader
        self.transform = transforms.Compose([
            transforms.Resize((300,300)),
            transforms.RandomCrop((opt.size,opt.size)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.current_size,self.current_size), Image.ANTIALIAS),
            transforms.ToTensor()
        ])
        self.dataset = ISIC_GAN('train_gan.csv', shuffle=True, rotate=True, transform=self.transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size,
            shuffle=True, num_workers=8, worker_init_fn=__worker_init_fn__(), drop_last=True)
    def update_trainer(self, stage, inter_epoch):
        """
        update status of trainer
        :param stage: stage number; starting from 1
        :param inter_epoch: epoch number within the current stage; starting from 0 within each stage
        :return current_alpha: value of alpha (parameter for fade in) after updating trainer
        """
        flag_opt = False
        print("\nupdating trainer ...\n")
        if stage == 1:
            assert inter_epoch < opt.unit_epoch, 'Invalid epoch number!'
            current_alpha = 0
        else:
            total_stages = int(math.log2(opt.size/4)) + 1
            assert stage <= total_stages, 'Invalid stage number!'
            assert inter_epoch < opt.unit_epoch * 2, 'Invalid epoch number!'
            # adjust dataloder (new current_size)
            if inter_epoch == 0:
                self.current_size *= 2
                self.transform = transforms.Compose([
                    transforms.Resize((300,300)),
                    transforms.RandomCrop((opt.size,opt.size)),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((self.current_size,self.current_size), Image.ANTIALIAS),
                    transforms.ToTensor()
                ])
                self.dataset = ISIC_GAN('train_gan.csv', shuffle=True, rotate=True, transform=self.transform)
                self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=opt.batch_size,
                    shuffle=True, num_workers=8, worker_init_fn=__worker_init_fn__(), drop_last=True)

            delta = 1. / (opt.unit_epoch-1.)
            # grow networks
            if inter_epoch == 0:
                self.G.module.grow_network()
                self.D.module.grow_network()
                self.G_EMA.grow_network()
                flag_opt = True
            # fade in
            elif (inter_epoch > 0) and (inter_epoch < opt.unit_epoch):
                self.G.module.model.fadein.update_alpha(delta)
                self.D.module.model.fadein.update_alpha(delta)
                self.G_EMA.model.fadein.update_alpha(delta)
                flag_opt = False
            # flush networks
            elif inter_epoch == opt.unit_epoch:
                self.G.module.flush_network()
                self.D.module.flush_network()
                self.G_EMA.flush_network()
                flag_opt = True
            # stablization
            else:
                print("\nnothing to update about trainer ...\n")

            # archive alpha
            try:
                current_alpha = self.G.module.model.fadein.get_alpha()
            except:
                current_alpha = 1

            # move to devie & update optimizer
            if flag_opt:
                self.G.to(device)
                self.D.to(device)
                self.G_EMA.to('cpu')
                state_G = self.opt_G.state
                state_D = self.opt_D.state
                self.opt_G = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
                self.opt_D = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
                self.opt_G.state = state_G
                self.opt_D.state = state_D
        print("\ndone\n")
        return current_alpha
    def update_moving_average(self, decay=0.999):
        """
        update exponential running average (EMA) for the weights of the generator
        :param decay: the EMA is computed as W_EMA_t = decay * W_EMA_{t-1} + (1-decay) * W_G
        :return : None
        """
        with torch.no_grad():
            param_dict_G = dict(self.G.module.named_parameters())
            for name, param_EMA in self.G_EMA.named_parameters():
                param_G = param_dict_G[name]
                assert (param_G is not param_EMA)
                param_EMA.data.copy_(decay * param_EMA.data + (1. - decay) * param_G.detach().cpu())
    def update_network(self, real_data):
        """
        perform one step of gradient descent
        :param real_data: batch of real image; the dynamic range must has been adjusted to [-1,1]
        :return [G_loss, D_loss, Wasserstein_Dist]
        """
        # switch to training mode
        self.G.train(); self.D.train()
        ##########
        ## Train Discriminator
        ##########
        # clear grad cache
        self.D.zero_grad()
        self.opt_D.zero_grad()
        # D loss - real data
        pred_real = self.D.forward(real_data)
        loss_real = pred_real.mean()
        loss_real_drift = 0.001 * pred_real.pow(2.).mean()
        # D loss - fake data
        z = torch.FloatTensor(real_data.size(0), opt.nz).normal_(0.0, 1.0).to(device)
        fake_data = self.G.forward(z)
        pred_fake = self.D.forward(fake_data.detach())
        loss_fake = pred_fake.mean()
        # D loss - gradient penalty
        gp = self.gradient_penalty(real_data, fake_data)
        # update D
        D_loss = loss_fake - loss_real + loss_real_drift + gp
        Wasserstein_Dist = loss_fake.item() - loss_real.item()
        D_loss.backward()
        self.opt_D.step()
        ##########
        ## Train Generator
        ##########
        # clear grad cache
        self.G.zero_grad()
        self.opt_G.zero_grad()
        # G loss
        z = torch.FloatTensor(real_data.size(0), opt.nz).normal_(0.0, 1.0).to(device)
        fake_data = self.G.forward(z)
        pred_fake = self.D.forward(fake_data)
        G_loss = pred_fake.mean().mul(-1.)
        G_loss.backward()
        self.opt_G.step()
        return [G_loss.item(), D_loss.item(), Wasserstein_Dist]
    def gradient_penalty(self, real_data, fake_data):
        LAMBDA = 10.
        alpha = torch.rand(real_data.size(0),1,1,1).to(device)
        interpolates = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
        interpolates.requires_grad_(True)
        disc_interpolates = self.D.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = LAMBDA * gradients.norm(2, dim=1).sub(1.).pow(2.).mean()
        return gradient_penalty
    def train(self):
        global_step = 0
        global_epoch = 0
        disp_circle = 10 if opt.unit_epoch > 10 else 1
        total_stages = int(math.log2(opt.size/4)) + 1
        for stage in range(1, total_stages+1):
            M = opt.unit_epoch if stage == 1 else opt.unit_epoch * 2
            for epoch in range(M):
                current_alpha = self.update_trainer(stage, epoch)
                self.writer.add_scalar('archive/current_alpha', current_alpha, global_epoch)
                torch.cuda.empty_cache()
                for aug in range(opt.num_aug):
                    for i, data in enumerate(self.dataloader, 0):
                        real_data_current = data
                        if stage > 1:
                            real_data_previous = F.interpolate(F.avg_pool2d(real_data_current, 2), scale_factor=2., mode='nearest')
                            real_data = (1 - current_alpha) * real_data_previous + current_alpha * real_data_current
                        else:
                            real_data = real_data_current
                        real_data = real_data.mul(2.).sub(1.) # [0,1] --> [-1,1]
                        real_data =  real_data.to(device)
                        G_loss, D_loss, Wasserstein_Dist = self.update_network(real_data)
                        self.update_moving_average()
                        if i % 10 == 0:
                            self.writer.add_scalar('train/G_loss', G_loss, global_step)
                            self.writer.add_scalar('train/D_loss', D_loss, global_step)
                            self.writer.add_scalar('train/Wasserstein_Dist', Wasserstein_Dist, global_step)
                            print("[stage {}/{}][epoch {}/{}][aug {}/{}][iter {}/{}] G_loss {:.4f} D_loss {:.4f} W_Dist {:.4f}" \
                                .format(stage, total_stages, epoch+1, M, aug+1, opt.num_aug, i+1, len(self.dataloader), G_loss, D_loss, Wasserstein_Dist))
                        global_step += 1
                global_epoch += 1
                if epoch % disp_circle == disp_circle-1:
                    print('\nlog images...\n')
                    I_real = utils.make_grid(real_data, nrow=4, normalize=True, scale_each=True)
                    self.writer.add_image('stage_{}/real'.format(stage), I_real, epoch)
                    with torch.no_grad():
                        self.G_EMA.eval()
                        z = torch.FloatTensor(real_data.size(0), opt.nz).normal_(0.0, 1.0).to('cpu')
                        fake_data = self.G_EMA.forward(z)
                        I_fake = utils.make_grid(fake_data, nrow=4, normalize=True, scale_each=True)
                        self.writer.add_image('stage_{}/fake'.format(stage), I_fake, epoch)
            # after each stage: save checkpoints
            print('\nsaving checkpoints...\n')
            checkpoint = {
                'G_state_dict': self.G.module.state_dict(),
                'G_EMA_state_dict': self.G_EMA.state_dict(),
                'D_state_dict': self.D.module.state_dict(),
                'opt_G_state_dict': self.opt_G.state_dict(),
                'opt_D_state_dict': self.opt_D.state_dict(),
                'stage': stage
            }
            torch.save(checkpoint, os.path.join(opt.outf,'stage{}.pth'.format(stage)))

#----------------------------------------------------------------------------
# main function
# perform training
if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data_gan('../data_2017')
    gan_trainer = trainer()
    gan_trainer.train()
