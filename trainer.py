import os
import math
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
from data_gan import ISIC_GAN
from transforms import *

def _worker_init_fn_():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

class Trainer:
    def __init__(self, arg, device, device_ids):
        print("\ninitializing trainer ...\n")
        # network architecture
        self.nc = arg.nc
        self.nz = arg.nz
        self.init_size = arg.init_size
        self.size = arg.size
        # training
        self.batch_size = arg.batch_size
        self.unit_epoch = arg.unit_epoch
        self.lambda_gp  = arg.lambda_gp
        self.lambda_drift = arg.lambda_drift
        self.num_aug = arg.num_aug
        self.lr = arg.lr
        self.outf = arg.outf
        self.device = device
        self.device_ids = device_ids
        self.writer = SummaryWriter(self.outf)
        self.init_trainer()
        print("done\n")
    def init_trainer(self):
        # networks
        self.G = Generator(nc=self.nc, nz=self.nz, size=self.size)
        self.D = Discriminator(nc=self.nc, nz=self.nz, size=self.size)
        self.G_EMA = copy.deepcopy(self.G)
        # move to GPU
        self.G = nn.DataParallel(self.G, device_ids=self.device_ids).to(self.device)
        self.D = nn.DataParallel(self.D, device_ids=self.device_ids).to(self.device)
        self.G_EMA = self.G_EMA.to('cpu') # keep this model on CPU to save GPU memory
        for param in self.G_EMA.parameters():
            param.requires_grad_(False) # turn off grad because G_EMA will only be used for inference
        # optimizers
        self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
        self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
        # data loader
        self.transform = transforms.Compose([
            RatioCenterCrop(1.),
            transforms.Resize((300,300), Image.ANTIALIAS),
            transforms.RandomCrop((self.size,self.size)),
            RandomRotate(),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.dataset = ISIC_GAN('train_gan.csv', transform=self.transform)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=8, worker_init_fn=_worker_init_fn_(), drop_last=True)
        # tickers (used for fading in)
        self.tickers = self.unit_epoch * self.num_aug * len(self.dataloader)
    def update_trainer(self, stage, inter_ticker):
        if stage == 1:
            current_alpha = 0
        else:
            total_stages = int(math.log2(self.size/self.init_size)) + 1
            assert stage <= total_stages, 'Invalid stage number!'
            flag_opt = False
            delta = 1. / self.tickers
            if inter_ticker == 0:
                self.G.module.grow_network()
                self.D.module.grow_network()
                self.G_EMA.grow_network()
                flag_opt = True
            elif (inter_ticker > 0) and (inter_ticker < self.tickers):
                self.G.module.model.fadein.update_alpha(delta)
                self.D.module.model.fadein.update_alpha(delta)
                self.G_EMA.model.fadein.update_alpha(delta)
                flag_opt = False
            elif inter_ticker == self.tickers:
                self.G.module.flush_network()
                self.D.module.flush_network()
                self.G_EMA.flush_network()
                flag_opt = True
            else:
                flag_opt = False;
            # archive alpha
            try:
                current_alpha = self.G.module.model.fadein.get_alpha()
            except:
                current_alpha = 1
            # move to devie & update optimizer
            if flag_opt:
                self.G.to(self.device)
                self.D.to(self.device)
                self.G_EMA.to('cpu')
                # opt_G
                opt_G_state_dict = self.opt_G.state_dict()
                old_opt_G_state = opt_G_state_dict['state']
                self.opt_G = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
                new_opt_G_param_id =  self.opt_G.state_dict()['param_groups'][0]['params']
                opt_G_state = copy.deepcopy(old_opt_G_state)
                for key in old_opt_G_state.keys():
                    if key not in new_opt_G_param_id:
                        del opt_G_state[key]
                opt_G_state_dict['param_groups'] = self.opt_G.state_dict()['param_groups']
                opt_G_state_dict['state'] = opt_G_state
                self.opt_G.load_state_dict(opt_G_state_dict)
                # opt_D
                opt_D_state_dict = self.opt_D.state_dict()
                old_opt_D_state = opt_D_state_dict['state']
                self.opt_D = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0,0.99), eps=1e-8, weight_decay=0.)
                new_opt_D_param_id =  self.opt_D.state_dict()['param_groups'][0]['params']
                opt_D_state = copy.deepcopy(old_opt_D_state)
                for key in old_opt_D_state.keys():
                    if key not in new_opt_D_param_id:
                        del opt_D_state[key]
                opt_D_state_dict['param_groups'] = self.opt_D.state_dict()['param_groups']
                opt_D_state_dict['state'] = opt_D_state
                self.opt_D.load_state_dict(opt_D_state_dict)
        return current_alpha
    def update_moving_average(self, decay=0.999):
        # update exponential running average (EMA) for the weights of the generator
        # W_EMA_t = decay * W_EMA_{t-1} + (1-decay) * W_G
        with torch.no_grad():
            param_dict_G = dict(self.G.module.named_parameters())
            for name, param_EMA in self.G_EMA.named_parameters():
                param_G = param_dict_G[name]
                assert (param_G is not param_EMA)
                param_EMA.copy_(decay * param_EMA + (1. - decay) * param_G.detach().cpu())
    def update_network(self, real_data):
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
        loss_real = pred_real.mean().mul(-1.)
        loss_real_drift = pred_real.pow(2.).mean()
        # D loss - fake data
        z = torch.FloatTensor(real_data.size(0), self.nz).normal_(0.0, 1.0).to(self.device)
        fake_data = self.G.forward(z)
        pred_fake = self.D.forward(fake_data.detach())
        loss_fake = pred_fake.mean()
        # D loss - gradient penalty
        gp = self.gradient_penalty(real_data, fake_data)
        # update D
        D_loss = loss_real + loss_fake + self.lambda_drift * loss_real_drift + self.lambda_gp * gp
        W_dist = loss_real.item() + loss_fake.item()
        D_loss.backward()
        self.opt_D.step()
        ##########
        ## Train Generator
        ##########
        # clear grad cache
        self.G.zero_grad()
        self.opt_G.zero_grad()
        # G loss
        z = torch.FloatTensor(real_data.size(0), self.nz).normal_(0.0, 1.0).to(self.device)
        fake_data = self.G.forward(z)
        pred_fake = self.D.forward(fake_data)
        # update G
        G_loss = pred_fake.mean().mul(-1.)
        G_loss.backward()
        self.opt_G.step()
        return [G_loss.item(), D_loss.item(), W_dist]
    def gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(real_data.size(0),1,1,1).to(self.device)
        interpolates = alpha * real_data.detach() + (1 - alpha) * fake_data.detach()
        interpolates.requires_grad_(True)
        disc_interpolates = self.D.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(self.device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = gradients.norm(2, dim=1).sub(1.).pow(2.).mean()
        return gradient_penalty
    def train(self):
        global_step = 0
        global_epoch = 0
        total_stages = int(math.log2(self.size/self.init_size)) + 1
        fixed_z = torch.FloatTensor(self.batch_size, self.nz).normal_(0.0, 1.0).to('cpu')
        for stage in range(1, total_stages+1):
            if stage == 1:
                M = self.unit_epoch
            elif stage <= 4:
                M = self.unit_epoch * 2
            else:
                M = self.unit_epoch * 3
            current_size = self.init_size * (2 ** (stage-1))
            ticker = 0
            for epoch in range(M):
                torch.cuda.empty_cache()
                for aug in range(self.num_aug):
                    for i, data in enumerate(self.dataloader, 0):
                        current_alpha = self.update_trainer(stage, ticker)
                        self.writer.add_scalar('archive/current_alpha', current_alpha, global_step)
                        real_data_current = data
                        real_data_current = F.adaptive_avg_pool2d(real_data_current, current_size)
                        if stage > 1 and current_alpha < 1:
                            real_data_previous = F.interpolate(F.avg_pool2d(real_data_current, 2), scale_factor=2., mode='nearest')
                            real_data = (1 - current_alpha) * real_data_previous + current_alpha * real_data_current
                        else:
                            real_data = real_data_current
                        real_data = real_data.mul(2.).sub(1.) # [0,1] --> [-1,1]
                        real_data = real_data.to(self.device)
                        G_loss, D_loss, W_dist = self.update_network(real_data)
                        self.update_moving_average()
                        if i % 10 == 0:
                            self.writer.add_scalar('train/G_loss', G_loss, global_step)
                            self.writer.add_scalar('train/D_loss', D_loss, global_step)
                            self.writer.add_scalar('train/W_dist', W_dist, global_step)
                            print("[stage {}/{}][epoch {}/{}][aug {}/{}][iter {}/{}] G_loss {:.4f} D_loss {:.4f} W_Dist {:.4f}" \
                                .format(stage, total_stages, epoch+1, M, aug+1, self.num_aug, i+1, len(self.dataloader), G_loss, D_loss, W_dist))
                        global_step += 1
                        ticker += 1
                global_epoch += 1
                if epoch % 10 == 9:
                    # log image
                    print('\nlog images...\n')
                    I_real = utils.make_grid(real_data, nrow=4, normalize=True, scale_each=True)
                    self.writer.add_image('stage_{}/real'.format(stage), I_real, epoch)
                    with torch.no_grad():
                        self.G_EMA.eval()
                        fake_data = self.G_EMA.forward(fixed_z)
                        I_fake = utils.make_grid(fake_data, nrow=4, normalize=True, scale_each=True)
                        self.writer.add_image('stage_{}/fake'.format(stage), I_fake, epoch)
                    # save checkpoints
                    print('\nsaving checkpoints...\n')
                    checkpoint = {
                        'G_state_dict': self.G.module.state_dict(),
                        'G_EMA_state_dict': self.G_EMA.state_dict(),
                        'D_state_dict': self.D.module.state_dict(),
                        'opt_G_state_dict': self.opt_G.state_dict(),
                        'opt_D_state_dict': self.opt_D.state_dict(),
                        'stage': stage
                    }
                    torch.save(checkpoint, os.path.join(self.outf,'stage{}.tar'.format(stage))) # overwrite if exist
