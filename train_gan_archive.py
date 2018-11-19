import os
import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.autograd as autograd
from PIL import Image
from tensorboardX import SummaryWriter
from networks import Generator, Discriminator
from data import preprocess_data_gan,  ISIC_GAN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0,1]

parser = argparse.ArgumentParser(description="PGAN-Skin-Lesion")

parser.add_argument("--preprocess", action='store_true')

parser.add_argument("--nc", type=int, default=3, help="number of channels of the generated image")
parser.add_argument("--nz", type=int, default=512, help="dimension of the input noise")
parser.add_argument("--size", type=int, default=256, help="the final size of the generated image")

parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--unit_epoch", type=int, default=50)
parser.add_argument("--num_aug", type=int, default=5, help="times of data augmentation (num_aug times through the dataset is one actual epoch)")
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
        self.current_size = 4
        self.writer = SummaryWriter(opt.outf)
        self.init_trainer()
    def init_trainer(self):
        # networks
        self.G = Generator(nc=opt.nc, nz=opt.nz, size=opt.size)
        self.D = Discriminator(nc=opt.nc, size=opt.size)
        # move to GPU
        self.G = nn.DataParallel(self.G, device_ids=device_ids).to(device)
        self.D = nn.DataParallel(self.D, device_ids=device_ids).to(device)
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
            shuffle=True, num_workers=8, worker_init_fn=__worker_init_fn__)
    def update_trainer(self, stage, inter_epoch):
        if stage == 1:
            assert inter_epoch < opt.unit_epoch, 'Invalid epoch number!'
            G_alpha = 0
            D_alpha = 0
        else:
            total_stages = int(math.log2(opt.size/4)) + 1
            assert stage <= total_stages, 'Invalid stage number!'
            assert inter_epoch < opt.unit_epoch*3, 'Invalid epoch number!'
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
                    shuffle=True, num_workers=8, worker_init_fn=__worker_init_fn__)
            # grow networks
            delta = 1. / (opt.unit_epoch-1)
            if inter_epoch == 0:
                self.G.module.grow_network()
                self.D.module.grow_network()
            # fade in G (# epochs: unit_epoch)
            if inter_epoch < opt.unit_epoch:
                if inter_epoch > 0:
                    self.G.module.model.fadein.update_alpha(delta)
            # fade in D (# epochs: unit_epoch)
            elif inter_epoch < opt.unit_epoch*2:
                if inter_epoch == opt.unit_epoch:
                    self.G.module.flush_network()
                if inter_epoch > opt.unit_epoch:
                    self.D.module.model.fadein.update_alpha(delta)
            # stablization (# epochs: unit_epoch)
            elif inter_epoch < opt.unit_epoch*3:
                if inter_epoch == opt.unit_epoch*2:
                    self.D.module.flush_network()
            # archive alpha
            try:
                G_alpha = self.G.module.model.fadein.get_alpha()
            except:
                G_alpha = 1
            try:
                D_alpha = self.D.module.model.fadein.get_alpha()
            except:
                D_alpha = 1
        self.G.to(device)
        self.D.to(device)
        return [G_alpha, D_alpha]
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
        loss_real = pred_real.mean().mul(-1) + 0.001 * pred_real.pow(2.).mean()
        loss_real.backward()
        # D loss - fake data
        z = torch.FloatTensor(real_data.size(0), opt.nz).normal_(0.0, 1.0).to(device)
        fake_data = self.G.forward(z)
        pred_fake = self.D.forward(fake_data.detach())
        loss_fake = pred_fake.mean()
        loss_fake.backward()
        # D loss - gradient penalty
        gp = self.gradient_penalty(real_data, fake_data)
        gp.backward()
        # update D
        D_loss = loss_real.item() + loss_fake.item() + gp.item()
        Wasserstein_Dist = - (loss_real.item() + loss_fake.item())
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
        loss_fake = pred_fake.mean().mul(-1)
        loss_fake.backward()
        G_loss = loss_fake.item()
        self.opt_G.step()
        return [G_loss, D_loss, Wasserstein_Dist]
    def gradient_penalty(self, real_data, fake_data):
        LAMBDA = 10.
        alpha = torch.rand(real_data.size(0),1,1,1).to(device)
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        disc_interpolates = self.D.forward(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates).to(device), create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = gradients.norm(2, dim=1).sub(1.).pow(2.).mean()
        return gradient_penalty * LAMBDA
    def train(self):
        global_step = 0
        global_epoch = 0
        disp_img = []
        total_stages = int(math.log2(opt.size/4)) + 1
        for stage in range(1, total_stages+1):
            M = opt.unit_epoch if stage == 1 else opt.unit_epoch * 3
            for epoch in range(M):
                G_alpha, D_alpha = self.update_trainer(stage, epoch)
                self.writer.add_scalar('archive/G_alpha', G_alpha, global_epoch)
                self.writer.add_scalar('archive/D_alpha', D_alpha, global_epoch)
                disp_img.clear()
                for aug in range(opt.num_aug):
                    for i, data in enumerate(self.dataloader, 0):
                        real_data = data
                        real_data = real_data.mul(2.).sub(1.) # [0,1] --> [-1,1]
                        if epoch % 10 == 9 and aug == 0 and i == 0:
                            disp_img.append(real_data) # archive for logging image
                        real_data =  real_data.to(device)
                        G_loss, D_loss, Wasserstein_Dist = self.update_network(real_data)
                        if i % 10 == 0:
                            self.writer.add_scalar('train/G_loss', G_loss, global_step)
                            self.writer.add_scalar('train/D_loss', D_loss, global_step)
                            self.writer.add_scalar('train/Wasserstein_Dist', Wasserstein_Dist, global_step)
                            print("[stage {}/{}][epoch {}/{}][aug {}/{}][iter {}/{}] G_loss {:.4f} D_loss {:.4f} W_Dist {:.4f}" \
                                .format(stage, total_stages, epoch+1, M, aug+1, opt.num_aug, i+1, len(self.dataloader), G_loss, D_loss, Wasserstein_Dist))
                        global_step += 1
                global_epoch += 1
                if epoch % 10 == 9:
                    print('\nlog images...\n')
                    I_real = utils.make_grid(disp_img[0], nrow=4, normalize=True, scale_each=True)
                    self.writer.add_image('stage_{}/real'.format(stage), I_real, global_epoch)
                    with torch.no_grad():
                        self.G.eval()
                        z = torch.FloatTensor(disp_img[0].size(0), opt.nz).normal_(0.0, 1.0).to(device)
                        fake_data = self.G.forward(z)
                        I_fake = utils.make_grid(fake_data, nrow=4, normalize=True, scale_each=True)
                        self.writer.add_image('stage_{}/fake'.format(stage), I_fake, epoch)
            torch.save(self.G.state_dict(), os.path.join(opt.outf, 'netG_stage{}.pth'.format(stage)))
            torch.save(self.D.state_dict(), os.path.join(opt.outf, 'netD_stage{}.pth'.format(stage)))

#----------------------------------------------------------------------------
# main function
# perform training
if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data_gan('../data_2017')
    gan_trainer = trainer()
    gan_trainer.train()
