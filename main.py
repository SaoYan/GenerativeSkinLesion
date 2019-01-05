import os
import argparse
from trainer import Trainer
from data_gan import preprocess_data_gan_2017, preprocess_data_gan_2018

###
# train for stage 1-7
# device: 4 NVIDIA P100 Pascal GPUs
# training time: ~4d
# stage 1: 50 epoch
# stage 2-4: 50 epoch transition + 50 epoch stability
# stage 5-7: 50 epoch transition + 100 epoch stability
###

###
# switch between ISIC 2017 and 2018
# modify the following contents:
# 1. root_dir of preprocess_data
# 2. num_aug
###

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1,2,3]

    parser = argparse.ArgumentParser(description="PGAN-Skin-Lesion")

    # data preprocess
    parser.add_argument("--preprocess", action='store_true')

    # network architecture
    parser.add_argument("--nc", type=int, default=3, help="number of channels of the generated image")
    parser.add_argument("--nz", type=int, default=512, help="dimension of the input noise")
    parser.add_argument("--init_size", type=int, default=4, help="the initial size of the generated image")
    parser.add_argument("--size", type=int, default=256, help="the final size of the generated image")

    # network training
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--unit_epoch", type=int, default=50)
    parser.add_argument("--num_aug", type=int, default=10, help="times of data augmentation (num_aug times through the dataset is one actual epoch)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--outf", type=str, default="logs", help='path of log files')

    config = parser.parse_args()

    if config.preprocess:
        preprocess_data_gan_2018('../data_2018')
    gan_trainer = Trainer(config, device, device_ids)
    gan_trainer.train()
