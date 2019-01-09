import os
import torch
import argparse
from trainer import Trainer
from gen import ImageGenerator
from data_gan import preprocess_data_gan

###
# training:
# device: 4 NVIDIA P100 Pascal GPUs
# training time: ~4d
# stage 1: 50 epoch
# stage 2-7: 50 epoch transition + 50 epoch stability
# stage 5-7: 50 epoch transition + 100 epoch stability
###

###
# num_aug: 10 for ISIC 2017; 5 for ISIC 2018
###

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0,1,2,3]

    parser = argparse.ArgumentParser(description="PGAN-Skin-Lesion")

    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--mode", type=str, default="train", help="train / test")

    # network architecture
    parser.add_argument("--nc", type=int, default=3, help="number of channels of the generated image")
    parser.add_argument("--nz", type=int, default=512, help="dimension of the input noise")
    parser.add_argument("--init_size", type=int, default=4, help="the initial size of the generated image")
    parser.add_argument("--size", type=int, default=256, help="the final size of the generated image")

    # network training
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--unit_epoch", type=int, default=50, help="number of transition epochs")
    parser.add_argument("--lambda_gp", type=float, default=10., help="weight of gradient penalty in WGAN")
    parser.add_argument("--lambda_drift", type=float, default=0.001, help="weight of drift term in D_loss")
    parser.add_argument("--num_aug", type=int, default=5, help="times of data augmentation (num_aug times through the dataset is one actual epoch)")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--outf", type=str, default="logs", help='path of log files')

    arg = parser.parse_args()

    if arg.preprocess:
        preprocess_data_gan('../data_2018/Train/MEL')
        # preprocess_data_gan('../data_2017/Train/melanoma')

    assert arg.mode == "train" or arg.mode == "test", "invalid argument!"
    if arg.mode == "train":
        gan_trainer = Trainer(arg, device, device_ids)
        gan_trainer.train()
    if arg.mode == "test":
        gan_generator = ImageGenerator(arg, device)
        gan_generator.generate(1000)
