import random
import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata

def preprocess_data_gan(root_dir):
    print('pre-processing data for GAN ...\n')
    melanoma = glob.glob(os.path.join(root_dir, '*.jpg'))
    melanoma.sort()
    with open('train_gan.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in melanoma:
            writer.writerow([filename])

class ISIC_GAN(udata.Dataset):
    def __init__(self, csv_file, shuffle=True, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.files = [row for row in reader]
        if shuffle:
            random.shuffle(self.files)
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def  __getitem__(self, idx):
        image = Image.open(self.files[idx][0])
        if self.transform:
            image = self.transform(image)
        return image
