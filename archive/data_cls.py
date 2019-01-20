import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata

def preprocess_data(root_dir, seg_dir='Train_Lesion'):
    print('pre-processing data ...\n')
    # training data
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            writer.writerow([filename] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            writer.writerow([filename] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            writer.writerow([filename] + ['0'])
    # training data oversample
    melanoma = glob.glob(os.path.join(root_dir, 'Train', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Train', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Train', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    with open('train_oversample.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for i in range(4):
            for k in range(len(melanoma)):
                filename = melanoma[k]
                writer.writerow([filename] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            writer.writerow([filename] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            writer.writerow([filename] + ['0'])
    # val data
    melanoma = glob.glob(os.path.join(root_dir, 'Val', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Val', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Val', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    with open('val.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            writer.writerow([filename] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            writer.writerow([filename] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            writer.writerow([filename] + ['0'])
    # test data
    melanoma = glob.glob(os.path.join(root_dir, 'Test', 'melanoma', '*.jpg')); melanoma.sort()
    nevus    = glob.glob(os.path.join(root_dir, 'Test', 'nevus', '*.jpg')); nevus.sort()
    sk       = glob.glob(os.path.join(root_dir, 'Test', 'seborrheic_keratosis', '*.jpg')); sk.sort()
    with open('test.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for k in range(len(melanoma)):
            filename = melanoma[k]
            writer.writerow([filename] + ['1'])
        for k in range(len(nevus)):
            filename = nevus[k]
            writer.writerow([filename] + ['0'])
        for k in range(len(sk)):
            filename = sk[k]
            writer.writerow([filename] + ['0'])

class ISIC(udata.Dataset):
    def __init__(self, csv_file, transform=None):
        file = open(csv_file, newline='')
        reader = csv.reader(file, delimiter=',')
        self.pairs = [row for row in reader]
        self.transform = transform
    def __len__(self):
        return len(self.pairs)
    def  __getitem__(self, idx):
        pair = self.pairs[idx]
        image = Image.open(pair[0])
        label = int(pair[1])
        # transform
        if self.transform:
            image = self.transform(image)
        return image, label
