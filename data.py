import random
import csv
import os
import os.path
from PIL import Image
import glob
import numpy as np
import torch
import torch.utils.data as udata

def preprocess_data(root_dir, mel_only=True):
    # mel_only = true: only train (unconditional) GAN on mel data
    print('pre-processing data ...\n')
    # training data
    MEL   = glob.glob(os.path.join(root_dir, 'Train', 'MEL', '*.jpg')); MEL.sort()
    NV    = glob.glob(os.path.join(root_dir, 'Train', 'NV', '*.jpg')); NV.sort()
    BCC   = glob.glob(os.path.join(root_dir, 'Train', 'BCC', '*.jpg')); BCC.sort()
    AKIEC = glob.glob(os.path.join(root_dir, 'Train', 'AKIEC', '*.jpg')); AKIEC.sort()
    BKL   = glob.glob(os.path.join(root_dir, 'Train', 'BKL', '*.jpg')); BKL.sort()
    DF    = glob.glob(os.path.join(root_dir, 'Train', 'DF', '*.jpg')); DF.sort()
    VASC  = glob.glob(os.path.join(root_dir, 'Train', 'VASC', '*.jpg')); VASC.sort()
    with open('train.csv', 'wt', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for filename in MEL:
            writer.writerow([filename] + ['0'])
        if not mel_only:
            for filename in NV:
                writer.writerow([filename] + ['1'])
            for filename in BCC:
                writer.writerow([filename] + ['2'])
            for filename in AKIEC:
                writer.writerow([filename] + ['3'])
            for filename in BKL:
                writer.writerow([filename] + ['4'])
            for filename in DF:
                writer.writerow([filename] + ['5'])
            for filename in VASC:
                writer.writerow([filename] + ['6'])
    if not mel_only:
        # val data
        MEL   = glob.glob(os.path.join(root_dir, 'Val', 'MEL', '*.jpg')); MEL.sort()
        NV    = glob.glob(os.path.join(root_dir, 'Val', 'NV', '*.jpg')); NV.sort()
        BCC   = glob.glob(os.path.join(root_dir, 'Val', 'BCC', '*.jpg')); BCC.sort()
        AKIEC = glob.glob(os.path.join(root_dir, 'Val', 'AKIEC', '*.jpg')); AKIEC.sort()
        BKL   = glob.glob(os.path.join(root_dir, 'Val', 'BKL', '*.jpg')); BKL.sort()
        DF    = glob.glob(os.path.join(root_dir, 'Val', 'DF', '*.jpg')); DF.sort()
        VASC  = glob.glob(os.path.join(root_dir, 'Val', 'VASC', '*.jpg')); VASC.sort()
        with open('val.csv', 'wt', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for filename in MEL:
                writer.writerow([filename] + ['0'])
            for filename in NV:
                writer.writerow([filename] + ['1'])
            for filename in BCC:
                writer.writerow([filename] + ['2'])
            for filename in AKIEC:
                writer.writerow([filename] + ['3'])
            for filename in BKL:
                writer.writerow([filename] + ['4'])
            for filename in DF:
                writer.writerow([filename] + ['5'])
            for filename in VASC:
                writer.writerow([filename] + ['6'])
        # test data
        MEL   = glob.glob(os.path.join(root_dir, 'Test', 'MEL', '*.jpg')); MEL.sort()
        NV    = glob.glob(os.path.join(root_dir, 'Test', 'NV', '*.jpg')); NV.sort()
        BCC   = glob.glob(os.path.join(root_dir, 'Test', 'BCC', '*.jpg')); BCC.sort()
        AKIEC = glob.glob(os.path.join(root_dir, 'Test', 'AKIEC', '*.jpg')); AKIEC.sort()
        BKL   = glob.glob(os.path.join(root_dir, 'Test', 'BKL', '*.jpg')); BKL.sort()
        DF    = glob.glob(os.path.join(root_dir, 'Test', 'DF', '*.jpg')); DF.sort()
        VASC  = glob.glob(os.path.join(root_dir, 'Test', 'VASC', '*.jpg')); VASC.sort()
        with open('test.csv', 'wt', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for filename in MEL:
                writer.writerow([filename] + ['0'])
            for filename in NV:
                writer.writerow([filename] + ['1'])
            for filename in BCC:
                writer.writerow([filename] + ['2'])
            for filename in AKIEC:
                writer.writerow([filename] + ['3'])
            for filename in BKL:
                writer.writerow([filename] + ['4'])
            for filename in DF:
                writer.writerow([filename] + ['5'])
            for filename in VASC:
                writer.writerow([filename] + ['6'])

## Imbalanced Dataset Sampler
## thanks to: https://github.com/ufoym/imbalanced-dataset-sampler

class ImbalancedDatasetSampler(udata.sampler.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
    def _get_label(self, dataset, idx):
        __, label = dataset[idx]
        return label
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))
    def __len__(self):
        return self.num_samples

class ISIC_GAN(udata.Dataset):
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
        if self.transform:
            image = self.transform(image)
        return image, label
