import os
import csv
import random
import argparse
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from networks import VGG, ResNet
from data import preprocess_data_classify, ISIC
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]

parser = argparse.ArgumentParser(description="Classifier")

parser.add_argument("--preprocess", action='store_true', help="run preprocess_data")
parser.add_argument("--model", type=str, default="VGNet", help='VGGNet or ResNet')
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')

opt = parser.parse_args()

def __worker_init_fn__():
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32-1
    random.seed(torch_seed)
    np.random.seed(np_seed)

def main():
    # load data
    print('\nloading the dataset ...\n')
    num_aug = 5
    im_size = 224
    transform_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(im_size),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.6901, 0.5442, 0.4867), (0.0810, 0.1118, 0.1266)) # without aug
        transforms.Normalize((0.7216, 0.5598, 0.4962), (0.0889, 0.1230, 0.1390)) # with aug
    ])
    transform_test = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop(im_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.6901, 0.5442, 0.4867), (0.0810, 0.1118, 0.1266)) # without aug
        transforms.Normalize((0.7216, 0.5598, 0.4962), (0.0889, 0.1230, 0.1390)) # with aug
    ])
    trainset = ISIC(csv_file='train.csv', shuffle=True, rotate=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size, shuffle=True, num_workers=8, worker_init_fn=__worker_init_fn__())
    testset = ISIC(csv_file='test.csv', shuffle=False, rotate=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=8)
    print('\ndone\n')
    '''
    Mean = torch.zeros(3)
    Std = torch.zeros(3)
    for data in trainloader:
        I, __ = data
        N, C, __, __ = I.size()
        Mean += I.view(N,C,-1).mean(2).sum(0)
        Std += I.view(N,C,-1).std(2).sum(0)
    Mean /= len(trainset)
    Std  /= len(trainset)
    print('mean: '), print(Mean.numpy())
    print('std: '), print(Std.numpy())
    return
    '''

    # load models
    print('\nloading the model ...\n')
    if opt.model == 'VGGNet':
        print('Using VGG-16')
        net = VGG(num_classes=2)
    if opt.model == 'ResNet':
        print('Uing ResNet-50')
        net = ResNet(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    print('\ndone\n')

    # move to GPU
    print('\nmoving models to GPU ...\n')
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    criterion.to(device)
    print('\ndone\n')

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)
    lr_lambda = lambda epoch : np.power(0.5, epoch//10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # training
    print('\nstart training ...\n')
    step = 0
    EMA_accuracy = 0
    writer = SummaryWriter(opt.outf)
    for epoch in range(opt.epochs):
        torch.cuda.empty_cache()
        # adjust learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('train/learning_rate', current_lr, epoch)
        print("\nepoch %d learning rate %f\n" % (epoch, current_lr))
        # run for one epoch
        for aug in range(num_aug):
            for i, data in enumerate(trainloader, 0):
                # warm up
                model.train()
                model.zero_grad()
                optimizer.zero_grad()
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # forward
                pred = model.forward(inputs)
                # backward
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                # display results
                if i % 10 == 0:
                    model.eval()
                    pred = model.forward(inputs)
                    predict = torch.argmax(pred, 1)
                    total = labels.size(0)
                    correct = torch.eq(predict, labels).sum().double().item()
                    accuracy = correct / total
                    EMA_accuracy = 0.98*EMA_accuracy + 0.02*accuracy
                    writer.add_scalar('train/loss_c', loss.item(), step)
                    writer.add_scalar('train/accuracy', accuracy, step)
                    writer.add_scalar('train/EMA_accuracy', EMA_accuracy, step)
                    print("[epoch %d][aug %d/%d][%d/%d] loss %.4f accuracy %.2f%% EMA_accuracy %.2f%%"
                        % (epoch, aug, num_aug-1, i, len(trainloader)-1, loss.item(), (100*accuracy), (100*EMA_accuracy)))
                step += 1
        # the end of each epoch
        model.eval()
        # save checkpoints
        print('\none epoch done, saving checkpoints ...\n')
        checkpoint = {
            'state_dict': model.module.state_dict(),
            'opt_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, os.path.join(opt.outf,'checkpoint.pth'))
        if epoch == opt.epochs / 2:
            torch.save(checkpoint, os.path.join(opt.outf, 'checkpoint_%d.pth' % epoch))
        # log test results
        total = 0
        correct = 0
        with torch.no_grad():
            with open('test_results.csv', 'wt', newline='') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',')
                for i, data in enumerate(testloader, 0):
                    images_test, labels_test = data
                    images_test, labels_test = images_test.to(device), labels_test.to(device)
                    pred_test = model.forward(images_test)
                    predict = torch.argmax(pred_test, 1)
                    total += labels_test.size(0)
                    correct += torch.eq(predict, labels_test).sum().double().item()
                    # record test predicted responses
                    responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                    responses = [responses[i] for i in range(responses.shape[0])]
                    csv_writer.writerows(responses)
            # log scalars
            precision, recall, precision_mel, recall_mel = compute_mean_pecision_recall('test_results.csv')
            mAP, AUC, ROC = compute_metrics('test_results.csv')
            writer.add_scalar('test/accuracy', correct/total, epoch)
            writer.add_scalar('test/mean_precision', precision, epoch)
            writer.add_scalar('test/mean_recall', recall, epoch)
            writer.add_scalar('test/precision_mel', precision_mel, epoch)
            writer.add_scalar('test/recall_mel', recall_mel, epoch)
            writer.add_scalar('test/mAP', mAP, epoch)
            writer.add_scalar('test/AUC', AUC, epoch)
            writer.add_image('curve/ROC', ROC, epoch)
            print("\n[epoch %d] test result: accuracy %.2f%% \nmean precision %.2f%% mean recall %.2f%% \
                    \nprecision for mel %.2f%% recall for mel %.2f%% \nmAP %.2f%% AUC %.4f\n" %
                    (epoch, 100*correct/total, 100*precision, 100*recall, 100*precision_mel, 100*recall_mel, 100*mAP, AUC))

if __name__ == "__main__":
    if opt.preprocess:
        preprocess_data_classify(root_dir='data_2017_aug')
    main()
