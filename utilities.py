import numpy as np
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, precision_score, recall_score

import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

def compute_metrics(result_file):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]
    # prediction
    pred = []
    i = 0
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = list(map(float, row))
            pred.append(prob[1])
            i += 1
    # compute mAP
    mAP = average_precision_score(gt, pred, average='macro')
    # compute AUC
    AUC = roc_auc_score(gt, pred)
    # plot ROC curve
    precision, recall, __ = precision_recall_curve(gt, pred, pos_label=1)
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    ax.step(recall, precision, color='b', alpha=0.2, where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    canvas.draw()
    I = np.fromstring(canvas.tostring_rgb(), dtype='uint8', sep='')
    I = I.reshape(canvas.get_width_height()[::-1]+(3,))
    I = np.transpose(I, [2,0,1])
    return mAP, AUC, torch.Tensor(np.float32(I))

def compute_mean_pecision_recall(result_file, threshold=0.5):
    # groundtruth
    with open('test.csv', 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        gt = [int(row[1]) for row in reader]
    # prediction
    pred = []
    with open(result_file, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            prob = np.array(list(map(float, row)))
            pred.append(np.float32(prob[1] >= threshold))
    # compute precision & recall
    precision  = precision_score(gt, pred, average='macro')
    recall     = recall_score(gt, pred, average='macro')
    precision_mel  = precision_score(gt, pred, average='binary', pos_label=1)
    recall_mel = recall_score(gt, pred, average='binary', pos_label=1)
    return precision, recall, precision_mel, recall_mel
