from asyncio import base_tasks
import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ae import AE
from utils import load_train_data, save_validation_losses_density, save_confusion_matrix, save_originals, save_reconstructions, save_loss, save_roc_curve
from data import transform_data

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors


import os
import sys
import argparse
import pandas as pd
import numpy as np
import timeit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


ALL_CLASSES = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper','casting','MSD','casting_augmented']
latent_spaces = [128,256,512]

def parse_args():
    parser = argparse.ArgumentParser('AE')
    parser.add_argument("--classe", type=str, default='all')
    parser.add_argument("--save_path", type=str, default="./results/validation/ERCON")   ## ERCON for reconstruction error
    parser.add_argument("-ls","--latent_space", type=int, default=256)
    return parser.parse_args()


def test():

    """ Runs AutoEncoder Performance Tests Using Reconstrucntion Error 
    
    Saves metrics results in an excel files ( roc_auc.xlsx and f1score.xlsx)
    """

    args = parse_args()

    latent_spaces = [args.latent_space]
    save_path = args.save_path
    data_path = './dataset/'
    mvtec_classes = [args.classe]
    if args.classe == 'all':
        mvtec_classes = ALL_CLASSES

    total_roc_auc = np.zeros((len(mvtec_classes), len(latent_spaces)))
    total_f1_score = np.zeros((len(mvtec_classes), len(latent_spaces)))
    for i, classe in enumerate(mvtec_classes):
        for j, latent_space in enumerate(tqdm(latent_spaces,desc=f' | {classe} | ')):
            model = AE(latent_dim=latent_space,scale=1).to('cuda')
            model.load_state_dict(torch.load(f'results/Training/models/{classe}/latent_dim_{latent_space}.pt'))
            model.eval()
            roc_auc, f1_score = one_classe_test(save_path, data_path, model,latent_space,classe)
            total_roc_auc[i][j] = roc_auc
            total_f1_score[i][j] = f1_score
    
    
    metrics_path = save_path + '/metrics/' + args.classe 
    if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
    df = pd.DataFrame(total_roc_auc.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([str(latent) for latent in latent_spaces]))
    df.to_excel(metrics_path + f'/roc_auc_{latent_space}.xlsx')
    df = pd.DataFrame(total_f1_score.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([str(latent) for latent in latent_spaces]))
    df.to_excel(metrics_path + f'/f1_score_{latent_space}.xlsx')


def one_classe_test(save_path, data_path, model,latent_space, classe):

    """ Runs AutoEncoder predictions and metric calculations for ONE Data (classe)"""

    transform_train, transform_test = transform_data(classe,False), transform_data(classe, False)
    transform = {'train' : transform_train, 'test' : transform_test}
    _, testloader = load_train_data(transform, classe,data_path, 32,1)

    loss_function = nn.MSELoss()
    classe_path = save_path + '/classes/' + classe 
    affiche_tqdm_class = classe + f' | latent_dim_{latent_space}'
    ## Validation
    roc_auc, f1_score = validate(model, testloader, loss_function, classe_path, latent_space)

    return roc_auc, f1_score


def myMSE(x,y):
    return torch.mean((x-y)**2,dim = [1,2,3])

def validate(model, testloader, loss_function, classe_path, latent_space):

    """ Runs AutoEncoder Predictions And Metrics Evaluation For All Data"""
    
    classes = ['normal','deformed']
    anomaly_scores = []
    val_labels = []
    predictions = []
    val_labels_binary = []
    model.eval()
    with torch.no_grad():
        for val_image, label in testloader:
            val_x_hat = model(val_image.to('cuda'))
            loss = myMSE(val_x_hat.detach().cpu(),val_image)
            anomaly_scores.append(loss.numpy())
            val_labels_binary.append(label.detach().cpu().numpy())

    anomaly_scores = np.concatenate(anomaly_scores)
    val_labels_binary = np.concatenate(val_labels_binary)
    val_labels = [classes[label] for label in val_labels_binary]

    roc_auc = roc_auc_score(val_labels_binary, anomaly_scores)
    precision, recall, thresholds = precision_recall_curve(val_labels_binary, anomaly_scores)
    F1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
    )
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    f1_score = F1_scores[np.argmax(F1_scores)]
    predictions  = (anomaly_scores >= optimal_threshold).astype(int)
    predictions = ['deformed' if _ == 1 else 'normal' for _ in predictions]
    save_confusion_matrix(val_labels, predictions, classe_path,latent_space)
    save_validation_losses_density(anomaly_scores, val_labels,classe_path, latent_space)
    # saving roc curve
    save_roc_curve(val_labels_binary, anomaly_scores, latent_space, classe_path)


    return roc_auc, f1_score



if __name__ == '__main__':
    start = timeit.default_timer()
    test()
    stop = timeit.default_timer()
    print('Inference Time: ', stop - start)  
