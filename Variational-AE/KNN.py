import torch

from vae import VAE
#from pl_bolts.models.autoencoders import VAE
from utils import load_train_data, save_validation_losses_density_KNN, save_confusion_matrix_KNN, save_originals, save_reconstructions, save_loss
from utils import get_indexes, save_roc_curve_KNN, save_tsne
from data import transform_data

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from tqdm import tqdm

import os
import sys
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import NearestNeighbors
from utils import load_train_data


MVTEC_DATASET = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper']
latent_spaces = [128,256,512]
k_list = [1,3,5,10]

def parse_args():
    parser = argparse.ArgumentParser('VAE')
    parser.add_argument("-d","--data", type=str, default='all')
    parser.add_argument("--save_path", type=str, default="./results/validation/without_recons_weight/KNN/")
    parser.add_argument("-ls","--latent_space", type=int, default=256)
    parser.add_argument("-k","--k_num", type=int, default=5)
    return parser.parse_args()

############################
# Main Test Function 
# Launches Tests for differentes values of K (nearest neighbors numbers) and latent space dimension
# Saves results in an excel file
##############################
def test():
    args = parse_args()

    k = args.k_num
    latent_spaces = [args.latent_space]
    save_path = args.save_path
    data_path = './dataset/'
    mvtec_classes = [args.data]
    if args.data == 'all':
        mvtec_classes = MVTEC_DATASET

    total_roc_auc = np.zeros((len(mvtec_classes), len(latent_spaces)*len(k_list)))
    total_f1_score = np.zeros((len(mvtec_classes), len(latent_spaces)*len(k_list)))
    for i, classe in enumerate(mvtec_classes):
        for j, latent_space in enumerate(latent_spaces):
            model = VAE(latent_dim=latent_space).to('cuda')
            model.load_state_dict(torch.load(f'results/Training/without_recons_weight/models/{classe}/latent_dim_{latent_space}.pt'))
            model.eval()

            transform_train, transform_test = transform_data(classe,False), transform_data(classe, False)
            transform = {'train' : transform_train, 'test' : transform_test}
            trainloader, testloader = load_train_data(transform, classe,data_path, 32, 32)

            classe_path = save_path + 'classes/' + classe
            for indx, k in enumerate(k_list):
                roc_auc, f1_score = KNN(trainloader, testloader,model,k, classe_path, classe, latent_space)
                total_roc_auc[i][j*len(k_list)+indx] = roc_auc
                total_f1_score[i][j*len(k_list)+indx] = f1_score
    
    indexes = get_indexes(k_list, latent_spaces)

    metrics_path = os.path.join(save_path,'metrics',args.data)
    if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
    df = pd.DataFrame(total_roc_auc.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([f'({latent},{k})' for latent,k in indexes]))
    df.to_excel(metrics_path + f'/roc_auc_{latent_space}.xlsx')
    df = pd.DataFrame(total_f1_score.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([f'({latent},{k})' for k,latent in indexes]))
    df.to_excel(metrics_path + f'/f1_score_{latent_space}.xlsx')

def KNN(trainloader, testloader,model,k, classe_path, classe,latent_space):
    features = []
    for images, labels in tqdm(trainloader, f'{classe} features | K = {k} | '):
        encoded = model.encoder(images.to('cuda'))
        mu = model.fc_mu(encoded)
        log_var = model.fc_var(encoded)
        std = torch.exp(log_var / 2)
        features_batch = np.array([torch.distributions.Normal(mu,std).rsample().detach().cpu().numpy() for i in range(10)]).mean(axis=0)

        features.append(features_batch)


    features = np.concatenate(features, axis=0)

    Nk = NearestNeighbors(n_neighbors=k)
    Nk.fit(features)

    classes = ['normal','deformed']
    anomaly_scores = []
    test_labels = []
    val_labels_binary = []
    test_features = []
    for images, labels in tqdm(testloader, 'Anomaly scores | test data | '):
        encoded = model.encoder(images.to('cuda'))
        mu = model.fc_mu(encoded)
        log_var = model.fc_var(encoded)
        std = torch.exp(log_var / 2)
        test_feature = np.array([torch.distributions.Normal(mu,std).rsample().detach().cpu().numpy() for i in range(10)]).mean(axis=0)
        test_features.append(test_feature)
        anomaly_scores.append(Nk.kneighbors(test_feature)[0].mean(axis=1))
        val_labels_binary.append(labels.detach().cpu().numpy())
    

    test_features = np.concatenate(test_features)
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
    
    save_confusion_matrix_KNN(val_labels, predictions, classe_path, latent_space, k)
    save_validation_losses_density_KNN(anomaly_scores, val_labels,classe_path, latent_space, k)
    # saving roc curve
    save_roc_curve_KNN(val_labels_binary, anomaly_scores, latent_space, classe_path, k)

    if k == 1: #temporary solution
        save_tsne(test_features, val_labels, classe_path)






    return roc_auc, f1_score



if __name__ == '__main__':
    test()
