import torch

from ae import AE
from utils import load_train_data, save_validation_losses_density, save_confusion_matrix, save_originals
from utils import save_reconstructions, save_loss, save_roc_curve, save_tsne, get_indexes
from data import transform_data

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from tqdm import tqdm

from itertools import product
import os
import sys
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import timeit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import load_train_data


MVTEC_DATASET = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper','casting','MSD','casting_augmented']
# latent_spaces = [512]
k_list = [1,3,5,10]


def parse_args():
    parser = argparse.ArgumentParser('AE')
    parser.add_argument('-d',"--data", type=str, default='all')
    parser.add_argument("--save_path", type=str, default="./results/validation/KNN/")
    parser.add_argument("-ls","--latent_space", type=int, default=256)
    parser.add_argument("-k","--k_num", type=int, default=0)
    return parser.parse_args()

############################
# Main Test Function 
# Launches Tests for differentes values of K (nearest neighbors numbers) and latent space dimension
# Saves results in an excel file
##############################
def test():
    args = parse_args()

    k_list = [args.k_num]
    if k_list[0] == 0:
        k_list = k_list
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
            model = AE(latent_dim=latent_space,scale=1).to('cuda')
            model.load_state_dict(torch.load(f'results/Training/models/{classe}/latent_dim_{latent_space}.pt'))
            model.eval()

            transform_train, transform_test = transform_data(classe,False), transform_data(classe, False)
            transform = {'train' : transform_train, 'test' : transform_test}
            trainloader, testloader = load_train_data(transform, classe,data_path, 32, 1)

            classe_path = save_path + 'classes/' + classe
            for indx, k in enumerate(k_list):
                roc_auc, f1_score = KNN(trainloader, testloader,model,k, classe_path, classe, latent_space)
                total_roc_auc[i][j*len(k_list)+indx] = roc_auc
                total_f1_score[i][j*len(k_list)+indx] = f1_score
    
    
    indexes = get_indexes(k_list, latent_spaces)

    metrics_path = save_path + '/metrics/' + args.data 
    if not os.path.exists(metrics_path):
            os.makedirs(metrics_path)
    df = pd.DataFrame(total_roc_auc.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([f'({latent},{k})' for latent,k in indexes]))
    df.to_excel(metrics_path + f'/roc_auc.xlsx')
    df = pd.DataFrame(total_f1_score.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([f'({latent},{k})' for k,latent in indexes]))
    df.to_excel(metrics_path + f'/f1_score.xlsx')



def KNN(trainloader, testloader,model,k, classe_path, classe,latent_space):
    features = []
    for images, labels in tqdm(trainloader, f'{classe} features | K = {k} | '):
        features_batch = model.encoder(images.to('cuda')).detach().cpu().numpy()
        features.append(features_batch)

    features = np.concatenate(features, axis=0)

    Nk = NearestNeighbors(n_neighbors=k)
    Nk.fit(features)

    classes = ['normal','deformed']
    anomaly_scores = []
    val_labels = []
    val_labels_binary = []
    test_features = []
    for images, labels in tqdm(testloader, 'Anomaly scores | test data | '):
        test_feature = model.encoder(images.to('cuda')).detach().cpu().numpy()
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

    save_confusion_matrix(val_labels, predictions, classe_path, latent_space)
    save_validation_losses_density(anomaly_scores, val_labels,classe_path, latent_space)
    # saving roc curve
    save_roc_curve(val_labels_binary, anomaly_scores, latent_space, classe_path)
    save_tsne(test_features, val_labels, classe_path,latent_space)




    return roc_auc, f1_score



if __name__ == '__main__':
    start = timeit.default_timer()
    test()
    stop = timeit.default_timer()
    print('Time: ', stop - start)
