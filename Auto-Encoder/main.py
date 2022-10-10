import torch
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from ae import AE
from utils import load_train_data, save_validation_losses_density_every_epoch, save_confusion_matrix, save_originals, save_reconstructions, save_loss
from data import transform_data

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

import os
import sys
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def parse_args():
    parser = argparse.ArgumentParser('AE')
    parser.add_argument("--classe", type=str, default='bottle')
    parser.add_argument("--save_path", type=str, default="./results_new/Training/")
    parser.add_argument("-ls","--latent_space", type=int, default=512)
    parser.add_argument("-e","--epochs", type=int, default=500)
    return parser.parse_args()



ALL_CLASSES_1 = ['pill','screw','tile','toothbrush','transistor','wood','zipper']
ALL_CLASSES_2 = ['bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut']
MVTEC_DATASET = ['pill','screw','tile','toothbrush','transistor','wood','zipper','bottle','cable','capsule','carpet','grid','hazelnut','leather','metal_nut']



def main():
    args = parse_args()

    epochs = args.epochs
    latent_spaces = [args.latent_space]
    save_path = args.save_path
    data_path = './dataset/'
    mvtec_classes = [args.classe]

    if args.classe == 'all1':
        mvtec_classes = ALL_CLASSES_1
    if args.classe == 'all2':
        mvtec_classes = ALL_CLASSES_2

    total_roc_auc = np.zeros((len(mvtec_classes), len(latent_spaces)))
    total_f1_score = np.zeros((len(mvtec_classes), len(latent_spaces)))
    for i, classe in enumerate(mvtec_classes):
        for j, latent_space in enumerate(latent_spaces):
            roc_auc, f1_score = run(save_path, data_path, latent_space,classe,epochs)
            total_roc_auc[i][j] = roc_auc
            total_f1_score[i][j] = f1_score
    
    if not os.path.exists(save_path + '/metrics/'):
            os.makedirs(save_path + '/metrics/')
    df = pd.DataFrame(total_roc_auc.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([str(latent) for latent in latent_spaces]))
    df.to_excel(save_path + 'metrics/roc_auc.xlsx')
    df = pd.DataFrame(total_f1_score.transpose(), columns=mvtec_classes)
    df = df.set_index(pd.Series([str(latent) for latent in latent_spaces]))
    df.to_excel(save_path + 'metrics/f1_score.xlsx')



def run(save_path, data_path,latent_space, classe,epochs):

    """ 
    Launches Model Traing and Validation for a given Data and latent space dimension 
    
    Parameters :
    classe : Data to train on the autoencoder model
    latent_space : latent space dimenson of the AE model

    Returns:
    roc_auc, f1_score : metrics for the given data and latent space dimension
    """

    ## Getting Train Data
    transform_train, transform_test = transform_data(classe), transform_data(classe)
    transform = {'train' : transform_train, 'test' : transform_test}
    trainloader, testloader = load_train_data(transform, classe,data_path, 8,1)

    ## Model
    model = AE(latent_dim=latent_space,scale=1).to('cuda')
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()

    ## PATHS
    classe_path = save_path + 'reports/' + classe + f'/latent_dim_{latent_space}'
    affiche_tqdm_class = classe + f'/latent_dim_{latent_space}'

    losses = train(model,opt,loss_function, trainloader, testloader, epochs,classe_path,affiche_tqdm_class)
    save_loss(losses,classe_path)

    ## Getting Metrics (AUC and F1SCORE)
    roc_auc, f1_score = validate(model, testloader, loss_function, classe_path,latent_space)
    
    ## Saving Model
    if not os.path.exists(save_path + f'models/{classe}/'):
            os.makedirs(save_path + f'models/{classe}/')
    torch.save(model.state_dict(), save_path + f'models/{classe}/latent_dim_{latent_space}.pt')

    return roc_auc, f1_score


#################
### Training function
def train(model,opt,loss_function, trainloader, testloader, epochs, classe_path, affiche_tqdm_class):

    """
    Training the AutoEncoder model with validation every a fixed number of epochs 

    Returns:
    losses: validation loss after the last epoch of training
    """

    classes = ['normal','deformed']
    losses = []
    step_loss = 0
    #for epoch in tqdm(range(epochs) , f' |{affiche_tqdm_class}|'):
    for epoch in range(epochs):
        model.train()    
        
        for images, labels in tqdm(trainloader,affiche_tqdm_class+f'| epoch {epoch} | loss : {step_loss}'):
        #for images, labels in trainloader:
            x_hat = model(images.to('cuda'))
            loss = loss_function(x_hat, images.to('cuda'))
            opt.zero_grad()
            #step_loss += loss.item()
            loss.backward()
            opt.step()
            
            losses.append(loss.item())
            

        step_loss = loss.item()  
    
        # Launching Validation 4 times per training
        if epoch % int(epochs/5) == int(epochs/5) - 1:    
            val_losses = []
            val_labels = []
            val_labels_binary = []
            model.eval()
            with torch.no_grad():
                for val_image, label in testloader:
                    val_x_hat = model(val_image.to('cuda'))
                    loss = loss_function(val_x_hat,val_image.to('cuda'))
                    val_losses.append(loss.item())
                    val_labels.append(classes[label.item()])
                    val_labels_binary.append(label.item())
                    

        
                # Save losses density image
                save_validation_losses_density_every_epoch(epoch, val_losses, val_labels, classe_path)
                
                
                # Save reconstruction
                save_originals(images[0].unsqueeze(0), epoch, classe_path)
                save_reconstructions(x_hat[0].unsqueeze(0).cpu(), epoch, classe_path)

                     
            
    return losses



def validate(model, testloader, loss_function, classe_path,latent_space):

    """ 
    Validation fucntion 

    returns: 
    roc_auc, f1_score : evaluation metrics for the specified latent space
     """

    classes = ['normal','deformed']
    val_losses = []
    val_labels = []
    predictions = []
    val_labels_binary = []
    model.eval()
    with torch.no_grad():
        for val_image, label in testloader:
            val_x_hat = model(val_image.to('cuda'))
            loss = loss_function(val_x_hat,val_image.to('cuda'))
            val_losses.append(loss.item())
            val_labels.append(classes[label.item()])
            val_labels_binary.append(label.item())

    roc_auc = roc_auc_score(val_labels_binary, val_losses)
    precision, recall, thresholds = precision_recall_curve(val_labels_binary, val_losses)
    F1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
    )
    optimal_threshold = thresholds[np.argmax(F1_scores)]
    f1_score = F1_scores[np.argmax(F1_scores)]
    predictions  = (val_losses >= optimal_threshold).astype(int)
    predictions = ['deformed' if _ == 1 else 'normal' for _ in predictions]
    save_confusion_matrix(val_labels, predictions,classe_path, latent_space)


    return roc_auc, f1_score



if __name__ == '__main__':
    main()
