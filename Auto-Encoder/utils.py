import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from sklearn.manifold import TSNE


def load_train_data(transform, classe,data_path, train_batch_size=8, test_batch_size=1):

    """ 
    Loads and returns train and test dataloaders.
    """

    train_path = data_path + classe + '/train'
    trainset = torchvision.datasets.ImageFolder(train_path, transform = transform['train'])
    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle = True, num_workers=5)

    test_path = data_path + classe + '/test'
    testet = torchvision.datasets.ImageFolder(test_path, transform = transform['test'])
    testloader = DataLoader(testet, batch_size=test_batch_size, shuffle = False)

    return trainloader, testloader


def save_validation_losses_density_every_epoch(epoch, val_losses, val_labels,classe_path):

    """ Saves loss density every epoch """


    if not os.path.exists(classe_path + "/density_images/"):
            os.makedirs(classe_path + "/density_images/")
    df = pd.DataFrame(val_losses, columns=['loss'])
    df['labels'] = val_labels
    #df_def = pd.DataFrame(losses_def, columns=['loss_def'])
    sns.displot(df,x="loss" ,hue="labels")
    save_path = classe_path + f"/density_images/Density-epoch-{epoch+1}.png"
    plt.savefig(save_path)


def save_validation_losses_density(val_losses, val_labels,classe_path, latent_space):

    """ Saves loss density """

    if not os.path.exists(classe_path):
            os.makedirs(classe_path)
    df = pd.DataFrame(val_losses, columns=['loss'])
    df['labels'] = val_labels
    #df_def = pd.DataFrame(losses_def, columns=['loss_def'])
    sns.displot(df,x="loss" ,hue="labels")
    save_path = classe_path + f"/Density_latent_dim_{latent_space}.png"
    plt.savefig(save_path)


def save_confusion_matrix(labels, predictions,classe_path, latent_space):

    """ Saves confusion matrix """

    if not os.path.exists(classe_path):
            os.makedirs(classe_path)
    classes = ['normal','deformed']
    matrix = confusion_matrix(labels, predictions, labels=classes)
    annot = [["00","01"],["10","11"]]
    for i in range(2):
        for j in range(2):
            percentage = matrix[i][j] / sum(matrix[i]) * 100
            annot[i][j] = f"{matrix[i][j]} ({percentage:.1f}%)"

    # Plot the confusion matrix with Seaborn
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot()

    # annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(matrix, annot=annot, square=True, fmt="s", ax=ax)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels([classes[0], classes[1]])
    ax.yaxis.set_ticklabels([classes[0], classes[1]])
    plt.savefig(classe_path + f"/confusion_matrix_latent_dim_{latent_space}.png")



def save_originals(images,epoch,classe_path):

    """ Saves Original Images """

    batch_size = images.shape[0]
    fig=plt.figure()
    for i in range(1,batch_size+1):
        img = images[i-1].permute(1,2,0).detach().cpu().numpy()
        fig.add_subplot(1,batch_size,i)
        plt.imshow(img)
    if not os.path.exists(classe_path + '/reconstructions/'):
        os.makedirs(classe_path + '/reconstructions/')
    plt.savefig(classe_path + f'/reconstructions/origin-epochs-{epoch+1}.png')

def save_reconstructions(images,epoch,classe_path):

    """ Saves Reconstructed Images """

    batch_size = images.shape[0]
    fig=plt.figure()
    for i in range(1,batch_size+1):
        img = images[i-1].permute(1,2,0).detach().numpy()
        fig.add_subplot(1,batch_size,i)
        plt.imshow(img)
    if not os.path.exists(classe_path + '/reconstructions/'):
        os.makedirs(classe_path + '/reconstructions/')
    plt.savefig(classe_path + f'/reconstructions/recons-epochs-{epoch+1}.png')



def save_loss(losses, classe_path):

    """ Saves loss at the end of training """

    fig=plt.figure()
    plt.plot(losses)
    plt.savefig(classe_path + '/losses.png')


def save_roc_curve(val_labels_binary, anomaly_scores, latent_space, classe_path):

    """ Saves The ROC Curve """

    plt.figure()
        # PLOTING ROC CURVE
    fpr, tpr, _ = roc_curve(val_labels_binary,  anomaly_scores)

    #create ROC curve
    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(classe_path + f'/roc_curve_latent_dim_{latent_space}.png' )


def save_tsne(test_features, val_labels_binary, classe_path, latent_space):

    tsne = TSNE(n_components=2)
    embedded = tsne.fit_transform(test_features)

    df_tsne = pd.DataFrame(embedded[:,0], columns=['one'])
    df_tsne['two'] = embedded[:,1]
    df_tsne['labels'] = val_labels_binary


    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="one", y="two",
    hue="labels",
    palette=sns.color_palette("hls", 2),
    data=df_tsne,
    legend="full",
)
    plt.savefig(classe_path + f'/tsne_latent_dim_{latent_space}.png')


def get_indexes(k_list, latent_spaces):
    indexes = []
    for l in latent_spaces:
        for k in k_list:
            indexes.append((l,k))
    return indexes