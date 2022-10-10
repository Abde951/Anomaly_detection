import streamlit as st
import pandas as pd
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from ae import AE
from vae import VAE
from camera import Basler
import time
import os
from PIL import Image
import imageio
from PIL import Image
from matplotlib import cm

from patchcore.prediction import compute_optimal_score, prediction



PATH = "./results/validation/ERCON"
DATA_PATH = "./dataset/"
taken_photos_path = "./images/train/"
ae_latent_space = 128
vae_latent_space = 256
classe = 'bottle'
loss_function = torch.nn.MSELoss()
CLASSES = ['NORMAL','DEFORMED']
i = 0
@st.cache
def incremente(i):
    return i+1

left_column, right_column = st.columns(2)


st.title('Anomaly detection')
st.text('')

#side bar
models = ['PatchCore','Variational AE','Convolutional AE']
menu = ["camera","automatic","manuel"]
DATASETS = ['roulement','seringues','bottle']
image = Image.open('figures/uca.png').resize((200,200))
st.sidebar.image(image)
st.sidebar.header("Menu")
model_choice = st.sidebar.selectbox("Select Model",models)
dataset = st.sidebar.selectbox('dataset',DATASETS)
uploading_choice = st.sidebar.selectbox("Uploading images",menu)





def calc_anomaly_score(choice,image,dataset):
    ## Load model

    if choice == 'Convolutional AE':
        ae = AE(latent_dim=ae_latent_space,scale=1).to('cuda')
        ae.load_state_dict(torch.load(f'models/AE/{dataset}/latent_dim_{ae_latent_space}.pt'))
        ae.eval()
        return calc_anomaly_score_AE(ae,image)

    elif choice == 'Variational AE':
        vae = VAE(latent_dim=vae_latent_space).to('cuda')
        vae.load_state_dict(torch.load(f'models/VAE/{dataset}/latent_dim_{vae_latent_space}.pt'))
        vae.eval()
        return calc_anomaly_score_VAE(vae,image)

    elif choice == 'PatchCore':
        return compute_optimal_score(dataset)



def main():
   
    if uploading_choice == "automatic":
        automatic(model_choice,dataset)

    if uploading_choice == "manuel":
        manuel(model_choice,dataset)

    if uploading_choice == "camera":
        camera(model_choice,dataset)



def manuel(model_choice,dataset):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        ])
    def load_image(image_file):
        img = Image.open(image_file)
        return img
    label = st.radio('select label : ',["normal","deformed"])
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])
    label = 0 if label == "normal" else 1
    if image_file is not None:
        image = load_image(image_file)
        image = transform(image)
        pred(model_choice,image.unsqueeze(0),label,dataset)


def automatic(model_choice,dataset):
    if model_choice in ['Variational AE','Convolutional AE','PatchCore']:
        ## Load test image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        test_path = DATA_PATH + dataset + '/test'
        testset = torchvision.datasets.ImageFolder(test_path, transform = transform)
        if st.button('take an image'):
            testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle = True, num_workers=5)
            image , label = next(iter(testloader))
            st.image(image.squeeze().permute(1,2,0).numpy())

            pred(model_choice,image,label.item(),dataset,method='automatic')

    # elif model_choice == 'PatchCore':
    #     compute_optimal_score(classname=dataset)
    #     predicted = prediction(classname=dataset)
    #     st.text(f'predicted class is {predicted} ')

def camera(model_choice,dataset):
    count  = 1
    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor()
    ])
    if st.button('Take an image'):
        img_array = Basler()
        if model_choice =='Variational AE' or model_choice == 'Convolutional AE':
            exist = os.path.exists(taken_photos_path + f'image_{count}.png')
            while exist:
                count += 1
                exist = os.path.exists(taken_photos_path + f'image_{count}.png')
            imageio.imsave(taken_photos_path + f'image_{count}.png', img_array)
        elif model_choice == 'PatchCore':
            imageio.imsave('./dataset/'+dataset+'_captured/test/captured/' + 'image.png', img_array)

        st.text('image saved !')
        image_PIL = Image.fromarray(img_array)
        image = transform(image_PIL)
        image = image.unsqueeze(0)
        st.subheader('test image')
        st.image(image.squeeze().permute(1,2,0).numpy())
        pred(model_choice, image, 0, dataset)
    


def pred(model_choice,image,label,dataset,method='camera'):
    thresholds = {'bottle':0.0014,'roulement':0.107, 'seringues':0.1}
    start_time = time.time()
    anomaly_score = calc_anomaly_score(model_choice,image,dataset)
    prediction = int(anomaly_score > thresholds[dataset])
    inference_time = time.time() - start_time
    
    # if method == 'automatic':
    #     st.text('ground_truth : {}'.format(CLASSES[label]))
    # st.text('anomaly score: {:+.5f}'.format(anomaly_score))
    # st.text('prediction : {}'.format(CLASSES[prediction]))
    # st.text('inference time : {}'.format(inference_time))
    
    results = {
        'Anomaly score': [float("{:.4f}".format(anomaly_score))],
        'Prediction': [CLASSES[prediction]],
        'Inference time (s)': [float("{:.4f}".format(inference_time))],
    }
    df = pd.DataFrame(results).reset_index(drop=True)
    st.table(df)
    
    ######## Saving results in a csv file ##########

    # app_results_path = 'app_results/'+model_choice+'/'+dataset+'/'
    # os.makedirs(app_results_path,exist_ok=True)
    # if os.path.exists(app_results_path+'df.csv'):
    #     old_df = pd.read_csv(app_results_path+'df.csv')
    #     old_df = old_df.loc[:,['ground_truth','anomaly_score','prediction']]  
    #     df = pd.concat([df,old_df],axis=0)
    #     open(app_results_path+'df.csv', 'w').write(df.to_csv())
    # else:
    #     open(app_results_path+'df.csv', 'w').write(df.to_csv())

    # accuracy = (df['prediction'] == df['ground_truth']).sum()
    # st.text('accuracy : {}/{}'.format(accuracy,len(df['prediction'])))


def calc_anomaly_score_AE(model,image):
    output = model(image.to('cuda'))
    anomaly_score = loss_function(output,image.to('cuda')).item()
    return anomaly_score

def calc_anomaly_score_VAE(model,image):
    output, _, _, _ = model(image.to('cuda'))
    anomaly_score = loss_function(output,image.to('cuda')).item()
    return anomaly_score




if __name__ == '__main__':
    main()    