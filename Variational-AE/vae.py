import os
import random
from torchvision.utils import save_image
import pytorch_lightning as pl
from torch import nn
import torch
from pl_bolts.models.autoencoders.components import (
    resnet18_decoder,
    resnet18_encoder,
)


class VAE(nn.Module):
    def __init__(self,latent_dim=512):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        #my encoder
        self.conv1 = nn.Conv2d(3,32,3,padding='same')
        self.conv2 = nn.Conv2d(32,64,3,padding='same')
        self.conv3 = nn.Conv2d(64,64,3,padding='same')
        self.conv4 = nn.Conv2d(64,128,3,padding='same')
        self.conv5 = nn.Conv2d(128,16,3,padding='same')

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2,2)
        
        #my decoder
        self.deconv1 = nn.Conv2d(int(self.latent_dim/16),16,3,padding='same')
        self.deconv2 = nn.Conv2d(16,128,3,padding='same')
        self.deconv3 = nn.Conv2d(128,64,3,padding='same')
        self.deconv4 = nn.Conv2d(64,32,3,padding='same')
        self.deconv5 = nn.Conv2d(32,3,3,padding='same')

        self.debn1 = nn.BatchNorm2d(16)
        self.debn2 = nn.BatchNorm2d(128)
        self.debn3 = nn.BatchNorm2d(64)
        self.debn4 = nn.BatchNorm2d(32)
        self.debn5 = nn.BatchNorm2d(3)

        self.upsample1 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        
        # distribution parameters
        self.fc_mu = nn.Linear(256, self.latent_dim)  # 256 inplace of 1024 for images of size 128,128
        self.fc_var = nn.Linear(256, self.latent_dim)

        #flattens
        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1,(int(latent_dim/16),4,4))

        #activations 
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        
        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

    def encoder(self,x):
        x = self.pool(self.conv1(x))
        x = self.bn1(self.relu(x))
        x = self.pool(self.conv2(x))
        x = self.bn2(self.relu(x))
        x = self.pool(self.conv3(x))
        x = self.bn3(self.relu(x))
        x = self.pool(self.conv4(x))
        x = self.bn4(self.relu(x))
        x = self.pool(self.conv5(x))
        x = self.bn5(self.relu(x))
        x = self.flatten(x)
        return x
    
    def decoder(self, x):
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.upsample2(x)  #upsample2 for images with resize 128,128
        x = self.debn1(self.relu(x))
        x = self.deconv2(x)
        x = self.upsample2(x)
        x = self.debn2(self.relu(x))
        x = self.deconv3(x)
        x = self.upsample2(x)
        x = self.debn3(self.relu(x))
        x = self.deconv4(x)
        x = self.upsample2(x)
        x = self.debn4(self.relu(x))
        x = self.deconv5(x)
        x = self.upsample2(x)
        x = self.debn5(self.sigmoid(x))
        return x
        
    def gaussian_likelihood(self, x_hat, logscale, x):
        scale = torch.exp(logscale)
        mean = x_hat
        dist = torch.distributions.Normal(mean, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z, mu, std):
        # --------------------------
        # Monte carlo KL divergence
        # --------------------------
        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        # 2. get the probabilities from the equation
        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)
        return kl
    
    def loss_elbo(self, x, x_hat, z, mu, std):
        kl = self.kl_divergence(z, mu, std)
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale, x)
        elbo = (kl - recon_loss)
        return elbo.mean()
    
    def reparametrize(self,mu,log_var):
        #Reparametrization Trick to allow gradients to backpropagate from the 
        #stochastic part of the model
        sigma = torch.exp(0.5*log_var)
        z = torch.distributions.Normal(torch.zeros_like(mu),1).rsample()
        
        return mu + sigma*z
        
    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        std = torch.exp(log_var / 2)
        
        z = self.reparametrize(mu,log_var)
        
        x_hat = self.decoder(z)
        
        #elbo = self.loss_elbo(x, x_hat, z, mu, std)
        
        return x_hat, z, mu, std

        