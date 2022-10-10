import os
import random
from torchvision.utils import save_image
from torch import nn
import torch
from torchsummary import summary


class AE(nn.Module):
    def __init__(self,latent_dim=32):
        super(AE, self).__init__()
        
        self.latent_dim = latent_dim
        
        #my encoder
        self.conv1 = nn.Conv2d(3,32,3,padding='same')
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32,64,3,padding='same')
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,64,3,padding='same')
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64,128,3,padding='same')
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128,16,3,padding='same')
        self.bn5 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2,2)
        
        #my decoder
        self.deconv1 = nn.Conv2d(int(self.latent_dim/16),16,3,padding='same')
        self.debn1 = nn.BatchNorm2d(16)
        self.deconv2 = nn.Conv2d(16,128,3,padding='same')
        self.debn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.Conv2d(128,64,3,padding='same')
        self.debn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.Conv2d(64,32,3,padding='same')
        self.debn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.Conv2d(32,3,3,padding='same')
        self.debn5 = nn.BatchNorm2d(3)
        self.upsample1 = nn.Upsample(scale_factor=4)
        self.upsample2 = nn.Upsample(scale_factor=2)
        
        self.fc1 = nn.Linear(1024,512)   # 1024, 512 if image size 256 and || if image size is 128 then 256,256
        self.fc2 = nn.Linear(512, self.latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)

        self.flatten = nn.Flatten()
        self.unflatten = nn.Unflatten(1,(int(latent_dim/16),4,4))

        #activations 
        self.relu = nn.ReLU()
        #self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        

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
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return x
    
    def decoder(self, x):
        x = self.unflatten(x)
        x = self.deconv1(x)
        x = self.upsample1(x)   #self.upsample2 if image size is 128 and upsample1 if 256
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
        
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    ae = AE(256,1)
    summary(ae, (3,256,256),device='cpu')        