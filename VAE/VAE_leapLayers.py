import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F
from torch.utils import data
from .DataBase import DataBase

class normalizing_layer(Module):
    def __init__(self, mean, std, inverse:bool = False):
        super(normalizing_layer, self).__init__()
        self.mean = mean
        self.std = std
        self.inverse = inverse
        
    def forward(self, x):
        if not self.inverse: return (x-self.mean) / (self.std + 1e-6)
        else: return x * self.std + self.mean

class VAE(Module):
    # This is the AutoEncoder Class
    def __init__(self, database:DataBase, latent_dim:int): # latent_dim MUST be a multiple of 4
        super(VAE, self).__init__()

        # Define params        
        self.poseDim = database.poseDim   
        self.featureDim = database.featureDim
        self.latent_dim = int(latent_dim / 4)
        self.database:DataBase = database

        hiddenBaseSize = 128

        self.enc1 = nn.Linear(self.poseDim, hiddenBaseSize)
        self.enc2 = nn.Linear(hiddenBaseSize, int(hiddenBaseSize/2))
        self.enc3 = nn.Linear(int(hiddenBaseSize/2), int(hiddenBaseSize/4))
        self.enc4 = nn.Linear(int(hiddenBaseSize/4), int(hiddenBaseSize/8))

        self.enc1_out = nn.Linear(int(hiddenBaseSize/1), 2*self.latent_dim)
        self.enc2_out = nn.Linear(int(hiddenBaseSize/2), 2*self.latent_dim)
        self.enc3_out = nn.Linear(int(hiddenBaseSize/4), 2*self.latent_dim)
        self.enc4_out = nn.Linear(int(hiddenBaseSize/8), 2*self.latent_dim)

        self.dec4 = nn.Linear(self.latent_dim, int(hiddenBaseSize/4))
        self.dec3 = nn.Linear(int(hiddenBaseSize/4) + self.latent_dim, int(hiddenBaseSize/2))
        self.dec2 = nn.Linear(int(hiddenBaseSize/2) + self.latent_dim, int(hiddenBaseSize))
        self.dec1 = nn.Linear(int(hiddenBaseSize/1) + self.latent_dim, self.poseDim)

        self.norm = normalizing_layer(database.meanPose, database.stdPose)
        self.denorm = normalizing_layer(database.meanPose, database.stdPose ,True)

    def encoder(self, x:torch.tensor):
        h0 = self.norm(x)
        h1 = F.elu(self.enc1(h0))
        z1m, z1std = torch.split(self.enc1_out(h1), (self.latent_dim, self.latent_dim), dim=-1)
        h2 = F.elu(self.enc2(h1))
        z2m, z2std = torch.split(self.enc2_out(h2), (self.latent_dim, self.latent_dim), dim=-1)
        h3 = F.elu(self.enc3(h2))
        z3m, z3std = torch.split(self.enc3_out(h3), (self.latent_dim, self.latent_dim), dim=-1)
        h4 = F.elu(self.enc4(h3))
        z4m, z4std = torch.split(self.enc4_out(h4), (self.latent_dim, self.latent_dim), dim=-1)

        return torch.cat((z1m, z2m, z3m, z4m, z1std, z2std, z3std, z4std), dim=-1)
    
    def decoder(self, z:torch.tensor):
        z1, z2, z3, z4 = torch.split(z, (self.latent_dim, self.latent_dim, self.latent_dim, self.latent_dim), dim=-1)

        h4 = F.elu(self.dec4(z4))
        h3 = F.elu(self.dec3(torch.cat((h4,z3), dim=-1)))
        h2 = F.elu(self.dec2(torch.cat((h3,z2), dim=-1)))
        h1 = self.dec1(torch.cat((h2,z1), dim=-1))
        h0 = self.denorm(h1)

        return h0

    def forward(self, pose, feature):
        
        mean, logvar = torch.split(self.encoder(pose), (self.latent_dim*4,self.latent_dim*4), dim = -1)

        if self.training: latentCode = torch.randn_like(mean) * logvar.mul(0.5).exp() + mean
        else: latentCode = mean
        #latentCode = mean
        output = self.decoder(latentCode)
    
        return output, latentCode, logvar
