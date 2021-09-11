from LowFrequency.DataBase import DataBase
from ProcessData.Utils import rotType
import torch
import Format


def PCA_svd(X, k, center=True):
    """From https://www.programmersought.com/article/70225667343/"""
    s, m = torch.std_mean(X)
    X = (X-m)/(s+1e-6)
    n = X.size()[0]
    ones = torch.ones(n).view([n,1])
    h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
    H = torch.eye(n) - h
    X_center =  torch.mm(H.double(), X.double())
    u, s, v = torch.svd(X_center)
    components  = v[:k]
    return components

db = DataBase('TrainingData', 'VAE/VAE_'+Format.name+'.pymodel')

torch.set_printoptions(precision=10)
print(PCA_svd(db.latents, 2))

