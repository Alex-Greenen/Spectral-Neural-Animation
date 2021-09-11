import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
import torch.onnx
from VAE.VAE import VAE
from HighFrequency_2.HighFrequency import HighFrequency
from LowFrequency_2.LowFrequency import LowFrequency

class ModelCompatibilityRefinement(Module):
    def __init__(self, VAE_dir:str, LF_dir:str, HF_dir:str, featureDim:int, latentDim:int, poseDim:int):
        super(ModelCompatibilityRefinement, self).__init__()
        self.featureDim = featureDim
        self.latentDim = latentDim
        self.poseDim = poseDim

        self.VAE = VAE(self.featureDim, self.poseDim, self.latentDim)
        self.VAE.load_state_dict(torch.load(VAE_dir))
        self.Encoder = self.VAE.encoder
        self.Decoder = self.VAE.decoder
        self.VAE.eval()

        self.LF = LowFrequency(self.featureDim, self.latentDim)
        self.LF.load_state_dict(torch.load(LF_dir))
        self.LF.train()
        
        self.HF = HighFrequency(self.latentDim, self.featureDim)
        self.HF.load_state_dict(torch.load(HF_dir))
        self.HF.train()

    def export_to_onnx(self, folder):
        self.VAE.eval()
        self.LF.eval()
        self.HF.eval()
        self.VAE.export(folder)
        self.LF.export(folder)
        self.HF.export(folder)
        self.VAE.train()
        self.LF.train()
        self.HF.train()

    

    
