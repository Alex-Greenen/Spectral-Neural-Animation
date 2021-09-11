import torch
import torch.nn as nn
import torch.nn.functional  as F
from torch.nn import Module
from ProcessData.NormData import *

class PhasingSize(Module):
    def __init__(self, featureDim:int, latentDim:int):
        super(PhasingSize, self).__init__()

        self.featureDim = featureDim
        self.latentDim = latentDim
        self.n_input = self.featureDim+self.latentDim
        self.n_output = self.latentDim
        self.hg = 32

        self.phasing0 = nn.Linear(self.n_input, self.hg)
        self.phasing1 = nn.Linear(self.hg, self.hg)
        self.phasing2 = nn.Linear(self.hg, 1)
    
    def forward(self, feature:torch.Tensor, latent_last:torch.Tensor) -> torch.Tensor:

        feature = norm_feature(feature)

        if self.training: 
            feature = dropoutFeatureExt(feature)

        x = torch.cat((feature, latent_last), dim=-1)

        # Gating function
        out = F.elu(self.phasing0(x))
        out = F.elu(self.phasing1(out))
        out = Format.epRange[0] + torch.sigmoid(self.phasing2(out)) * (Format.epRange[1] - Format.epRange[0])

        return out

    def export(self, folder = "ONNX_networks", file = "PhasingSize"):
        torch.onnx.export(self,
                (torch.zeros(1, self.featureDim), torch.zeros(1, self.featureDim), torch.zeros(1, self.latentDim)),
                "{}/{}.onnx".format(folder, file),
                verbose=True,
                input_names= ['LastFeature', 'CurrentFeature', 'CurrentLatent'], 
                output_names = ['PhaseSize'], 
                export_params=True,
                opset_version=10)
