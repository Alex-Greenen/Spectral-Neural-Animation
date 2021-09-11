import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional  as F
import torch.nn.init as init
import math
from torch.nn import Module

from ProcessData.NormData import *


class LowFrequency(Module):
    def __init__(self, featureDim:int, latentDim:int):
        super(LowFrequency, self).__init__()

        self.featureDim = featureDim
        self.latentDim = latentDim
        self.n_input = self.featureDim*2+self.latentDim
        self.n_output = self.latentDim
        self.hg = 32
        self.h = 512

        self.gating0 = nn.Linear(self.n_input, self.hg)
        self.gating1 = nn.Linear(self.hg, self.hg)
        self.gating2 = nn.Linear(self.hg, 3)
        
        self.LL10 = nn.Linear(self.n_input, self.h)
        self.LL11 = nn.Linear(self.n_input, self.h)
        self.LL12 = nn.Linear(self.n_input, self.h)

        self.LL20 = nn.Linear(self.h, self.h)
        self.LL21 = nn.Linear(self.h, self.h)
        self.LL22 = nn.Linear(self.h, self.h)

        self.LL30 = nn.Linear(self.h, self.h)
        self.LL31 = nn.Linear(self.h, self.h)
        self.LL32 = nn.Linear(self.h, self.h)

        self.LL40 = nn.Linear(self.h, self.n_output)
        self.LL41 = nn.Linear(self.h, self.n_output)
        self.LL42 = nn.Linear(self.h, self.n_output)

    def forward(self, feature_1:torch.Tensor, feature_2:torch.Tensor, latent_last:torch.Tensor) -> torch.Tensor:

        feature_1 = norm_feature(feature_1)
        feature_2 = norm_feature(feature_2)

        if self.training: 
            feature_1 = dropoutFeatureExt(feature_1)
            feature_1 = torch.dropout(feature_1, 0.2, self.training)
            feature_2 = dropoutFeatureExt(feature_2)
        
        x = torch.cat((feature_1, feature_2, latent_last), dim=-1)

        # Gating function
        Gate = F.elu(self.gating0(x))
        Gate = F.elu(self.gating1(Gate))
        Gate = F.softmax(self.gating2(Gate), dim=-1).unsqueeze(-2)

        # 3 interpolated linear layers
        x = self.LL10(x) * Gate[..., 0] + self.LL11(x) * Gate[..., 1] + self.LL12(x) * Gate[..., 2]
        x = F.relu(x)
        
        x = self.LL20(x) * Gate[..., 0] + self.LL21(x) * Gate[..., 1] + self.LL22(x) * Gate[..., 2]
        x = F.relu(x)
        
        x = self.LL30(x) * Gate[..., 0] + self.LL31(x) * Gate[..., 1] + self.LL32(x) * Gate[..., 2]
        x = F.relu(x)

        x = self.LL40(x) * Gate[..., 0] + self.LL41(x) * Gate[..., 1] + self.LL42(x) * Gate[..., 2]

        return x, Gate

    def export(self, folder = "ONNX_networks", file = "LF"):
        torch.onnx.export(self,
                (torch.zeros(1, self.featureDim), torch.zeros(1, self.featureDim), torch.zeros(1, self.latentDim)),
                "{}/{}.onnx".format(folder, file),
                verbose=True,
                input_names= ['LastFeature', 'CurrentFeature', 'CurrentLatent'], 
                output_names = ['NextLatent', 'Mode'], 
                export_params=True,
                opset_version=10)