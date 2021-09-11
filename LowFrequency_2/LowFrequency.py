import torch
import torch.nn as nn
import torch.nn.functional  as F
from torch.nn import Module
from ProcessData.NormData import *


class LowFrequency(Module):
    def __init__(self, featureDim:int, latentDim:int):
        super(LowFrequency, self).__init__()

        self.featureDim = featureDim
        self.latentDim = latentDim
        self.n_input = self.featureDim+self.latentDim + 1
        self.n_output = self.latentDim
        self.hg = 32
        self.h = 512

        self.Time_embedding = nn.Linear(1, self.n_input)

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

    def forward(self, feature:torch.Tensor, latent_last:torch.Tensor, frames_to_pred:torch.Tensor,) -> torch.Tensor:

        feature = norm_feature(feature)

        if self.training: 
            feature = dropoutFeatureExt(feature)

        t_emb = torch.tanh(self.Time_embedding(frames_to_pred))
        
        x = torch.cat((feature, latent_last, frames_to_pred), dim=-1) + t_emb

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
                (torch.zeros(1, self.featureDim), torch.zeros(1, self.latentDim), torch.zeros(1, 1)),
                "{}/{}.onnx".format(folder, file),
                verbose=True,
                input_names= ['CurrentFeature', 'CurrentLatent', 'Frames_to_pred'], 
                output_names = ['NextLatent', 'Mode'], 
                export_params=True,
                opset_version=10)