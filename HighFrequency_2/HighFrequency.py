import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from typing import Tuple
from ProcessData.NormData import norm_feature


from ProcessData.NormData import *

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_to_ingate = nn.Linear(input_size, hidden_size)
        self.input_to_forgetgate = nn.Linear(input_size, hidden_size)
        self.input_to_cellgate = nn.Linear(input_size, hidden_size)
        self.input_to_outgate = nn.Linear(input_size, hidden_size)

        self.hidden_to_ingate = nn.Linear(hidden_size, hidden_size, bias=False) 
        self.hidden_to_forgetgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_to_cellgate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_to_outgate = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, inp:torch.Tensor, hx:torch.Tensor, cx:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        ingate = self.input_to_ingate(inp) + self.hidden_to_ingate(hx)
        forgetgate = self.input_to_forgetgate(inp) + self.hidden_to_forgetgate(hx)
        cellgate = self.input_to_cellgate(inp) + self.hidden_to_cellgate(hx)
        outgate = self.input_to_outgate(inp) + self.hidden_to_outgate(hx)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, hy, cy


class MHUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(MHUCell, self).__init__()

        assert (input_size == hidden_size)

        self.Wz= nn.Linear(input_size, hidden_size)
        self.Wv = nn.Linear(hidden_size, hidden_size)
        self.Uvh = nn.Linear(hidden_size, hidden_size)

    def forward(self, inp:torch.Tensor, hx:torch.Tensor) -> torch.Tensor:

        v = self.Wv(torch.relu(self.Uvh(hx)))
        z = torch.sigmoid(self.Wz(torch.relu(inp)))
        out = (1-z) * v + z * inp

        return out, v

class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.input_to_reset = nn.Linear(input_size, hidden_size)
        self.input_to_update = nn.Linear(input_size, hidden_size)
        self.input_to_candidate = nn.Linear(input_size, hidden_size)

        self.hidden_to_reset = nn.Linear(hidden_size, hidden_size) 
        self.hidden_to_update = nn.Linear(hidden_size, hidden_size)
        self.outhidden_to_candidate = nn.Linear(hidden_size, hidden_size)

    def forward(self, inp:torch.Tensor, hx:torch.Tensor) -> torch.Tensor:

        resetgate = torch.sigmoid(self.input_to_reset(inp) + self.hidden_to_reset(hx))
        updategate = torch.sigmoid(self.input_to_update(inp) + self.hidden_to_update(hx))
        candidategate = torch.tanh(self.input_to_candidate(inp) + self.outhidden_to_candidate(resetgate * hx))
        hiddengate = (1-updategate) * hx + updategate * candidategate

        return hiddengate, hiddengate

class HighFrequency(Module):

    def __init__(self, latentDim:int, featureDim:int, feature_red:int = 10, latent_red:int = 512):
        super(HighFrequency, self).__init__()
        self.latentDim = latentDim
        self.featureDim = featureDim
        self.latent_red = latent_red
        self.feature_red = feature_red

        self.Time_embedding_add = nn.Linear(3, 3 * self.latentDim + 3)

        self.LL_in = nn.Linear(3 * self.latentDim + 3, self.latent_red)
        self.LSTM = LSTMCell(self.latent_red , self.latent_red)
        self.LL_out = nn.Linear(self.latent_red, self.latentDim)

    def forward(self, feature:torch.Tensor, lastlatent:torch.Tensor, nextlatent:torch.Tensor, time:torch.tensor, h1:torch.Tensor, h2:torch.Tensor, current_state:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        t = time[...,1]
        time[...,1] = (t - Format.epRange[0]) / (Format.epRange[1] - Format.epRange[0])
        t = time[...,0].unsqueeze(-1)
        time = torch.cat((time,t), dim=-1)

        inp = torch.cat((current_state, nextlatent, nextlatent - current_state, time), dim=-1) + self.Time_embedding_add(time)

        in1 = torch.tanh(self.LL_in(inp))

        out1, h1o, h2o = self.LSTM(in1, h1, h2)

        correction = self.LL_out(out1)

        current_state_up = current_state + correction

        return current_state_up, h1o, h2o, current_state_up

    def export(self, folder = "ONNX_networks", file = "HF"):
        torch.onnx.export(self, 
                (torch.zeros(1, self.featureDim), torch.zeros(1, self.latentDim), torch.zeros(1, self.latentDim), torch.zeros(1, 2), torch.zeros(1, self.latent_red), torch.zeros(1, self.latent_red), torch.zeros(1, self.latent_red)),
                "{}/{}.onnx".format(folder, file),
                verbose=True,
                input_names= ['Feature', 'LatentLast', 'LatentNext', 'Time_Frames', 'hidden1_in', 'hidden2_in', 'hidden3_in'], 
                output_names= ['Output', 'hidden1_out', 'hidden2_out', 'hidden3_out'], 
                export_params=True,
                opset_version=10)