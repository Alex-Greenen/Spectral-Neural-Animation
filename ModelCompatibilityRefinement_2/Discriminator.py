import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.activation import Sigmoid
from typing import Tuple

class Discriminator(Module):
    def __init__(self, poseDim:int, sequenceLength:int = 12):
        """ Used https://github.com/dandelin/Temporal-GAN-Pytorch/blob/master/model.py"""
        super(Discriminator, self).__init__()
        
        self.sequenceLength = sequenceLength
        self.poseDim = poseDim

        self.convolutions = nn.Sequential(
            nn.Conv1d(poseDim, 128, 4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(-2, -1))

        self.linear_size = int(self.convolutions(torch.zeros(1,poseDim,sequenceLength)).size(-1))

        self.linearLayers = nn.Sequential(
            nn.Linear(self.linear_size, 1),
            )

    def forward(self, inputFrames:torch.Tensor) -> torch.Tensor:
        x = inputFrames.permute(0,2,1)
        x = self.convolutions(x)
        x = self.linearLayers(x)
        return torch.reshape(x, (inputFrames.size(0), 1))
