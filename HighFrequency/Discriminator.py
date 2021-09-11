import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.modules.activation import Sigmoid
from typing import Tuple

class DiscriminatorLong(Module):
    def __init__(self, poseDim:int, sequenceLength:int = 12):
        """ Used https://github.com/dandelin/Temporal-GAN-Pytorch/blob/master/model.py"""
        super(DiscriminatorLong, self).__init__()

        self.convolutions = nn.Sequential(
            nn.Conv1d(poseDim, 128, 4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv1d(128, 256, 4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(-2, -1))

        self.linear_size = int(self.convolutions(torch.zeros(1,poseDim,sequenceLength)).size(-1)) # = (int(sequenceLength/2)-2) * 256

        self.linearLayers = nn.Sequential(
            nn.Linear(self.linear_size, 1),
            )

    def forward(self, inputFrames:torch.Tensor) -> torch.Tensor:
        x = inputFrames.permute(0,2,1)
        x = self.convolutions(x)
        x = self.linearLayers(x)
        return torch.reshape(x, (inputFrames.size(0), 1))
    
# class DiscriminatorShort(Module):
#     def __init__(self, poseDim:int):
#         """ Inspired by Robust Motion In-betweening """
#         super(DiscriminatorShort, self).__init__()

#         self.discrminatorShort = nn.Sequential(
#             nn.Linear(poseDim*2, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, inp: torch.Tensor) -> torch.Tensor:
#         return torch.reshape(self.discrminatorShort(inp), (inp.size(0), 1))


class Discriminator(Module):
    def __init__(self, poseDim:int, DiscLongSequenceLength:int = 12, interval: int = 13):
        super(Discriminator, self).__init__()

        self.poseDim = poseDim
        self.DiscLongSequenceLength = DiscLongSequenceLength
        self.interval = interval

        self.DiscLong = DiscriminatorLong(self.poseDim, self.DiscLongSequenceLength)

    def forward(self, input:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batches = input.size(0)
        frames = input.size(1)
        discriminator_score = torch.zeros(batches, 1)

        for i in range(0, frames-self.DiscLongSequenceLength, self.interval):
            discriminator_score = discriminator_score + self.DiscLong(input[:,i:i+self.DiscLongSequenceLength+1,:])

        discriminator_score = discriminator_score  / len(range(0, frames-self.DiscLongSequenceLength, self.interval))

        return discriminator_score


