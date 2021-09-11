import torch
import Format
import torch

## THESE ARE INITIALISED BY THE DATABASES WHEN THEY ARE CREATED
meanPose = torch.tensor([])
stdPose = torch.tensor([])
meanFeature = torch.tensor([])         
stdFeature  = torch.tensor([])  

def dropoutFeatureExt(feature, p=0.4):
    base, extr = torch.split(feature, Format.dropoutseperation, dim=-1)
    extr2 = torch.dropout(extr, p, True)
    return torch.cat((base, extr2), dim=-1)

def norm_feature(inp:torch.Tensor):
    return (inp - meanFeature) / (stdFeature + 1e-5)

def denorm_feature(inp:torch.Tensor):
    return inp * stdFeature + meanFeature

def norm_pose(inp:torch.Tensor):
    return (inp - meanPose) / (stdPose + 1e-5)

def denorm_pose(inp:torch.Tensor):
    return inp * stdPose + meanPose
    