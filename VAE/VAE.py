import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import functional as F

from ProcessData.NormData import *


class Encoder(Module):
    def __init__(self, featureDim:int, poseDim:int, latentDim:int):
        super(Encoder, self).__init__()
        # Define params        
        self.poseDim = poseDim   
        self.featureDim = featureDim
        self.latentDim = latentDim

        self.encoder = torch.nn.Sequential(
            nn.Linear(self.poseDim+self.featureDim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 2 * self.latentDim)
        )

    def forward(self, pose, feature):
        pose = norm_pose(pose)
        feature = norm_feature(feature)
        if self.training: feature = dropoutFeatureExt(feature)
        mean, logvar = torch.split(self.encoder(torch.cat((pose,feature),-1)), self.latentDim, dim = -1)
        return  mean, logvar

class Decoder(Module):
    def __init__(self, poseDim:int, latentDim:int):
        super(Decoder, self).__init__()
        # Define params        
        self.poseDim = poseDim   
        self.latentDim = latentDim

        self.decoder = torch.nn.Sequential(
            nn.Linear(self.latentDim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, self.poseDim),
        )

    def forward(self, latentCode):
        output = denorm_pose(self.decoder(latentCode))
        return  output

class VAE(Module):
    # This is the AutoEncoder Class
    def __init__(self, featureDim:int, poseDim:int, latentDim:int):
        super(VAE, self).__init__()

        # Define params        
        self.poseDim = poseDim   
        self.featureDim = featureDim
        self.latentDim = latentDim

        # Define encoding convolutional layers
        self.encoder = Encoder(featureDim, poseDim, latentDim)
        self.decoder = Decoder(poseDim, latentDim)
        
    def forward(self, pose, feature):
        
        mean, logvar = self.encoder(pose, feature)

        if self.training: latentCode = torch.randn_like(mean) * logvar.mul(0.5).exp() + mean
        else: latentCode = mean

        output = self.decoder(latentCode)
    
        return output, latentCode, logvar
    
    def export(self, folder = "ONNX_networks", fileEncode = "VAE_Encoder",  fileDecoder = "VAE_Decoder"):
        torch.onnx.export(self.encoder,
                (torch.zeros(1, self.poseDim), torch.zeros(1, self.featureDim)),
                "{}/{}.onnx".format(folder, fileEncode),
                input_names= ['Pose', 'Feature'], 
                output_names = ['Latent', 'LogVar'], 
                export_params=True,
                do_constant_folding=True,
                opset_version=9)
        
        torch.onnx.export(self.decoder,
                torch.zeros(1, self.latentDim),
                "{}/{}.onnx".format(folder, fileDecoder),
                input_names= ['Latent'], 
                output_names = ['Pose'], 
                export_params=True,
                do_constant_folding=True,
                opset_version=9)


# class Encoder(Module):
#     # This is the AutoEncoder Encoder Class
#     def __init__(self, inputDim:int, outputDim:int):
#         super(Encoder, self).__init__()

#         # Define params        
#         self.inputDim = inputDim   
#         self.outputDim = outputDim

#         # Define encoding layers
#         self.L1 = nn.Linear(inputDim, 2 * inputDim)
#         self.L2 = nn.Linear(2 * inputDim, 2 * inputDim)
#         self.L3 = nn.Linear(2 * inputDim, 2 * inputDim)
#         self.L4 = nn.Linear(2 * inputDim, 2 * inputDim)
#         self.L5 = nn.Linear(inputDim, outputDim)

#     def forward(self, x):
#         ret = F.elu(self.L1(x))
#         ret = F.elu(self.L3(F.elu(self.L2(ret)))) + ret
#         ret = self.L5(F.elu(self.L4(ret)))
#         return x

# class Decoder(Module):
#     # This is the AutoEncoder Decoder Class
#     def __init__(self, inputDim:int, outputDim:int):
#         super(Decoder, self).__init__()

#         # Define params        
#         self.inputDim = inputDim   
#         self.outputDim = outputDim

#         # Define decoding layers
#         self.L1 = nn.Linear(inputDim, 2 * outputDim)
#         self.L2 = nn.Linear(2 * outputDim, 2 * outputDim)
#         self.L3 = nn.Linear(2 * outputDim, 2 * outputDim)
#         self.L4 = nn.Linear(2 * outputDim, 2 * outputDim)
#         self.L5 = nn.Linear(2 * outputDim, outputDim)

#     def forward(self, x):
#         ret = F.elu(self.L1(x))
#         ret = F.elu(self.L3(F.elu(self.L2(ret)))) + ret
#         ret = self.L5(F.elu(self.L4(ret)))
#         return x
