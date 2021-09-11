import numpy as np
import torch
from torch.utils.data import Dataset
import os

from VAE.VAE import VAE
from VAE.DataBase import DataBase as VAE_DataBase
from ProcessData.Utils import *
from ProcessData.Skeleton import Skeleton

import Format
from ProcessData import NormData

class DataBase(Dataset):

    def __init__(self, source_dir, vae_dir):
        self.fileNames = []
        loadedFiles = []
        self.fileLengths = []


        for file in os.listdir(source_dir):
            if os.path.splitext(file)[-1] == '.PoseData': 
                self.fileNames.append(os.path.splitext(file)[0])
                loadedFiles.append(torch.load(source_dir+'/'+file))
                self.fileLengths.append(loadedFiles[-1]['N_Frames'])

        print('Found Files: '+ str(self.fileNames))
        
        self.length = sum(self.fileLengths) - Format.epRange[1] * len(self.fileLengths)

        self.fileLengths= torch.tensor(self.fileLengths)
        
        self.poseDim = 0
        self.poseDimDiv = []
        self.poses = []

        self.featureDim = 0
        self.featureDimDiv = []
        self.features = []
        
        self.latents = []

        for c in Format.poseComponents:
            self.poseDimDiv.append(len(loadedFiles[0][c][0]))
        self.poseDim = sum(self.poseDimDiv)

        for c in Format.featureComponents:
            self.featureDimDiv.append(len(loadedFiles[0][c][0]))
        self.featureDim = sum(self.featureDimDiv)

        for f in range(len(loadedFiles)):
            for t in range(loadedFiles[f]['N_Frames']):
                frame_pose = []
                for c in Format.poseComponents:
                    frame_pose.append(loadedFiles[f][c][t])
                self.poses.append(torch.cat(frame_pose))
                                
                frame_feature = []
                for c in Format.featureComponents:
                    frame_feature.append(loadedFiles[f][c][t])
                self.features.append(torch.cat(frame_feature))

        self.AE_network = VAE(self.featureDim, self.poseDim, Format.latentDim)
        self.AE_network.load_state_dict(torch.load(vae_dir, map_location=torch.device('cpu')))
        self.AE_network.eval()

        for p in self.AE_network.parameters(): p.requires_grad = False

        self.poses = torch.vstack(self.poses)
        self.features = torch.vstack(self.features)

        NormData.stdPose, NormData.meanPose = torch.std_mean(self.poses, dim=0, keepdim=True)
        NormData.stdFeature, NormData.meanFeature = torch.std_mean(self.features, dim=0, keepdim=True)
        
        with torch.no_grad(): self.latents = self.AE_network.encoder(self.poses, self.features)[0]

        del loadedFiles
    
    def __len__(self):
        return self.length

    def get_decoder(self):
        return self.AE_network.decoder
    
    def get_encoder(self):
        return self.AE_network.encoder

    def __getitem__(self, idx):
        deltaT = np.random.randint(Format.epRange[0], Format.epRange[1])
        cumsum = torch.cumsum(self.fileLengths - deltaT, 0)
        file_idx = torch.where(cumsum > idx)[0][0].item()
        frame = idx + (deltaT) * file_idx
        
        features_tm1 = self.features[frame]
        code_tm1 = self.latents[frame]
        code_t = self.latents[frame + deltaT]
        pose_t = self.poses[frame + deltaT]
        deltaT = (deltaT - Format.epRange[0] + 1)

        return features_tm1, code_tm1, code_t, pose_t, torch.tensor([deltaT])
