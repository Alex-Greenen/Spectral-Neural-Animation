import torch
from torch.utils.data import Dataset
import os
from typing import Tuple
from ProcessData.Utils import *
from ProcessData import NormData
import numpy as np
import Format
from VAE.VAE import VAE

class DataBase(Dataset):

    def __init__(self, source_dir:str, sequenceLengthLong:int = 30, sampling_down_factor:float=0.1):
        self.fileNames = []
        loadedFiles = []
        self.fileLengths = []
        self.latentDim = Format.latentDim
        self.rotation = Format.rotation
        self.deltaT = Format.deltaT
        self.sequenceLengthLong = sequenceLengthLong
        self.sampling_down_factor = sampling_down_factor

        self.skeleton = Format.skeleton

        for file in os.listdir(source_dir):
            if os.path.splitext(file)[-1] == '.PoseData': 
                self.fileNames.append(os.path.splitext(file)[0])
                loadedFiles.append(torch.load(source_dir+'/'+file))
                self.fileLengths.append(loadedFiles[-1]['N_Frames'])

        print('Found Files: '+ str(self.fileNames))
        
        self.length = sum(self.fileLengths) - self.deltaT * (self.sequenceLengthLong+1) * len(self.fileLengths)

        self.fileLengths= torch.tensor(self.fileLengths)
    
        self.poseDim = 0
        self.poseDimDiv = []
        self.poses = []

        self.featureDim = 0
        self.featureDimDiv = []
        self.features = []
        
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

        self.poses = torch.vstack(self.poses)
        self.features = torch.vstack(self.features)

        NormData.stdPose, NormData.meanPose = torch.std_mean(self.poses, dim=0, keepdim=True)
        NormData.stdFeature, NormData.meanFeature = torch.std_mean(self.features, dim=0, keepdim=True)

        self.AE_network = VAE(self.featureDim, self.poseDim, Format.latentDim)
        self.AE_network.load_state_dict(torch.load("VAE/VAE_"+Format.name+".pymodel", map_location=torch.device('cpu')))
        self.AE_network.eval()
        for p in self.AE_network.parameters(): p.requires_grad = False
        with torch.no_grad(): self.latents = self.AE_network.encoder(self.poses,self.features)[0]

        del loadedFiles
    
    def __len__(self) -> int:
        return int(self.length * self.sampling_down_factor)

    def __getitem__(self, idx:int) -> Tuple[torch.Tensor]:

        # idx = min(int(idx / self.sampling_down_factor), self.length-1)
        idx = np.random.randint(self.length)

        cumsum = torch.cumsum(self.fileLengths-self.deltaT * self.sequenceLengthLong, 0)
        file_idx = torch.where(cumsum > idx)[0][0].item()
        frame = idx + (self.deltaT * self.sequenceLengthLong) * file_idx

        features = []
        true_poses = []
        true_latents = []

        for i in range(0, self.deltaT * self.sequenceLengthLong):
            features.append(self.features[frame + i])
            true_poses.append(self.poses[frame + i])
            true_latents.append(self.latents[frame + i])
        
        features = torch.vstack(features)
        true_poses = torch.vstack(true_poses)
        true_latents = torch.vstack(true_latents)

        return features, true_poses, true_latents