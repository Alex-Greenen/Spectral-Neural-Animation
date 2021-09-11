import numpy as np
import torch
from torch.utils.data import Dataset
import os
from typing import Tuple

from VAE.VAE import VAE

from ProcessData.Skeleton import Skeleton
from ProcessData.Utils import *
from ProcessData import NormData

import Format

class DataBase(Dataset):
    
    def __init__(self, source_dir:str, vae_dir:str, sequenceLengthLong:int = 30, sampling_down_factor=0.1):
        """ Create a database for the High Frequency NN. 
        Args:
            source_dir (str): directory of pose data
            vae_dir (str): directory of VAE to use for encoding to latent code
            sequenceLengthLong (int, optional): length of training sequence in number of keyframes. Defaults to 10.
        """

        self.sampling_down_factor=sampling_down_factor
        
        self.fileNames = []
        loadedFiles = []
        self.fileLengths = []

        self.sequenceLengthLong = sequenceLengthLong

        for file in os.listdir(source_dir):
            if os.path.splitext(file)[-1] == '.PoseData': 
                self.fileNames.append(os.path.splitext(file)[0])
                loadedFiles.append(torch.load(source_dir+'/'+file))
                self.fileLengths.append(loadedFiles[-1]['N_Frames'])

        print('Found Files: '+ str(self.fileNames))
        
        self.length = sum(self.fileLengths) - Format.deltaT * self.sequenceLengthLong * len(self.fileLengths)

        self.fileLengths= torch.tensor(self.fileLengths)

        self.skeleton = Skeleton(loadedFiles[0]['Offsets'], loadedFiles[0]['Parents'])
        
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
        
        with torch.no_grad(): self.latents = self.AE_network.encoder(self.poses,self.features)[0]

        del loadedFiles
        
    
    def __len__(self) -> int:
        """Get number of possible queries to database"""
        return int(self.length * self.sampling_down_factor)

    def get_decoder(self) -> torch.nn.Module:
        """Get the decoder of the VAE network used by this DB"""
        return self.AE_network.decoder
    
    def get_encoder(self) -> torch.nn.Module:
        """Get the encoder of the VAE network used by this DB"""
        return self.AE_network.encoder

    def __getitem__(self, idx: int) -> Tuple[torch.tensor, torch.tensor]:
        """ Get data for an episode of data """

        idx = np.random.randint(self.length)

        cumsum = torch.cumsum(self.fileLengths - Format.deltaT * len(self.fileLengths), 0)
        file_idx = torch.where(cumsum > idx)[0][0].item()
        episode_start_frame = idx + Format.deltaT * self.sequenceLengthLong * file_idx

        latentLast = []
        latentNext = []
        times = []
        feature = []
        goalTensor = []
        goalPoses = []
        start_end_true = []

        for i in range(self.sequenceLengthLong-1):
            for j in range(1, Format.deltaT+1):
                latentLast.append(self.latents[episode_start_frame + i * Format.deltaT])
                latentNext.append(self.latents[episode_start_frame + (i+1) * Format.deltaT])
                times.append(torch.tensor((j-1)/Format.deltaT).unsqueeze(0))
                feature.append(self.features[episode_start_frame + i * Format.deltaT + j])
                start_end_true.append(torch.cat((self.poses[episode_start_frame + i * Format.deltaT], self.poses[episode_start_frame + (i+1) * Format.deltaT])))
                goalTensor.append(self.latents[episode_start_frame + i * Format.deltaT + j])
                goalPoses.append(self.poses[episode_start_frame + i * Format.deltaT + j])
        
        latentLast = torch.vstack(latentLast)
        latentNext = torch.vstack(latentNext)
        times = torch.vstack(times)
        feature = torch.vstack(feature)
        goalTensor = torch.vstack(goalTensor)
        goalPoses = torch.vstack(goalPoses)
        start_end_true = torch.vstack(start_end_true)

        return latentLast, latentNext, times, feature, goalTensor, goalPoses, start_end_true

