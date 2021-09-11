import numpy as np
import torch
from torch.utils.data import Dataset
from ProcessData.Utils import rotType
from ProcessData.Skeleton import Skeleton
import Format
from ProcessData import NormData

import os


class DataBase(Dataset):
    
    def __init__(self, source_dir):
        self.fileNames = []
        loadedFiles = []
        self.fileLengths = []

        self.deltaT = 1

        for file in os.listdir(source_dir):
            if os.path.splitext(file)[-1] == '.PoseData': 
                self.fileNames.append(os.path.splitext(file)[0])
                loadedFiles.append(torch.load(source_dir+'/'+file))
                self.fileLengths.append(loadedFiles[-1]['N_Frames'])

        print('Found Files: '+ str(self.fileNames))
        
        self.length = sum(self.fileLengths) - self.deltaT * len(self.fileLengths)
        
        self.fileLengths = torch.tensor(self.fileLengths)

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

        del loadedFiles

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx): 
 
        cumsum = torch.cumsum(self.fileLengths-self.deltaT, 0)
        file_idx = torch.where(cumsum > idx)[0][0].item()
        frame_idx = idx + self.deltaT * file_idx

        return self.poses[frame_idx + self.deltaT], self.features[frame_idx + self.deltaT], self.poses[frame_idx], self.features[frame_idx]




