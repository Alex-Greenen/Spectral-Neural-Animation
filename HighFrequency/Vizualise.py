from ProcessData.Skeleton import Skeleton
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from typing import Tuple

from HighFrequency.LossType import HighFrequencyLossType
from HighFrequency.DataBase import DataBase
from ProcessData.Utils import *
import Format

def plotState(last_pose:torch.tensor, next_pose:torch.tensor, latent_true:torch.tensor, latent_interpolated:torch.tensor, errors: Tuple[float], database:DataBase) -> None:
    
    fig = plt.figure(figsize=(12,8), dpi= 100)
    skeleton = Format.skeleton
    parents = skeleton._parents
    rotation = Format.rotation

    # Truth Figure
    ax = fig.add_subplot(1, 2 if errors!=None else 1, 1, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 200)

    for l in latent_true:
        pose_gt = database.AE_network.decoder(l.unsqueeze(0))
        R = torch.split(pose_gt, database.poseDimDiv, dim=-1)[0]
        R = reformat(R)
        R = correct(R, rotation)
        X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
        for i in range(len(X)):
            p = parents[i]
            if p != -1:
                ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 0.5, linewidth=0.5, c='black')
    
    for l in latent_interpolated:
        pose_gt = database.AE_network.decoder(l.unsqueeze(0))
        R = torch.split(pose_gt, database.poseDimDiv, dim=-1)[0]
        R = reformat(R)
        R = correct(R, rotation)
        X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
        for i in range(len(X)):
            p = parents[i]
            if p != -1:
                ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 1.0, linewidth=0.5, c='green')
            
    R = torch.split(last_pose, database.poseDimDiv, dim=-1)[0]
    R = reformat(R.unsqueeze(0))
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    for i in range(len(X)):
        p = parents[i]
        if p != -1:
            ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 1.0, linewidth=0.5, c='red')

    R = torch.split(next_pose, database.poseDimDiv, dim=-1)[0]
    R = reformat(R.unsqueeze(0))
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 1.0, linewidth=0.5, c='red')

    # Errors
    if errors!= None:
        ax = fig.add_subplot(1, 2, 2)
        ax.bar([str(l.name) for l in list(HighFrequencyLossType)], errors)
        plt.xticks(rotation= 45, fontsize = 10) 

    plt.show(block=False)
    