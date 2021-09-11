from ProcessData.Skeleton import Skeleton
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch

from ProcessData.TrainingLoss import TrainingLoss
from LowFrequency.DataBase import DataBase
from ProcessData.Utils import *

import Format

def plotState(latent_tm1:torch.tensor, latent_est:torch.tensor, pose_goal:torch.tensor, errors: TrainingLoss, database:DataBase) -> None:
    """ This function plots the output of the low frequency network

    Args:
        pose_tm1 (torch.tensor?): pose one dT ago
        pose_Est (torch.tensor): estimated pose of current frame
        pose_Gt (torch.tensor): true pose of current frame
        errors (TrainingLoss?): errors of the model
        database (DataBase): Database used to generate data
    """
    
    fig = plt.figure(figsize=(12,8), dpi= 100)
    skeleton = Format.skeleton
    parents = skeleton._parents
    rotation = Format.rotation

    # Truth Figure
    ax = fig.add_subplot(1, 2 if errors!=None else 1, 1, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 200)

    pose_tm1 = database.AE_network.decoder(latent_tm1)
    R = torch.split(pose_tm1, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    for i in range(len(X)):
        p = parents[i]
        if p != -1:
            ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 0.3, c='black')

    R = torch.split(pose_goal, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 0.8, c='red')
        
    pose_Est = database.AE_network.decoder(latent_est)
    R = torch.split(pose_Est, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], alpha = 1, c='green')
    
    # Errors
    if errors!= None:
        ax = fig.add_subplot(1, 2, 2)
        ax.bar(errors.getNames(), errors.makefloat())
        plt.xticks(rotation= 45, fontsize = 10) 

    plt.show(block=False)
    