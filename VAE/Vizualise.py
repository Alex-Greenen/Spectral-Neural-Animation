import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from time import sleep

from VAE.TrainingLoss import TrainingLoss
from VAE.DataBase import DataBase
from ProcessData.Utils import *
import Format

def plotState(recon_1:torch.Tensor, true_1:torch.Tensor, recon_2:torch.Tensor, true_2:torch.Tensor, dt:float, errors: TrainingLoss, database:DataBase):

    fig = plt.figure(figsize=(12,8), dpi= 100)
    skeleton = parents = Format.skeleton
    parents = skeleton._parents
    rotation = Format.rotation

    # Truth Figure
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 200)

    # True 1
    R = torch.split(true_1, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))
    #ax.scatter(X[:,2], X[:,1], X[:,0], alpha = 0.2, c='g')
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], c='g')
    
    # True 2
    R = torch.split(true_2, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))

    #ax.scatter(X[:,2], X[:,1], X[:,0], c='r')
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], c='r')
    
    # Recon Figure
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(0, 200)

    # Recon 1
    R = torch.split(recon_1, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))

    #ax.scatter(X[:,2], X[:,1], X[:,0], alpha = 0.2, c='g')
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], c='g')
    
    # Recon 2
    R = torch.split(recon_2, database.poseDimDiv, dim=-1)[0]
    R = reformat(R)
    R = correct(R, rotation)
    X = np.reshape(getX(R, skeleton, rotation).clone().detach().cpu().numpy()[0], (-1,3))

    #ax.scatter(X[:,2], X[:,1], X[:,0], c='r')
    for i in range(len(X)):
          p = parents[i]
          if p != -1:
              ax.plot([X[i,2], X[p,2]], [X[i,1], X[p,1]], [X[i,0], X[p,0]], c='r')

    # Errors
    ax = fig.add_subplot(2, 2, 2)
    halfidx = int(len(errors.getNames())/2)
    ax.bar(errors.getNames()[:halfidx], errors.makefloat()[:halfidx])
    plt.xticks(rotation= 45, fontsize = 10) 
    ax = fig.add_subplot(2, 2, 4)
    ax.bar(errors.getNames()[halfidx:], errors.makefloat()[halfidx:])
    plt.xticks(rotation= 45, fontsize = 10) 

    plt.show(block=False)
    plt.pause(dt)
    sleep(dt)
    plt.close()
    