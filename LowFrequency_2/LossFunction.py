import torch
from torch.nn import functional as F
from ProcessData.Skeleton import Skeleton
from ProcessData.TrainingLoss import TrainingLoss
from ProcessData.Utils import *
from LowFrequency_2.DataBase import DataBase
from LowFrequency_2.LossType import LowFrequencyLossType
import Format

def LossFunction(latentRecon:torch.Tensor, latentTrue:torch.Tensor, pose_goal:torch.Tensor, database:DataBase) -> TrainingLoss:

    skeleton = Format.skeleton
    rotation = Format.rotation
    AE_network = database.AE_network

    # Compare first Frame
    with torch.no_grad(): 
        R_r, Root_Height_r1, Root_Position_Velocity_r1, Root_Rotation_xz_r1, Root_RotationVel_y_r1, contacts_r1  = torch.split(AE_network.decoder(latentRecon), database.poseDimDiv, dim=-1)
        R_t, Root_Height_t1, Root_Position_Velocity_t1, Root_Rotation_xz_t1, Root_RotationVel_y_t1, contacts_t1  = torch.split(pose_goal, database.poseDimDiv, dim=-1)

    R_r = reformat(R_r)
    R_r = correct(R_r, rotation)
    X_r = getX(R_r, skeleton, rotation)
    feet_r = torch.flatten(X_r[..., Format.footIndices, :],-2,-1)
    X_r = torch.flatten(X_r,-2,-1)
    
    R_t = reformat(R_t)
    R_t = correct(R_t, rotation)
    X_t = getX(R_t, skeleton, rotation)
    feet_t = torch.flatten(X_t[..., Format.footIndices, :],-2,-1)
    X_t = torch.flatten(X_t, -2, -1)

    latent = vector_loss(latentRecon, latentTrue)
    angular = angle_loss(R_r, R_t, rotation)
    fk = vector_loss(X_r, X_t)
    posVel = vector_loss(Root_Position_Velocity_r1, Root_Position_Velocity_t1)
    RotXZ = vector_loss(Root_Rotation_xz_r1, Root_Rotation_xz_t1)
    RotVelY = vector_loss(Root_RotationVel_y_r1, Root_RotationVel_y_t1)
    height = vector_loss(Root_Height_r1, Root_Height_t1)
    Contacts = crossEntropy_loss(contacts_r1, contacts_t1)
    Feet = vector_loss(feet_r, feet_t)

    return TrainingLoss(LowFrequencyLossType, [latent, angular, fk, posVel, RotXZ, RotVelY, height, Feet, Contacts])