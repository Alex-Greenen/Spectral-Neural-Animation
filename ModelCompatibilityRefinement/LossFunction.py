import torch
from ModelCompatibilityRefinement.Database import DataBase
from ModelCompatibilityRefinement.LossType import ModelCompatibilityRefinementLossType
from ProcessData.TrainingLoss import TrainingLoss
from ProcessData.Utils import *
import Format

def LossFunction(latents, trueLatents, poses_perFrame:torch.Tensor, true_poses:torch.Tensor, nextLFposes:torch.Tensor, genLoss:torch.Tensor, database:DataBase, frames_per_ep) -> TrainingLoss:
    
    skeleton = Format.skeleton
    rotation = Format.rotation

    latent = vector_loss(latents, trueLatents)

    R_r, Root_Height_r, Root_Position_Velocity_r, Root_Rotation_xz_r, Root_RotationVel_y_r, contacts_r = torch.split(poses_perFrame, database.poseDimDiv, dim=-1)
    R_t, Root_Height_t, Root_Position_Velocity_t, Root_Rotation_xz_t, Root_RotationVel_y_t, contacts_t = torch.split(true_poses, database.poseDimDiv, dim=-1)

    R_r = reformat(R_r)
    R_r = correct(R_r, rotation)
    X_r = getX(R_r, skeleton, rotation)
    feet_r = torch.flatten(X_r[..., Format.footIndices, :],-2,-1)
    X_r = torch.flatten(X_r,-2,-1)
    
    R_t = reformat(R_t)
    R_t = correct(R_t, rotation)
    X_t = getX(R_t, skeleton, rotation)
    feet_t = torch.flatten(X_t[..., Format.footIndices, :],-2,-1)
    X_t = torch.flatten(X_t,-2,-1)

    R_lf, Root_Height_lf, Root_Position_Velocity_lf, Root_Rotation_xz_lf, Root_RotationVel_y_lf, _ = torch.split(nextLFposes, database.poseDimDiv, dim=-1)
    R_lf = reformat(R_lf)
    R_lf = correct(R_lf, rotation)
    LF_loss = angle_loss(R_lf, R_t[:, Format.deltaT-1::Format.deltaT], Format.rotation) / len(nextLFposes[0]) + \
        vector_loss(Root_Height_t[:, Format.deltaT-1::Format.deltaT], Root_Height_lf) / len(nextLFposes[0]) + \
        vector_loss(Root_Position_Velocity_t[:, Format.deltaT-1::Format.deltaT], Root_Position_Velocity_lf) / len(nextLFposes[0]) + \
        vector_loss(Root_Rotation_xz_t[:, Format.deltaT-1::Format.deltaT], Root_Rotation_xz_lf) / len(nextLFposes[0]) + \
        vector_loss(Root_RotationVel_y_t[:, Format.deltaT-1::Format.deltaT], Root_RotationVel_y_lf) / len(nextLFposes[0])

    angular = angle_loss(R_r, R_t, rotation)
    fk = vector_loss(X_r, X_t)
    posVel = vector_loss(Root_Position_Velocity_r, Root_Position_Velocity_t)
    RotXZ = vector_loss(Root_Rotation_xz_r, Root_Rotation_xz_t)
    RotVelY = vector_loss(Root_RotationVel_y_r, Root_RotationVel_y_t)
    height = vector_loss(Root_Height_r, Root_Height_t)
    feet = vector_loss(feet_r, feet_t)
    contacts = crossEntropy_loss(contacts_r, contacts_t)

    DirectReconSmooth = angular_velocity_loss(R_r[:, 1:-1], torch.roll(R_r,1,dims=-2)[:, 1:-1], R_t[:, 1:-1], torch.roll(R_t,1,dims=-2)[:, 1:-1], rotation)
    FkReconSmooth = vector_loss(torch.roll(X_r,1,dims=-2)[:, 1:-1]-X_r[:, 1:-1], torch.roll(X_t,1,dims=-2)[:, 1:-1]-X_t[:, 1:-1]) 
    PosVelSmooth = vector_loss(torch.roll(Root_Position_Velocity_r,1,dims=-2)[:, 1:-1]-Root_Position_Velocity_r[:, 1:-1], torch.roll(Root_Position_Velocity_t,1,dims=-2)[:, 1:-1]-Root_Position_Velocity_t[:, 1:-1]) 
    RotXZSmooth = vector_loss(torch.roll(Root_Rotation_xz_r,1,dims=-2)[:, 1:-1]-Root_Rotation_xz_r[:, 1:-1], torch.roll(Root_Rotation_xz_t,1,dims=-2)[:, 1:-1]-Root_Rotation_xz_t[:, 1:-1]) 
    RotVelYSmooth = vector_loss(torch.roll(Root_RotationVel_y_r,1,dims=-2)[:, 1:-1]-Root_RotationVel_y_r[:, 1:-1], torch.roll(Root_RotationVel_y_t,1,dims=-2)[:, 1:-1]-Root_RotationVel_y_t[:, 1:-1]) 
    HeightSmooth = vector_loss(torch.roll(Root_Height_r,1,dims=-2)[:, 1:-1]-Root_Height_r[:, 1:-1], torch.roll(Root_Height_t,1,dims=-2)[:, 1:-1]-Root_Height_t[:, 1:-1]) 
    FeetSmooth = vector_loss(torch.roll(feet_r,1,dims=-2)[:, 1:-1]-feet_r[:, 1:-1], torch.roll(feet_t,1,dims=-2)[:, 1:-1]-feet_t[:, 1:-1]) 

    return TrainingLoss(ModelCompatibilityRefinementLossType, [genLoss, latent, LF_loss, angular, fk, posVel, RotXZ, RotVelY, height, feet, contacts, DirectReconSmooth, FkReconSmooth, PosVelSmooth, RotXZSmooth, RotVelYSmooth, HeightSmooth, FeetSmooth])

