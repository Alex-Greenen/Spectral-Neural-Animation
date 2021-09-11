import torch
from torch.nn import functional as F
from ProcessData.Skeleton import Skeleton
from ProcessData.TrainingLoss import TrainingLoss
from ProcessData.Utils import *
from VAE.DataBase import DataBase
from VAE.LossType import VAELossType
import Format


def LossFunction(recon_1, true_1, mu_1, log_var_1,
                  recon_2, true_2, mu_2, log_var_2,
                  database:DataBase):

        # Compare first Frame
        R_r1, Root_Height_r1, Root_Position_Velocity_r1, Root_Rotation_xz_r1, Root_RotationVel_y_r1, contacts_r1 = torch.split(recon_1, database.poseDimDiv, dim=-1)
        R_r1 = reformat(R_r1)
        R_r1 = correct(R_r1, Format.rotation)
        X_r1 = getX(R_r1, Format.skeleton, Format.rotation)
        feet_r1 = torch.flatten(X_r1[..., Format.footIndices, :],-2,-1)
        X_r1 = torch.flatten(X_r1, -2, -1)

        R_t1, Root_Height_t1, Root_Position_Velocity_t1, Root_Rotation_xz_t1, Root_RotationVel_y_t1, contacts_t1 = torch.split(true_1, database.poseDimDiv, dim=-1)
        R_t1 = reformat(R_t1)
        R_t1 = correct(R_t1, Format.rotation)
        X_t1 = getX(R_t1, Format.skeleton, Format.rotation)
        feet_t1 = torch.flatten(X_t1[..., Format.footIndices, :],-2,-1)
        X_t1 = torch.flatten(X_t1, -2, -1)

        latent = -0.5 * torch.mean(1 + log_var_1 - mu_1 ** 2 - log_var_1.exp())
        DirectRecon = angle_loss(R_r1, R_t1, Format.rotation)
        FkRecon = vector_loss(X_r1, X_t1)
        PosVel = vector_loss(Root_Position_Velocity_r1, Root_Position_Velocity_t1)
        RotXZ = vector_loss(Root_Rotation_xz_r1, Root_Rotation_xz_t1)
        RotVelY = vector_loss(Root_RotationVel_y_r1, Root_RotationVel_y_t1)
        Height = vector_loss(Root_Height_r1, Root_Height_t1)
        Contacts = crossEntropy_loss(contacts_r1, contacts_t1)
        Feet = vector_loss(feet_r1, feet_t1)

        # Compare second Frame
        R_r2, Root_Height_r2, Root_Position_Velocity_r2, Root_Rotation_xz_r2, Root_RotationVel_y_r2, _ = torch.split(recon_2, database.poseDimDiv, dim=-1)
        R_r2 = reformat(R_r2)
        R_r2 = correct(R_r2, Format.rotation)
        X_r2 = getX(R_r2, Format.skeleton, Format.rotation)
        feet_r2 = torch.flatten(X_r2[..., Format.footIndices, :],-2,-1)
        X_r2 = torch.flatten(X_r2, -2, -1)
        
        R_t2, Root_Height_t2, Root_Position_Velocity_t2, Root_Rotation_xz_t2, Root_RotationVel_y_t2, _ = torch.split(true_2, database.poseDimDiv, dim=-1)
        R_t2 = reformat(R_t2)
        R_t2 = correct(R_t2, Format.rotation)
        X_t2 = getX(R_t2, Format.skeleton, Format.rotation)
        feet_t2 = torch.flatten(X_t2[..., Format.footIndices, :],-2,-1)
        X_t2 = torch.flatten(X_t2, -2, -1)

        # Get Inter-frame velocity errors
        LatentSmooth = vector_loss(mu_1, mu_2)
        DirectReconSmooth = angular_velocity_loss(R_r1, R_r2, R_t1, R_t2, Format.rotation)
        FkReconSmooth = vector_loss(X_r2-X_r1, X_t2-X_t1)
        PosVelSmooth = vector_loss(Root_Position_Velocity_r2-Root_Position_Velocity_r1, Root_Position_Velocity_t2-Root_Position_Velocity_t1)
        RotXZSmooth = vector_loss(Root_Rotation_xz_r2-Root_Rotation_xz_r1, Root_Rotation_xz_t2-Root_Rotation_xz_t1)
        RotYVelSmooth = vector_loss(Root_RotationVel_y_r2-Root_RotationVel_y_r1, Root_RotationVel_y_t2-Root_RotationVel_y_t1)
        HeightSmooth = vector_loss(Root_Height_r2-Root_Height_r1, Root_Height_t2-Root_Height_t1)
        FeetSmooth = vector_loss(feet_r1-feet_r2, feet_t1-feet_t2) 

        return TrainingLoss(VAELossType, [latent, DirectRecon, FkRecon, PosVel, RotXZ, RotVelY, Height, Feet, LatentSmooth, DirectReconSmooth, FkReconSmooth, PosVelSmooth, RotXZSmooth, RotYVelSmooth, HeightSmooth, FeetSmooth, Contacts])