import torch 
from ProcessData.Skeleton import Skeleton
from ModelCompatibilityRefinement.Database import DataBase
from typing import Tuple
from ProcessData.Utils import *
import Format

def updateFeature(inFeature: torch.Tensor, pose: torch.Tensor, database:DataBase, lastFeet: torch.Tensor, lastHands: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:

    joints, Root_Height, Root_Position_Velocity, Root_Rotation_xz, Root_RotationVel_y, Contacts = torch.split(pose, database.poseDimDiv, dim=-1)
    
    joints = reformat(joints)
    joints = correct(joints, Format.rotation)
    X_local = getX(joints, Format.skeleton, Format.rotation)
    X_local = X_local.reshape(-1, 22, 3)

    x_feet = X_local[:,[3, 7], :].flatten(-2,-1)
    x_hands = X_local[:,[17, 21], :].flatten(-2,-1)

    v_feet = x_feet - lastFeet

    v_hands = x_hands - lastHands

    f = torch.split(inFeature, database.featureDimDiv, dim=-1)[4:12]

    outFeature = torch.cat((Root_Height, Root_Position_Velocity, Root_Rotation_xz, Root_RotationVel_y, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], x_feet, v_feet, x_hands, v_hands), dim = -1)

    return outFeature, x_feet, x_feet

def getExtremeties(pose: torch.Tensor, database:DataBase)-> Tuple[torch.Tensor,torch.Tensor]:
    joints = torch.split(pose, database.poseDimDiv, dim=-1)[0]

    joints = reformat(joints)
    joints = correct(joints, Format.rotation)
    X_local = getX(joints, Format.skeleton, Format.rotation)
    X_local = X_local.reshape(-1, 22, 3)

    x_feet = X_local[:,[3, 7], :].flatten(-2,-1)
    x_hands = X_local[:,[17, 21], :].flatten(-2,-1)
    return x_feet, x_hands




