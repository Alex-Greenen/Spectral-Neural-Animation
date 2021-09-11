import torch
from torch.functional import norm
import torch.nn.functional as F
import os
from ProcessData.lafan1.extract import *
from ProcessData.Skeleton import *
from ProcessData.Utils import *
from numpy import pi
from typing import Tuple
import Format

def unwrap(x):
    """From https://discuss.pytorch.org/t/np-unwrap-function-in-pytorch/34688/2"""
    y = x % (2 * pi)
    return torch.where(y > pi, 2*pi - y, y)

from ModelCompatibilityRefinement.Vizualise import motion_animation

def ProcessData(bvh_files: Tuple[str], target_path: str, frame_ranges: Tuple[int], flip:bool = False) -> None:
    """Processes BVH files at bvh_path and outputs the corresponding pose descriptors at target_path"""

    print('\n')
    fileN = -1
    for file in bvh_files:
        fileN += 1
        ranger = -1
        for framerange in frame_ranges[fileN]:
            ranger = ranger + 1
            print('Reading file {}'.format(file))
            seq_path = os.path.join('bvh_data', file)
            anim = read_bvh(seq_path)

            anim.quats = torch.FloatTensor(anim.quats)
            anim.pos = torch.FloatTensor(anim.pos)
            anim.offsets = torch.FloatTensor(anim.offsets)

            if (flip):
                signChange = torch.tensor([[[1,-1,-1, 1]]])
                tmpL = torch.clone(anim.quats[:, Format.left])
                tmpR = torch.clone(anim.quats[:, Format.right])
                anim.quats[:, Format.left] = tmpR
                anim.quats[:, Format.right] = tmpL
                anim.quats = anim.quats * signChange
                signChange = signChange[0][:,1:]
                

            heights = torch.clone(anim.pos[framerange[0]:framerange[1],0:1, 1])
            anim.offsets[0] = torch.tensor([0,0,0])
            anim.parents = torch.IntTensor(anim.parents)

            anim.pos[:,0:1] = anim.pos[:,0:1] - anim.pos[0:1,0:1]
            anim.pos[:,0:1] = quat_mul_vec(quat_inv(anim.quats[0:1,0:1]).repeat(len(anim.pos),1,1), anim.pos[:,0:1])
            anim.quats[:,0:1] = quat_mul(quat_inv(anim.quats[0:1,0:1]), anim.quats[:,0:1])

            if (flip):
                anim.pos[:,0,1] *= -1

            anim.quats = anim.quats[framerange[0]:framerange[1]]
            anim.pos = anim.pos[framerange[0]:framerange[1]]
            anim.pos[:, 0:1, 0] = heights
            
            # create skeleton
            skeleton = Format.skeleton #Skeleton(anim.offsets, anim.parents)

            print('Extracting data...')
            # get rotation and position data
            RQ = quatSeriesCont(anim.quats)
            X_global = skeleton.forward_kinematics(RQ, anim.pos[:,0,:]) # (F, J, 3)
            contacts_l, contacts_r = extract_feet_contacts(X_global, [3, 4], [7, 8], velfactor=0.4) # (F, 1)
            contacts = torch.cat((contacts_l, contacts_r), dim=-1)

            print('Manipulating data...')

            # Get Root Position and center
            Root_Position = torch.clone(anim.pos[:,0,:])  # (F, 3)
            Root_Height = torch.clone(Root_Position[:, 0:1]) # (F, 1)
            
            # Get Root Rotation and straighten
            Root_Rotation = torch.clone(RQ[:, 0, :]) # (F, 4)
            Root_Rotation_euler = quat_to_euler(Root_Rotation)
            Root_turning_vel = Root_Rotation_euler[:,0]
            Root_turning_vel = torch.FloatTensor(np.unwrap(Root_turning_vel.numpy()))
            kernel = torch.tensor([[[1,4,6,4,1]]])
            kernel = kernel / torch.sum(kernel)
            
            # SMOOTHING!!!!!
            Root_turning_vel = F.conv1d(Root_turning_vel.unsqueeze(0).unsqueeze(0), kernel, padding = 2).squeeze()

            Root_turning_vel = Root_turning_vel - torch.roll(Root_turning_vel, 1, dims=0)
            Root_hip_angle = Root_Rotation_euler[:,1:]
            RQ[:, 0] = torch.FloatTensor([1,0,0,0]) # <-- Center horizontal rotation of root to 0
            X_local = skeleton.forward_kinematics(RQ, torch.tile(torch.FloatTensor([0,0,0]), (len(RQ), 1))) # (F, J, 3)

            # Get Extremity positions / velocities
            x_feet = X_local[:,[3, 7], :] # (F, 2, 3)
            v_feet = (x_feet - torch.roll(x_feet, 1, dims=0)) # <-- CAREFUL ! HAS NOT BEEN DEVIDED BY TIME
            x_hands = X_local[:,[17, 21], :] # (F, 2, 3)
            v_hands = (x_hands - torch.roll(x_hands, 1, dims=0)) # <-- CAREFUL ! HAS NOT BEEN DEVIDED BY TIME

            tempVelocity = torch.roll(Root_Position, -1, dims=0) - Root_Position

            # Isolate up rotation
            uprotation = torch.zeros_like(Root_Rotation_euler)
            uprotation[:,0] = Root_Rotation_euler[:,0]
            uprotation = euler_to_quat(uprotation)

            # Get Future/Past 10, 20, 30 frames
            offset = 10
            future_pos = []
            future_oridir = []
            future_veldir = []
            future_vel_speed = []
            past_pos = []
            past_oridir = []
            past_veldir = []
            past_vel_speed = []
            invrot = quat_inv(uprotation)
            for i in [-3,-2,-1]:
                future_pos.append(quat_mul_vec(invrot, (torch.roll(Root_Position, i*offset, dims = 0) - Root_Position))[..., 1:])
                future_oridir.append(-quat_to_euler(quat_getDif(Root_Rotation, torch.roll(Root_Rotation, i*offset, dims = 0)))[...,0:1])
                t = quat_mul_vec(invrot, torch.roll(tempVelocity, i*offset, dims = 0))[...,1:]
                future_vel_speed.append(torch.norm(t, p=2, dim=-1).unsqueeze(-1))
                future_veldir.append(normalise(t))
            for i in [1,2,3]:
                past_pos.append(quat_mul_vec(invrot, (torch.roll(Root_Position, i*offset, dims = 0) - Root_Position))[..., 1:])
                past_oridir.append(-quat_to_euler(quat_getDif(Root_Rotation, torch.roll(Root_Rotation, i*offset, dims = 0)))[...,0:1])
                t = quat_mul_vec(invrot, torch.roll(tempVelocity, i*offset, dims = 0))[...,1:]
                past_vel_speed.append(torch.norm(t, p=2, dim=-1).unsqueeze(-1))
                past_veldir.append(normalise(t))
            
            future_pos = torch.stack(future_pos, dim = -2)
            future_oridir = torch.stack(future_oridir, dim = -2)
            future_veldir = torch.stack(future_veldir, dim = -2)
            future_vel_speed = torch.stack(future_vel_speed, dim = -2)
            past_pos = torch.stack(past_pos, dim = -2)
            past_oridir = torch.stack(past_oridir, dim = -2)
            past_veldir = torch.stack(past_veldir, dim = -2)
            past_vel_speed = torch.stack(past_vel_speed, dim = -2)

            # SMOOTH ???
            # future_pos = F.conv1d(future_pos.flatten(-2,-1).unsqueeze(-2), kernel, padding = 2).squeeze()
            # future_oridir = F.conv1d(future_oridir.flatten(-2,-1).unsqueeze(-2), kernel, padding = 2).squeeze()
            # future_veldir = F.conv1d(future_veldir.flatten(-2,-1).unsqueeze(-2), kernel, padding = 2).squeeze()
            # future_vel_speed = F.conv1d(future_vel_speed.flatten(-2,-1).unsqueeze(-2), kernel, padding = 2).squeeze()

            # Get Root Position Velocity
            Root_Position_Velocity = quat_mul_vec(quat_inv(uprotation), tempVelocity)[:,1:]

            # Remove constant root pose
            X_local = RQ[:, 1:]
            RQ = RQ[:, 1:]

            # Remove start/end 30 frames (account for the 1/2 at 30fps frames decal) and reshape to 2D:
            offset = 3 * offset
            frames = len(RQ) - offset*2

            # Make sure to go back one frame
            Root_Height_last = torch.reshape(torch.roll(Root_Height, 1, dims=0)[offset:-offset], (frames, -1))
            Root_Position_Velocity_last = torch.reshape(torch.roll(Root_Position_Velocity, 1, dims=0)[offset:-offset], (frames, -1))
            Root_hip_angle_last = torch.reshape(torch.roll(Root_hip_angle, 1, dims=0)[offset:-offset], (frames, -1))
            Root_turning_vel_last = torch.reshape(torch.roll(Root_turning_vel, 1, dims=0)[offset:-offset], (frames, -1))

            future_pos = torch.reshape(torch.roll(future_pos, 1, dims=0)[offset:-offset], (frames, -1))
            future_oridir = torch.reshape(torch.roll(future_oridir, 1, dims=0)[offset:-offset], (frames, -1))
            future_veldir = torch.reshape(torch.roll(future_veldir, 1, dims=0)[offset:-offset], (frames, -1))
            future_vel_speed = torch.reshape(torch.roll(future_vel_speed, 1, dims=0)[offset:-offset], (frames, -1))
            past_pos = torch.reshape(torch.roll(past_pos, 1, dims=0)[offset:-offset], (frames, -1))
            past_oridir = torch.reshape(torch.roll(past_oridir, 1, dims=0)[offset:-offset], (frames, -1))
            past_veldir = torch.reshape(torch.roll(past_veldir, 1, dims=0)[offset:-offset], (frames, -1))
            past_vel_speed = torch.reshape(torch.roll(past_vel_speed, 1, dims=0)[offset:-offset], (frames, -1))

            x_feet = torch.reshape(torch.roll(x_feet, 1, dims=0)[offset:-offset], (frames, -1))
            v_feet = torch.reshape(torch.roll(v_feet, 1, dims=0)[offset:-offset], (frames, -1))
            x_hands = torch.reshape(torch.roll(x_hands, 1, dims=0)[offset:-offset], (frames, -1))
            v_hands = torch.reshape(torch.roll(v_hands, 1, dims=0)[offset:-offset], (frames, -1))

            RQ = RQ[offset:-offset]
            X_local = torch.reshape(X_local[offset:-offset], (frames, -1))
            Root_Height = torch.reshape(Root_Height[offset:-offset], (frames, -1))
            Root_Position_Velocity = torch.reshape(Root_Position_Velocity[offset:-offset], (frames, -1))
            Root_hip_angle = torch.reshape(Root_hip_angle[offset:-offset], (frames, -1))
            Root_turning_vel = torch.reshape(Root_turning_vel[offset:-offset], (frames, -1))
            contacts = torch.reshape(contacts[offset:-offset], (frames, -1))

            # Now compose the descriptors
            pose_descriptor = {
                "File": str(file),
                "N_Joints": len(anim.bones),
                "FPS": 30,
                "N_Frames": frames,
                "Offsets": anim.offsets,
                "Parents": anim.parents,

                "R{}".format(rotType.Quaternion.name): torch.reshape(RQ, (frames, -1)),
                "R{}".format(rotType.SixDim.name): torch.reshape(correct(quat_to_d6(RQ), rotType.SixDim), (frames, -1)),
                "R{}".format(rotType.AngleAxis.name): torch.reshape(correct(quat_to_aa(RQ), rotType.AngleAxis), (frames, -1)),
                "X_local": X_local,
                "Height": Root_Height,
                "Root_Position_Velocity": Root_Position_Velocity,
                "Root_HipAngleRad": Root_hip_angle,
                "Root_HipTurnVelRad": Root_turning_vel,
                "Contacts": contacts,

                "Height_last": Root_Height_last,
                "Root_Position_Velocity_last": Root_Position_Velocity_last,
                "Root_HipAngleRad_last": Root_hip_angle_last,
                "Root_HipTurnVelRad_last": Root_turning_vel_last,

                "Future_pos_last": future_pos,
                "Past_pos_last": past_pos,
                "Future_ori_last": future_oridir,
                "Past_ori_last": past_oridir,
                "Future_vel_last": future_veldir,
                "Past_vel_last": past_veldir,
                "Future_speed_last": future_vel_speed, 
                "Past_speed_last": past_vel_speed, 

                "X_feet_last": x_feet,
                "X_hands_last": x_hands,
                "V_feet_last": v_feet,
                "V_hands_last": v_hands,
            }

            print('Exporting Data...')
            if not flip:
                name = target_path+'/'+os.path.splitext(file)[0]+'_'+str(ranger)+'.PoseData'
            else:
                name = target_path+'/'+os.path.splitext(file)[0]+'_flipped_'+str(ranger)+'.PoseData'
            torch.save(pose_descriptor, name)
            print('Done\n')

