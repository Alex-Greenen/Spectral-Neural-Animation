import torch
from numpy import pi
from enum import Enum, auto, unique

# Rotation type:

@unique
class rotType(Enum):
    Quaternion = auto()
    AngleAxis = auto()
    SixDim = auto()

# Tensor Operations

def dot(t1:torch.Tensor, t2: torch.Tensor, keepdim=True)->torch.Tensor:
    return torch.sum(t1 * t2, dim = -1, keepdim = keepdim)

def normalise(x:torch.Tensor, eps=1e-8)-> torch.Tensor:
    return x / (torch.norm(x, dim =-1).unsqueeze(-1) + eps)

# Quaternion Operations

def quat_mul(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    x0, x1, x2, x3 = x[..., 0:1], x[..., 1:2], x[..., 2:3], x[..., 3:4]
    y0, y1, y2, y3 = y[..., 0:1], y[..., 1:2], y[..., 2:3], y[..., 3:4]
    return torch.cat((
        y0 * x0 - y1 * x1 - y2 * x2 - y3 * x3,
        y0 * x1 + y1 * x0 - y2 * x3 + y3 * x2,
        y0 * x2 + y1 * x3 + y2 * x0 - y3 * x1,
        y0 * x3 - y1 * x2 + y2 * x1 + y3 * x0), dim=-1)

def quat_getDif(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    #return quat_mul(x,quat_inv(quatBiCont(x,y)))
    return quat_mul(x,quat_inv(quatBiCont(x,y)))

def quat_mul_vec(q:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    t = 2.0 * torch.cross(q[..., 1:], x, dim=-1)
    return x + q[..., 0:1]* t + torch.cross(q[..., 1:], t)  # Dim=-1 ?

def quat_inv(q:torch.Tensor) -> torch.Tensor:
    if torch.cuda.is_available(): return torch.FloatTensor([1, -1, -1, -1]).cuda() * q
    else: return torch.FloatTensor([1, -1, -1, -1]) * q

# 6D relative rotation

def d6_getDif(x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    x = correct_6D(x)
    y = correct_6D(y)
    
    x_full_transp = d6_to_RotMat(x).view(*x.size()[:-1], 3,3).transpose(-1,-2)
    y_full = d6_to_RotMat(y).view(*y.size()[:-1], 3,3)

    product6d, _ = torch.split(torch.flatten(torch.mul(y_full, x_full_transp), -2,-1), (6,3), dim=-1)

    return product6d

# Foot Contact

def extract_feet_contacts(pos, lfoot_idx, rfoot_idx, velfactor=0.02):
    """
    Extracts binary tensors of feet contacts
    :param pos: tensor of global positions of shape (Timesteps, Joints, 3)
    :param lfoot_idx: indices list of left foot joints
    :param rfoot_idx: indices list of right foot joints
    :param velfactor: velocity threshold to consider a joint moving or not
    :return: binary tensors of left foot contacts and right foot contacts
    """
    lfoot_xyz = (pos[1:, lfoot_idx, :] - pos[:-1, lfoot_idx, :]) ** 2
    contacts_l = (torch.sum(torch.sum(lfoot_xyz, axis=-1),axis=-1, keepdims=True) < velfactor)

    rfoot_xyz = (pos[1:, rfoot_idx, :] - pos[:-1, rfoot_idx, :]) ** 2
    contacts_r = (torch.sum(torch.sum(rfoot_xyz, axis=-1),axis=-1, keepdims=True) < velfactor)

    # Duplicate the last frame for shape consistency
    contacts_l= torch.vstack((contacts_l, contacts_l[-2:-1,:]))
    contacts_r= torch.vstack((contacts_r, contacts_r[-2:-1,:]))

    return contacts_l.type(torch.FloatTensor), contacts_r.type(torch.FloatTensor)

# Rotation conversions

def quat_to_d6(quats:torch.Tensor) -> torch.Tensor: # take (...,4) --> (...,6)
    """This code is adapted from https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/transforms/rotation_conversions.py"""
    r, i, j, k = torch.unbind(quats, -1)
    two_s = 2.0 / (quats * quats).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    matrix = o.reshape(quats.shape[:-1] + (3, 3))
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)

def d6_to_quat(d6:torch.Tensor) -> torch.Tensor: # take (...,6) --> (...,4)
    """ See https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/"""
    # THIS WORKS BUT HAS VERY OCCASIONAL ERRORS
    # d6 = correct_6D(d6)
    # v0, v1 = d6[..., :3].unsqueeze(-2), d6[..., 3:].unsqueeze(-2)
    # v2 = torch.cross(v0, v1, dim=-1)

    # w = torch.sqrt(1. + v0[...,0] + v1[...,1] + v2[...,2]) / 2 + 0.000001
    # ret = torch.cat(( w, (v2[...,1]-v1[...,2])/(4*w), (v0[...,2]-v2[...,0])/(4*w), (v1[...,0]-v0[...,1])/(4*w)), dim = -1)
    # ret = normalise(ret)
    # return ret

    """This code is adapted from https://github.com/facebookresearch/pytorch3d/blob/master/pytorch3d/transforms/rotation_conversions.py"""
    d6 = correct_6D(d6)
    b1, b2 = d6[..., :3], d6[..., 3:]
    b3 = torch.cross(b1, b2, dim=-1)
    matrix = torch.stack((b1, b2, b3), dim=-2)
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(*batch_dim, 9), dim=-1)

    stacked = torch.stack([
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ], dim=-1)

    q_abs = torch.zeros_like(stacked)
    positive_mask = stacked > 0
    q_abs[positive_mask] = torch.sqrt(stacked[positive_mask])
        
    quat_by_rijk = torch.stack([
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ], dim=-2)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].clip(0.1))
    return quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(*batch_dim, 4)

def aa_to_quat(aa:torch.Tensor) -> torch.Tensor: # take (...,4) --> (...,4)
    """https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm"""
    angle, axis = torch.split(aa, (1,3), dim = -1)
    return torch.cat((torch.cos(angle/2), axis * torch.sin(angle/2)), dim=-1)

def quat_to_aa(quat:torch.Tensor) -> torch.Tensor: # take (...,4) --> (...,4)
    """https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToAngle/index.htm"""
    quat = normalise(quat)
    cangle, saxis = torch.split(quat, (1,3), dim = -1)
    angle = 2 * torch.atan2(torch.norm(saxis,dim=-1).unsqueeze(-1), cangle)
    s = torch.sqrt(1-torch.square(cangle)) # assuming quaternion normalised then w is less than 1, so term always positive.
    axis = torch.where(s < 0.001, saxis, saxis/s)
    return torch.cat((angle,axis), dim=-1)

def quat_to_RotMat(quat:torch.Tensor) -> torch.Tensor: # take (...,4) --> (...,9)
    """from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/"""
    q0, q1, q2, q3 = torch.split(quat,(1,1,1,1), dim=-1)
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    # 3x3 rotation matrix
    rot_matrix = torch.cat((r00, r01, r02, r10, r11, r12, r20, r21, r22), dim=-1)
    return rot_matrix

def AA_to_RotMat(aa:torch.Tensor) -> torch.Tensor: # take (...,4) --> (...,9)
    """Converts Axis-Angle (4d) to a rotation matrix"""
    return quat_to_RotMat(aa_to_quat(aa))

def d6_to_RotMat(aa:torch.Tensor) -> torch.Tensor: # take (...,6) --> (...,9)
    """Converts 6D to a rotation matrix, from: https://github.com/papagina/RotationContinuity/blob/master/Inverse_Kinematics/code/tools.py"""

    a1, a2 = torch.split(aa, (3,3), dim=-1)
    a3 = torch.cross(a1, a2, dim=-1)
    
    return torch.cat((a1,a2,a3), dim=-1)

# Quaternion To Euler:

def quat_to_euler(q):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)
    
    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]
    
    x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
    y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1, 1))
    z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))

    return torch.stack((x, y, z), dim=1).view(original_shape)

def euler_to_quat(e, order = 'xyz'):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3
    
    original_shape = list(e.shape)
    original_shape[-1] = 4
    
    e = e.reshape(-1, 3)
    
    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]
    
    rx = torch.stack((torch.cos(x/2), torch.sin(x/2), torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    ry = torch.stack((torch.cos(y/2), torch.zeros_like(y), torch.sin(y/2), torch.zeros_like(y)), dim=1)
    rz = torch.stack((torch.cos(z/2), torch.zeros_like(z), torch.zeros_like(z), torch.sin(z/2)), dim=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = quat_mul(result, r)
            
    # # Reverse antipodal representation to have a non-negative "w"
    # if order in ['xyz', 'yzx', 'zxy']:
    #     result *= -1
    
    return result.reshape(original_shape)

# Get Local Position from Rotations

def getX_from6D(d6: torch.Tensor, skeleton, rot_offset = None, pose_offset = None) -> torch.Tensor:
    Q_rot = d6_to_quat(d6)
    return getX_fromQ(Q_rot, skeleton, rot_offset, pose_offset)

def getX_fromAA(aa: torch.Tensor, skeleton, rot_offset = None, pose_offset = None) -> torch.Tensor:
    Q_rot = aa_to_quat(aa)
    return getX_fromQ(Q_rot, skeleton, rot_offset, pose_offset)

def getX_fromQ(Q_rot: torch.Tensor, skeleton, rot_offset = None, pose_offset = None) -> torch.Tensor:
    if rot_offset == None: rot_offset = torch.tile(torch.tensor([1., 0, 0, 0]), Q_rot.shape[:-2]+(1,1))
    if torch.cuda.is_available(): rot_offset = rot_offset.cuda()
    if pose_offset == None: pose_offset = torch.tile(torch.tensor([0, 0, 0]), Q_rot.shape[:-2]+(1,))
    if torch.cuda.is_available(): pose_offset = pose_offset.cuda()
    Q_rot = torch.cat((rot_offset, Q_rot),-2)
    return skeleton.forward_kinematics(Q_rot, pose_offset)

def getX(inpRot: torch.Tensor, skeleton, rtype:rotType, rot_offset = None, pose_offset = None)-> torch.Tensor:
    if rtype == rotType.Quaternion: return getX_fromQ(inpRot, skeleton, rot_offset, pose_offset)
    elif rtype == rotType.SixDim: return getX_from6D(inpRot, skeleton, rot_offset, pose_offset)
    elif rtype == rotType.AngleAxis: return getX_fromAA(inpRot, skeleton, rot_offset, pose_offset)
    else: raise

# Correct Data

def correct_6D(d6:torch.tensor) -> torch.Tensor:
    first, second = torch.split(d6, 3, dim= -1)
    first = normalise(first)
    second = normalise(second)

    c = dot(first, second)
    c = (1-torch.sqrt(1-torch.pow(c,2))) / (c+1e-8)
    temp = first.clone()
    first = first - c * second
    second = second - c * temp

    first = normalise(first)
    second = normalise(second)

    #first = normalise(first)
    # possible second round of corrections?
    # c = dot(first, second)
    # second = second - first * c
    # second = normalise(second)

    d6 = torch.cat((first, second), -1)
    return d6

def correct_Q(q:torch.tensor) -> torch.Tensor:
    return normalise(q)

def correct_AA(aa:torch.tensor) -> torch.Tensor:
    angle, axis = torch.split(aa, (1,3), dim = -1)
    axis = normalise(axis)
    return torch.cat((angle, axis), dim=-1)

def correct(inp:torch.tensor, rtype:rotType) -> torch.Tensor:
    if rtype == rotType.Quaternion: return correct_Q(inp)
    elif rtype == rotType.SixDim: return correct_6D(inp)
    elif rtype == rotType.AngleAxis: return correct_AA(inp)
    else: raise

def quatSeriesCont(q:torch.tensor):
    """
    https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py 
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance between two consecutive frames.
    """
    dot_products = torch.sum(q[1:]*q[:-1], axis=-1)
    mask = dot_products < 0
    mask = (torch.cumsum(mask, axis=0)%2).bool().unsqueeze(-1)
    mask = torch.repeat_interleave(mask, 4, dim=-1)
    qrest = torch.where(mask, -q[1:], q[1:])
    return torch.cat((q[0:1],qrest), dim=0)

def quatBiCont(q1:torch.tensor, q2:torch.tensor):
    """Makes sure that two quaternions have 'closest' idem rotations by possibly flipping the 2nd rotation"""
    dot_products = torch.sum(q1*q2, axis=-1, keepdim=True)
    mask = dot_products < 0
    q2new = torch.where(mask, -q2, q2)
    return q2new

# Reformat data:

def reformat(inp:torch.tensor) -> torch.Tensor:
    """Reformats a tensor into (N_frames, N_joints, rotationDim)"""
    return inp.view(*inp.shape[:-1], 21, -1)

# Sub-loss functions

def aaLoss(R1:torch.Tensor, R2:torch.Tensor) -> torch.Tensor:
    """Computes L1 loss between angle axis rotations by comparing their SO3 rotations"""
    return torch.sum(torch.norm(AA_to_RotMat(R1)-AA_to_RotMat(R2), p=2, dim = -1 )) / len(R1) 

def quatLoss(R1:torch.Tensor, R2:torch.Tensor) -> torch.Tensor:
    """Computes L1 loss between quaternion rotations by comparing their SO3 rotations"""
    return torch.sum(torch.norm(quat_to_RotMat(R1) - quat_to_RotMat(R2), p=2, dim = -1)) / len(R1)

def sixLoss(R1:torch.Tensor, R2:torch.Tensor) -> torch.Tensor:
    """Computes L2 loss between 6D rotations by comparing the SO3 rotations"""
    return torch.mean(torch.norm(d6_to_RotMat(R1) - d6_to_RotMat(R2), p=2, dim=-1))

def angle_loss(R1:torch.Tensor, R2:torch.Tensor, rtype:rotType) -> torch.Tensor:
    if rtype == rotType.Quaternion: return quatLoss(R1, R2)
    elif rtype == rotType.SixDim: return sixLoss(R1, R2)
    elif rtype == rotType.AngleAxis: return aaLoss(R1, R2) 
    else: raise
    
def vector_loss(X1:torch.Tensor, X2:torch.Tensor, p:int = 2) -> torch.Tensor:
    return torch.mean(torch.norm(X1-X2, p, dim = -1))

def crossEntropy_loss(X_r:torch.Tensor, X_t:torch.Tensor) -> torch.Tensor:
    X_rs = torch.sigmoid(X_r)
    CE = - X_t * torch.log(X_rs) - (1-X_t) * torch.log(1-X_rs)
    return torch.mean(CE)

def angular_velocity_loss(R1, R2, T1, T2, rotation:rotType ):
    if rotation == rotType.Quaternion: return angle_loss(quat_getDif(R1,R2), quat_getDif(T1,T2), rotType.Quaternion)
    elif rotation == rotType.AngleAxis: return angle_loss(quat_getDif(aa_to_quat(R1),aa_to_quat(R2)), quat_getDif(aa_to_quat(T1),aa_to_quat(T2)), rotType.Quaternion)
    elif rotation == rotType.SixDim: return angle_loss(d6_getDif(R1, R2), d6_getDif(T1,T2), rotation)
    else: raise

# Get Global Positions from Pose sequence

def getX_full(poses: torch.Tensor, poseDiv, skeleton, rtype:rotType)-> torch.Tensor:
    frames = len(poses)
    

    fullX = []
    Rot, Height, PosVel, RotXZ, RotVelY, _ = torch.split(poses.float(), poseDiv, dim = -1)

    Rot = reformat(Rot)

    state_position = torch.tensor([0., 0, 0])
    state_rotation = torch.tensor([0., RotXZ[0][0], RotXZ[0][1]])

    for i in range(frames):
        rottemp = torch.tensor([state_rotation[0], 0, 0])
        veltemp = torch.tensor([0., PosVel[i][0], PosVel[i][1]]) # 0 is forward 1 is side
        tmp = state_position + quat_mul_vec(euler_to_quat(rottemp),veltemp)
        state_position = torch.tensor([Height[0], tmp[1], tmp[2]])
        state_rotation = torch.tensor([state_rotation[0] + float(RotVelY[i]), RotXZ[i][0], RotXZ[i][1]])
        fullX.append(getX(Rot[i].unsqueeze(0), skeleton, rtype, euler_to_quat(state_rotation).unsqueeze(0).unsqueeze(0), state_position.unsqueeze(0)))

    return torch.cat(fullX, dim = 0).reshape(-1, 22, 3)
