from ProcessData.Utils import *
from ProcessData.Skeleton import Skeleton

# rotation = rotType.SixDim
# latentDim = 20 
# deltaT = 6
# poseComponents = ['R{}'.format(rotation.name), 'Height', 'Root_Position_Velocity', 'Root_HipAngleRad', 'Root_HipTurnVelRad']
# featureComponents = ['Height_last', 'Root_Position_Velocity_last', 'Root_HipAngleRad_last', 'Root_HipTurnVelRad_last', 'Future_past_pos_last', 'Future_past_ori_last', 'Future_past_vel_last', 'Future_past_speed_last', 'X_feet_last', 'V_feet_last', 'X_hands_last', 'V_hands_last']
#name = "N1_Six_20_6"

rotation = rotType.SixDim
latentDim = 25
deltaT = 6
tag = "_new" ##"_adaptive"

epRange = [5,15]

poseComponents = [  'R{}'.format(rotation.name), 
                    'Height', 
                    'Root_Position_Velocity', 
                    'Root_HipAngleRad', 
                    'Root_HipTurnVelRad', 
                    'Contacts']

featureComponents = [   'Height_last', 
                        'Root_Position_Velocity_last', 
                        'Root_HipAngleRad_last', 
                        'Root_HipTurnVelRad_last', 
                        "Future_pos_last",
                        "Future_ori_last",
                        "Past_pos_last",
                        "Past_ori_last",
                        'X_feet_last', 
                        'V_feet_last', 
                        'X_hands_last', 
                        'V_hands_last']
                        
name =  "{}_{}_{}".format(rotation.name, latentDim, deltaT) + tag
dropoutseperation = [1 + 2 + 2 + 1 + 6 + 3, 6 + 3 + 6 + 6 + 6 + 6] #[1 + 2 + 2 + 1 + 6 + 3 + 6 + 3, 6 + 3 + 6 + 3 + 6 + 6 + 6 + 6]
footIndices = [3,4,7,8]
skeleton = Skeleton(offsets=[   [ 0.00000000,  0.00000000,  0.00000000],
                                [ 1.0345e-01,  1.8578e+00,  1.0549e+01],
                                [ 4.3500e+01, -6.1000e-05,  0.0000e+00],
                                [ 4.2372e+01,  8.0000e-06, -2.0000e-06],
                                [ 1.7300e+01,  1.1000e-05,  1.2000e-05],
                                [ 1.0346e-01,  1.8578e+00, -1.0548e+01],
                                [ 4.3500e+01,  1.5000e-05,  6.0000e-06],
                                [ 4.2372e+01, -4.6000e-05,  1.7000e-05],
                                [ 1.7300e+01,  1.1000e-05,  1.5000e-05],
                                [ 6.9020e+00, -2.6037e+00, -4.0000e-06],
                                [ 1.2588e+01, -4.0000e-06, -1.0000e-06],
                                [ 1.2343e+01, -2.6000e-05,  3.0000e-06],
                                [ 2.5833e+01,  4.0000e-06, -6.0000e-06],
                                [ 1.1767e+01,  3.1000e-05,  4.0000e-06],
                                [ 1.9746e+01, -1.4803e+00,  6.0001e+00],
                                [ 1.1284e+01,  3.0000e-06, -1.5000e-05],
                                [ 3.3000e+01, -2.3000e-05,  2.7000e-05],
                                [ 2.5200e+01,  5.1000e-05,  2.1000e-05],
                                [ 1.9746e+01, -1.4804e+00, -6.0001e+00],
                                [ 1.1284e+01,  4.0000e-06, -1.5000e-05],
                                [ 3.3000e+01, -2.3000e-05,  1.1000e-05],
                                [ 2.5200e+01,  1.4800e-04,  4.2200e-04] ], 
                    parents= [-1,  0,  1,  2,  3,  0,  5,  6,  7,  0,  9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20])

left = [1,2,3,4,14,15,16,17]
right = [5,6,7,8,18,19,20,21]
central = [0,9,10,11,12,13]
training_source_dir = 'TrainingData'
validation_source_dir = 'ValidationData'
