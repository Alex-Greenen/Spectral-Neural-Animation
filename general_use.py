from ProcessData.TrainingLoss import TrainingLoss
import time
from ModelCompatibilityRefinement.Database import DataBase
from ModelCompatibilityRefinement.Train import Train, TrainingConfig
from ModelCompatibilityRefinement.LossType import ModelCompatibilityRefinementLossType
import Format 

# USE: conda activate /Users/alexgreenen/Documents/Project/training
#      /Users/alexgreenen/Documents/Project/training/bin/python -m cProfile -o program.prof general_use.py
#      snakeviz program.prof

DB_train = DataBase('TrainingData', sequenceLengthLong=15, sampling_down_factor=0.05)
DB_validation = DataBase('ValidationData', sequenceLengthLong=15, sampling_down_factor=0.05)

name = Format.name
network_dir = ['VAE/VAE_{}.pymodel'.format(name), 'LowFrequency/LF_{}.pymodel'.format(name), 'HighFrequency/HF_{}.pymodel'.format(name)]

training_config = TrainingConfig(10, 50, 0.002, 0.9, 2, 5)
error_config = TrainingLoss(ModelCompatibilityRefinementLossType, [
                20,             # Discriminator 
                10,             # Latent 
                2,              # LowFrequency
                0.5,            # Angle
                0.1,            # FK
                20.,            # PosVel
                1.,             # Rotflat
                10.,            # RotVelVert
                5.,             # Height
                0.1,            # Feet
                1,              # Contacts
                0.2,            # Angle Smooth
                0.01,           # FK Smooth
                1.,             # PosVel Smooth
                2.,             # Rotflat Smooth
                0.,             # RotVelVert Smooth
                1.,             # Height Smooth
                0.05,           # FeetSmooth
                ])

Train(training_config, error_config, "Refinement_"+name+time.strftime("_%m-%d-%H:%M"), DB_train, DB_validation, network_dir, True)


# Train(training_config, error_config, "HighFrequency_2/HF_"+name+".pymodel", "HF_"+name+time.strftime("_%m-%d-%H:%M"), DB_train, DB_validate, visual = True)

# ['Future_past_pos_last', 'Future_past_ori_last', 'Future_past_vel_last', 'Future_past_speed_last']
#
# inputs, features = DB_train[150]
# feature = features[0]
#
# #plt.scatter(0,0)
# for i in range(6):
#     plt.scatter(feature[2*i]/100, feature[2*i+1]/100, c='b')
#     plt.plot([feature[2*i]/100, feature[2*i]/100 + np.cos(feature[12+i])], 
#              [feature[2*i+1]/100, feature[2*i+1]/100 + np.sin(feature[12+i])], c='r')
#     plt.plot([feature[2*i]/100, feature[2*i]/100 + feature[30 + i] * feature[18+2*i]/100],
#              [feature[2*i+1]/100, feature[2*i+1]/100 + feature[30 + i] * feature[18+2*i+1]/100], c='g')
#
# plt.show()
