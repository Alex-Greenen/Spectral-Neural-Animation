from ModelCompatibilityRefinement.Database import DataBase
from ModelCompatibilityRefinement.ModelCompatibilityRefinement import ModelCompatibilityRefinement
import Format

name = Format.name
DB_train = DataBase('TrainingData')
network_dir = ['VAE/VAE_{}.pymodel'.format(name), 'LowFrequency/LF_{}.pymodel'.format(name), 'HighFrequency/HF_{}.pymodel'.format(name)]
mod = ModelCompatibilityRefinement(network_dir[0], network_dir[1], network_dir[2], DB_train.featureDim, DB_train.latentDim, DB_train.poseDim)
mod.export_to_onnx('ONNX_networks')