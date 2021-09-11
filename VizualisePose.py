from ProcessData.Utils import rotType, getX_full
from VAE.DataBase import DataBase
from ModelCompatibilityRefinement.Vizualise import motion_animation
import Format

db = DataBase('TrainingData')
poses = db.poses[400:800]
motion_animation(getX_full(poses, db.poseDimDiv, Format.skeleton, Format.rotation), Format.skeleton._parents, 'here', 3)
