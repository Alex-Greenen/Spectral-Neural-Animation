from LowFrequency.DataBase import DataBase
import Format

db = DataBase('TrainingData', 'VAE/VAE_'+Format.name+'.pymodel')

#Latents
with open('ONNX_networks_refined/Latents.txt', 'a') as file:
    for l in range(len(db.latents)):
        file.write( ",".join(str(float(x)) for x in db.latents[l]) + "\n")

#Features
with open('ONNX_networks_refined/Features.txt', 'a') as file:
    for l in range(len(db.features)):
        file.write( ",".join(str(float(x)) for x in db.features[l]) + "\n")

#Poses
with open('ONNX_networks_refined/Poses.txt', 'a') as file:
    for l in range(len(db.poses)):
        file.write( ",".join(str(float(x)) for x in db.poses[l]) + "\n")
