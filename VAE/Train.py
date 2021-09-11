import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ProcessData.TrainingLoss import TrainingLoss

from VAE.VAE import VAE
from VAE.DataBase import DataBase
from VAE.LossFuntion import LossFunction
from VAE.Vizualise import plotState

import Format

class TrainingConfig:
    def __init__(self, batch_size, epochs, learn_rate, decay_rate):
            self.batch_size = batch_size
            self.epochs = epochs
            self.learn_rate = learn_rate
            self.decay_rate = decay_rate

def Train(training_config:TrainingConfig, error_config:TrainingLoss, outFile:str, runName:str, database:DataBase, database_validation:DataBase, visual:bool = False) -> torch.nn.Module:
    print('Starting...')
    writer = SummaryWriter(log_dir='runs/'+runName)
    labels = error_config.getNames()

    database_loader = DataLoader(database, shuffle=True, batch_size=training_config.batch_size, num_workers= (1 if torch.cuda.is_available() else 0))
    database_loader_validation = DataLoader(database_validation, shuffle=True, batch_size=training_config.batch_size, num_workers= (1 if torch.cuda.is_available() else 0))
    
    if torch.cuda.is_available():
        print('Enabling CUDA...')
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        Format.skeleton = Format.skeleton.cuda()

    print('Creating NN & Trainer...')  
    if torch.cuda.is_available():
        model_raw = VAE(database.featureDim, database.poseDim, Format.latentDim)
        model = model_raw.to(device)
    else:
        model = VAE(database.featureDim, database.poseDim, Format.latentDim)

    optimizer = torch.optim.AdamW(model.parameters(), lr = training_config.learn_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: training_config.decay_rate**ep * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    print('Starting training...')  
    for ep in range(training_config.epochs):

        # TRAINING

        losses = [0.0] * error_config.length()
        losses_sum_weighted_trainng = 0

        model.train()
        for i, data in enumerate(database_loader):

            # split the data
            pose_1, feature_1, pose_2, feature_2 = data

            if torch.cuda.is_available():
                pose_1 = pose_1.cuda()
                feature_1 = feature_1.cuda()
                pose_2 = pose_2.cuda()
                feature_2 = feature_2.cuda()
                

            # set the parameter gradients to zero
            optimizer.zero_grad()

            # forward pass and losses
            recon_1, mean_1, logvar_1 = model.forward(pose_1, feature_1)
            recon_2, mean_2, logvar_2 = model.forward(pose_2, feature_2)

            losses_here = LossFunction(recon_1, pose_1, mean_1, logvar_1, 
                                        recon_2, pose_2, mean_2, logvar_2, 
                                        database)
            
            losses_here.applyWeights(error_config)
                
            #backward pass + optimisation
            totalLoss = sum(losses_here())

            totalLoss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step() 

            # update training_loss
            losses_sum_weighted_trainng = losses_sum_weighted_trainng + float(totalLoss) * (pose_1.size(0)/len(database))

            for j in range(len(labels)): losses[j] = losses[j] + losses_here.getUnweighted().makefloat()[j] * (pose_1.size(0)/len(database))
            
        # Step the scheduler
        scheduler.step()

        if visual: plotState(recon_1, pose_1, recon_2, pose_2, 0.1, losses_here, database)

        # Add to tensorboard
        writer.add_scalar("VAE_Training/Total_Weighted", losses_sum_weighted_trainng, ep)
        writer.add_scalar("VAE_Training/Total_Unweighted", sum(losses), ep)
        for i in range(len(labels)):
            writer.add_scalar("VAE_Training/"+labels[i], losses[i], ep)
        writer.flush()

        # VALIDATION
        
        losses = [0.0] * error_config.length()
        losses_sum_weighted_validation = 0

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(database_loader_validation):

                # split the data
                pose_1, feature_1, pose_2, feature_2 = data

                if torch.cuda.is_available():
                    pose_1 = pose_1.cuda()
                    feature_1 = feature_1.cuda()
                    pose_2 = pose_2.cuda()
                    feature_2 = feature_2.cuda()

                # forward pass and losses
                recon_1, mean_1, logvar_1 = model.forward(pose_1, feature_1)
                recon_2, mean_2, logvar_2 = model.forward(pose_2, feature_2)

                losses_here = LossFunction(recon_1, pose_1, mean_1, logvar_1, 
                                            recon_2, pose_2, mean_2, logvar_2, 
                                            database_validation)

                losses_here.applyWeights(error_config)

                totalLoss = sum(losses_here())
                losses_sum_weighted_validation = losses_sum_weighted_validation + float(totalLoss) * (pose_1.size(0)/len(database_validation))

                for j in range(len(labels)): losses[j] = losses[j] + losses_here.getUnweighted().makefloat()[j] * (pose_1.size(0)/len(database_validation))
        
        if visual: plotState(recon_1, pose_1, recon_2, pose_2, 0.1, losses_here, database_validation)

        # Add to tensorboard
        writer.add_scalar("VAE_Validation/Total_Weighted", losses_sum_weighted_validation, ep)
        writer.add_scalar("VAE_Validation/Total_Unweighted", sum(losses), ep)
        for i in range(len(labels)):
            writer.add_scalar("VAE_Validation/"+labels[i], losses[i], ep)
        writer.flush()
        
        print("\nEpoch: " + str(ep) + "\t \t E_train: " + str(losses_sum_weighted_trainng)+ "\t \t E_val: " + str(losses_sum_weighted_validation))

    torch.save(model.state_dict(), outFile)

    return model


