import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ProcessData.TrainingLoss import TrainingLoss
from ProcessData.Utils import getX_full

from typing import Tuple

from HighFrequency.HighFrequency import HighFrequency
from HighFrequency.Discriminator import Discriminator

from HighFrequency.DataBase import DataBase
from HighFrequency.LossFunction import LossFunction
from HighFrequency.Vizualise import plotState

import Format

class TrainingConfig:
    def __init__(self, batch_size , epochs, learn_rate, learning_decay, warm_up):
            self.batch_size = batch_size
            self.epochs = epochs
            self.learn_rate = learn_rate
            self.learning_decay = learning_decay
            self.warm_up = warm_up

def RunDiscriminator_latent(latentRecon:torch.Tensor, discriminator:Discriminator, database:DataBase) -> torch.Tensor:
    #generated_poses = []
    decoded = database.AE_network.decoder(latentRecon)
    #for i in range(len(latentRecon)):
        #generated_poses.append(getX_full(decoded[i], database.poseDimDiv, database.skeleton, Format.rotation)/100)
    #generated_poses = torch.flatten(torch.stack(generated_poses, dim=0), -2, -1)
    #generated_disc = discriminator(generated_poses)
    generated_disc = discriminator(decoded)
    return generated_disc

def RunDiscriminator_pose(true_data:torch.Tensor, discriminator:Discriminator, database:DataBase) -> torch.Tensor:
    #true_poses = []
    # for i in range(len(true_data)):
    #     true_poses.append(getX_full(true_data[i], database.poseDimDiv, database.skeleton, Format.rotation)/100)
    # true_poses = torch.flatten(torch.stack(true_poses, dim=0), -2, -1)
    # true_disc = discriminator(true_poses)
    true_disc = discriminator(true_data)
    return true_disc

def Train(training_config:TrainingConfig, error_config:TrainingLoss, outFile:str, runName:str, database:DataBase, database_validation:DataBase, visual:bool= False):
    print('Starting...')
    writer = SummaryWriter(log_dir='runs/'+runName)

    if torch.cuda.is_available():
        print('Enabling CUDA...')
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()

    print('Creating DB...')            
    
    database_loader = DataLoader(database, shuffle=True, batch_size=training_config.batch_size, num_workers= (1 if torch.cuda.is_available() else 0))
    database_loader_validation = DataLoader(database_validation, shuffle=True, batch_size=training_config.batch_size, num_workers= (1 if torch.cuda.is_available() else 0))

    print('Creating NN & Trainer...')  
    
    if torch.cuda.is_available():
        model_raw = HighFrequency(Format.latentDim, database.featureDim)
        model = model_raw.to(device)
        discriminator = Discriminator(database.poseDim)
        discriminator = discriminator.to(device)
    else:
        model = HighFrequency(Format.latentDim, database.featureDim)
        discriminator = Discriminator(database.poseDim)


    optimizer_model = torch.optim.AdamW(model.parameters(), lr = training_config.learn_rate )
    scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lambda ep: training_config.learning_decay**ep * (0.1 + np.cos(np.pi * ep/10)**2)/1.1)

    optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr = training_config.learn_rate )
    scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda=lambda ep: training_config.learning_decay**ep)

    # optimizer_model = torch.optim.AdamW(model.parameters(), lr = training_config.learn_rate)
    # scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lambda ep: training_config.learn_rate*(training_config.learning_decay**ep) * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    # optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr = training_config.learn_rate)
    # scheduler_disc= torch.optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda=lambda ep: training_config.learn_rate*(training_config.learning_decay**ep) * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    warm_up_frames = Format.deltaT * training_config.warm_up

    starter = 3

    print('Starting training...')  
    for ep in range(training_config.epochs):

        # TRAIN

        losses_sum_weighted_training = 0.0
        losses_training = [0.0] * error_config.length()
        
        model.train()
        discriminator.train()
        for i, data in enumerate(database_loader):
            
            # split the data
            latentLast, latentNext, times, features, latent_goal, true_data, start_end_true = data

            if torch.cuda.is_available():
                latentLast = latentLast.cuda()
                latentNext = latentNext.cuda()
                times = times.cuda()
                features = features.cuda()
                latent_goal = latent_goal.cuda()
                true_data = true_data.cuda()

            # set the parameter gradients to zero

            # forward pass and losses
            out_latent = model.forward_full(latentLast, latentNext, times, features)

            batches = true_data.size(0)
            frames = true_data.size(1)

            # # Discriminator
            # if ep>=starter:
            #     discriminator.zero_grad()
            #     genLoss = torch.mean(torch.square(RunDiscriminator_latent(out_latent[:, warm_up_frames:].detach(), discriminator, database)))
            #     discLoss = torch.mean(torch.square(RunDiscriminator_pose(true_data[:, warm_up_frames:], discriminator, database)-1))
            #     sumLosses = (genLoss+discLoss)/2
            #     if sumLosses >= 0.15:
            #         sumLosses.backward()
            #         # torch.nn.utils.clip_grad_value_(discriminator.parameters(), 1)
            #         optimizer_disc.step()
            #         print("DiscriminatorLosses: Generated: {} \t \t True: {}".format(float(genLoss.item()), float(discLoss.item())))

            
            # # Generator
            # model.zero_grad()
            # if ep>=starter+2:
            #     with torch.no_grad(): genLoss = torch.mean(torch.square(RunDiscriminator_latent(out_latent[:, warm_up_frames:], discriminator, database)-1))
            # else: 
            genLoss = 0
            losses_here = LossFunction(out_latent[:, warm_up_frames:], latent_goal[:, warm_up_frames:], true_data[:, warm_up_frames:], genLoss, database)
            losses_here.applyWeights(error_config)
            # backward pass + optimisation
            totalLoss = sum(losses_here())
            totalLoss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 2)
            optimizer_model.step() 

            # update training_loss
            losses_sum_weighted_training = losses_sum_weighted_training + float(totalLoss) * (batches/len(database))
            for j in range(len(losses_training)): losses_training[j] = losses_training[j] + losses_here.getUnweighted().makefloat()[j] * (batches/len(database))
            
        # Step the scheduler
        scheduler_model.step()
        if ep>=starter: scheduler_disc.step()

        with torch.no_grad():
            # Add to tensorboard
            labels = error_config.getNames()
            writer.add_scalar("HF_Training/Total_Weighted", losses_sum_weighted_training, ep)
            writer.add_scalar("HF_Training/Total_Unweighted", sum(losses_training), ep)
            for i in range(len(labels)):
                writer.add_scalar("HF_Training/"+labels[i], losses_training[i], ep)
            writer.flush()

            # Vizualise
            if visual:
                episode_start, episode_end = torch.split(start_end_true[0][(database.sequenceLengthLong-2) * Format.deltaT + int(Format.deltaT/2)],(database.poseDim,database.poseDim), dim=-1)
                frames_true = torch.split(latent_goal[0][(database.sequenceLengthLong-2) * Format.deltaT +1: (database.sequenceLengthLong-1) * Format.deltaT], 1, dim=-2)
                frames_predicted = torch.split(out_latent[0][(database.sequenceLengthLong-2) * Format.deltaT: (database.sequenceLengthLong-1) * Format.deltaT+1], 1, dim=-2)
                plotState(episode_start, episode_end, frames_true, frames_predicted, losses_training, database_validation)

        # VALIDATE
        
        with torch.no_grad():

            losses_sum_weighted_validation = 0.0
            losses_validation = [0.0] * error_config.length()
            model.eval()
            discriminator.eval()
            for i, data in enumerate(database_loader_validation):
                
                # split the data
                latentLast, latentNext, times, features, latent_goal, true_data, start_end_true = data

                if torch.cuda.is_available():
                    latentLast = latentLast.cuda()
                    latentNext = latentNext.cuda()
                    times = times.cuda()
                    features = features.cuda()
                    latent_goal = latent_goal.cuda()
                    true_data = true_data.cuda()

                # set the parameter gradients to zero

                # forward pass and losses
                out_latent = model.forward_full(latentLast, latentNext, times, features)

                batches = true_data.size(0)
                frames = true_data.size(1)

                # if ep>=starter:
                #     genLoss = torch.mean(torch.square(RunDiscriminator_latent(out_latent[:, warm_up_frames:].detach(), discriminator, database_validation)))
                #     discLoss = torch.mean(torch.square(RunDiscriminator_pose(true_data[:, warm_up_frames:], discriminator, database_validation)-1))
                #     sumLosses = (genLoss+discLoss)/2
                #     if sumLosses >= 0.15: print("DiscriminatorLosses: Generated: {} \t \t True: {}".format(float(genLoss.item()), float(discLoss.item())))
                
                # # Generator
                # if ep>=starter+2: genLoss = torch.mean(torch.square(RunDiscriminator_latent(out_latent[:, warm_up_frames:], discriminator, database_validation)-1))
                genLoss = 0
                losses_here = LossFunction(out_latent[:, warm_up_frames:], latent_goal[:, warm_up_frames:], true_data[:, warm_up_frames:], genLoss, database_validation)
                losses_here.applyWeights(error_config)
                totalLoss = sum(losses_here())

                # update validation_loss
                losses_sum_weighted_validation = losses_sum_weighted_validation + float(totalLoss) * (batches/len(database_validation))
                for j in range(len(losses_validation)): losses_validation[j] = losses_validation[j] + losses_here.getUnweighted().makefloat()[j] * (batches/len(database_validation))

            # Add to tensorboard
            labels = error_config.getNames()
            writer.add_scalar("HF_Validation/Total_Weighted", losses_sum_weighted_validation, ep)
            writer.add_scalar("HF_Validation/Total_Unweighted", sum(losses_validation), ep)
            for i in range(len(labels)):
                writer.add_scalar("HF_Validation/"+labels[i], losses_validation[i], ep)
            writer.flush()

            # Vizualise
            if visual:
                episode_start, episode_end = torch.split(start_end_true[0][(database_validation.sequenceLengthLong-2) * Format.deltaT + int(Format.deltaT/2)],(database_validation.poseDim,database_validation.poseDim), dim=-1)
                frames_true = torch.split(latent_goal[0][(database_validation.sequenceLengthLong-2) * Format.deltaT +1: (database_validation.sequenceLengthLong-1) * Format.deltaT], 1, dim=-2)
                frames_predicted = torch.split(out_latent[0][(database_validation.sequenceLengthLong-2) * Format.deltaT: (database_validation.sequenceLengthLong-1) * Format.deltaT+1], 1, dim=-2)
                plotState(episode_start, episode_end, frames_true, frames_predicted, losses_validation, database_validation)

        # Print
        print("\nEpoch: " + str(ep) + "\t \t ErrorTr: " + str(losses_sum_weighted_training) + "\t \t ErrorVal: " + str(losses_sum_weighted_validation))

    torch.save(model.state_dict(), outFile)

    return model
