from torch import optim
from ProcessData.Skeleton import Skeleton
from ProcessData.Utils import getX_full
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import typing

from ModelCompatibilityRefinement_2.Database import DataBase
from ModelCompatibilityRefinement_2.ModelCompatibilityRefinement import ModelCompatibilityRefinement
from ModelCompatibilityRefinement_2.PhasingSize import PhasingSize
from ModelCompatibilityRefinement_2.Discriminator import Discriminator
from ModelCompatibilityRefinement_2.LossFunction import LossFunction
from ModelCompatibilityRefinement_2.Vizualise import motion_animation
from ModelCompatibilityRefinement_2.UpdateFeature import updateFeature, getExtremeties

from ProcessData.TrainingLoss import TrainingLoss

import Format

class TrainingConfig:
    def __init__(self, batch_size, epochs, learn_rate, learning_decay, DiscriminatorPreTrainingEpochs, warmup):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.learning_decay = learning_decay
        self.DiscriminatorPreTrainingEpochs = DiscriminatorPreTrainingEpochs
        self.warmup = warmup

def RunDiscriminator_pose(true_data:torch.Tensor, discriminator:Discriminator, database:DataBase) -> torch.Tensor:
    true_disc = discriminator(true_data)
    return true_disc

def Train(training_config:TrainingConfig, error_config:TrainingLoss, runName:str, database:DataBase, database_validation:DataBase, modelDirectories: typing.List[str], visual = True):
    print('Starting Training of ModelCompatibilityRefinement...')
    writer = SummaryWriter(log_dir='runs/'+runName)

    if torch.cuda.is_available():
        print('Enabling CUDA...')
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()

    print('Creating DB...')            
    
    database_loader = DataLoader(database, shuffle=True, batch_size=training_config.batch_size, num_workers= (1 if torch.cuda.is_available() else 0))

    print('Creating NN & Trainer...')  
    
    if torch.cuda.is_available():
        model_raw = ModelCompatibilityRefinement(modelDirectories[0], modelDirectories[1], modelDirectories[2], database.featureDim, Format.latentDim, database.poseDim)
        model = model_raw.to(device)
        discriminator = Discriminator(database.poseDim, (database.sequenceLength - training_config.warmup) * Format.deltaT )
        discriminator = discriminator.to(device)

        phasing_model_raw = PhasingSize(database.featureDim, Format.latentDim)
        phasing_model = phasing_model_raw.to(device)
    else:
        model = ModelCompatibilityRefinement(modelDirectories[0], modelDirectories[1], modelDirectories[2], database.featureDim, Format.latentDim, database.poseDim)
        discriminator = Discriminator(database.poseDim, (database.sequenceLength - training_config.warmup))
        phasing_model = PhasingSize(database.featureDim, Format.latentDim)
    
    
    model.train()
    discriminator.train()

    params = list(model.HF.parameters()) + list(model.LF.parameters()) + list(phasing_model.parameters())

    optimizer_model = torch.optim.AdamW(params, lr = training_config.learn_rate)
    scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lambda ep: training_config.learn_rate*(training_config.learning_decay**ep) * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr = training_config.learn_rate)
    scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda=lambda ep: training_config.learn_rate*(training_config.learning_decay**ep) * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    starter = 3
    
    print('Starting training...')  
    for ep in range(training_config.epochs):
        
        # TRAINING LOOP:

        losses_sum_weighted_training = 0.0
        losses_training = [0.0] * error_config.length()

        model.train()
        discriminator.train()
        for i, data in enumerate(database_loader):
            
            # split the data
            features, true_poses = data

            if torch.cuda.is_available():
                features = features.cuda()
                true_poses = true_poses.cuda()
                        
            # Initialisations
            Nbatch = features.size(0)
            frames = features.size(1)
            h1 = torch.zeros(Nbatch, model.HF.latent_red)
            h2 = torch.zeros(Nbatch, model.HF.latent_red)
            h3 = torch.zeros(Nbatch, model.HF.latent_red)
            current_feature = features[:,0]
            last_latent = torch.zeros(Nbatch, Format.latentDim)
            next_latent = torch.zeros(Nbatch, Format.latentDim)
            pose = model.Decoder(torch.zeros(Nbatch, Format.latentDim))
            lastFeet, lastHands = getExtremeties(pose, database)
            lastFeet = lastFeet.detach()
            lastHands = lastHands.detach()

            poses = []
            nextLFposes = []
            true_nextLFposes = []

            deltaT = Format.deltaT
            counterT = 0

            for f in range(frames):

                current_feature, lastFeet, lastHands = updateFeature(features[:, f], pose, database, lastFeet, lastHands)

                current_feature = features[:,f]

                if (counterT == int(deltaT)):
                    last_latent = next_latent
                    deltaT = phasing_model(current_feature, last_latent)
                    counterT = 0
                    next_latent = model.LF(current_feature, last_latent, deltaT)[0]
                    nextLFposes.append(model.Decoder(next_latent).unsqueeze(1))
                    true_nextLFposes.append(true_poses[:,f].unsqueeze(1))

                time = torch.cat((torch.ones(Nbatch, 1) * counterT / deltaT, torch.ones(Nbatch, 1) * deltaT),dim=1)
                currentLatent, h1, h2, h3 = model.HF.forward(current_feature, last_latent, next_latent, time, h1, h2, h3)
                
                pose = model.Decoder(currentLatent)
                poses.append(pose.unsqueeze(1))

                counterT+=1

            poses = torch.cat(poses, dim=1)[:, training_config.warmup:]
            true_poses = true_poses[:, training_config.warmup:]
            nextLFposes = torch.cat(nextLFposes, dim=1)[:, 2:]
            true_nextLFposes = torch.cat(true_nextLFposes, dim=1)[:, 2:]
            
            # Discriminator
            discriminator.zero_grad()
            genLoss = torch.mean(torch.square(RunDiscriminator_pose(poses.detach(), discriminator, database)))
            discLoss = torch.mean(torch.square(RunDiscriminator_pose(true_poses, discriminator, database)-1))
            sumLosses = (genLoss+discLoss)/2
            if sumLosses >= 0.15:
                sumLosses.backward()
                optimizer_disc.step()
                print("DiscriminatorLosses: Generated: {} \t \t True: {}".format(float(genLoss.item()), float(discLoss.item())))

            # Generator
            model.zero_grad()
            if ep>=starter+2:
                with torch.no_grad(): genLoss = torch.mean(torch.square(RunDiscriminator_pose(poses, discriminator, database)-1))
            else: genLoss = 0

            losses_here = LossFunction(poses, true_poses, nextLFposes, true_nextLFposes, genLoss, database, frames - training_config.warmup)
            losses_here.applyWeights(error_config)
            # backward pass + optimisation
            totalLoss = sum(losses_here())
            totalLoss.backward()
            optimizer_model.step() 
            print("Iteration {}: {}".format(i, float(totalLoss)))

             # update training_loss
            losses_sum_weighted_training = losses_sum_weighted_training + float(totalLoss) * (Nbatch/len(database))
            for j in range(len(losses_training)): losses_training[j] = losses_training[j] + losses_here.getUnweighted().makefloat()[j] * (Nbatch/len(database))

        # Add to tensorboard
        labels = error_config.getNames()
        writer.add_scalar("Refinement_Training_2/Total_Weighted", float(losses_sum_weighted_training), ep)
        writer.add_scalar("Refinement_Training_2/Total_Unweighted", float(sum(losses_training)), ep)
        for i in range(len(labels)):
            writer.add_scalar("Refinement_Training_2/"+labels[i], float(losses_training[i]), ep)

        if visual: motion_animation(getX_full(poses[0], database.poseDimDiv, Format.skeleton, Format.rotation), Format.skeleton._parents, ep, folder = "ModelCompatibilityRefinement_2/Anim/")

        # Step the scheduler
        if ep>=starter: scheduler_disc.step()
        scheduler_model.step()

    model.export_to_onnx("ONNX_networks_refined")

    return model
