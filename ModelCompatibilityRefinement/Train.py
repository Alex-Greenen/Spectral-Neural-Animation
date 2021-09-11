from torch import optim
from ProcessData.Skeleton import Skeleton
from ProcessData.Utils import getX_full
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import typing

from ModelCompatibilityRefinement.Database import DataBase
from ModelCompatibilityRefinement.ModelCompatibilityRefinement import ModelCompatibilityRefinement
from ModelCompatibilityRefinement.Discriminator import Discriminator
from ModelCompatibilityRefinement.LossFunction import LossFunction
from ModelCompatibilityRefinement.Vizualise import motion_animation
from ModelCompatibilityRefinement.UpdateFeature import updateFeature, getExtremeties

from ProcessData.TrainingLoss import TrainingLoss
from ProcessData.Utils import *

import Format

class TrainingConfig:
    def __init__(self, batch_size, epochs, learn_rate, learning_decay, DiscriminatorPreTrainingEpochs, warmup):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learn_rate = learn_rate
        self.learning_decay = learning_decay
        self.DiscriminatorPreTrainingEpochs = DiscriminatorPreTrainingEpochs
        self.warmup = warmup

def RunDiscriminator_pose(data:torch.Tensor, discriminator:Discriminator, database:DataBase) -> torch.Tensor:

    skeleton = Format.skeleton
    rotation = Format.rotation

    R_r, Root_Height_r, Root_Position_Velocity_r, Root_Rotation_xz_r, Root_RotationVel_y_r, contacts_r = torch.split(data, database.poseDimDiv, dim=-1)

    R_r = reformat(R_r)
    R_r = correct(R_r, rotation)
    X_r = getX(R_r, skeleton, rotation)
    X_r = torch.flatten(X_r,-2,-1)
    X_r = torch.cat((X_r, Root_Height_r, Root_Position_Velocity_r, Root_Rotation_xz_r, Root_RotationVel_y_r, contacts_r), dim = -1)

    true_disc = discriminator(X_r)
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
    database_validation_loader = DataLoader(database_validation, shuffle=True, batch_size=training_config.batch_size, num_workers= (1 if torch.cuda.is_available() else 0))

    print('Creating NN & Trainer...')  
    
    if torch.cuda.is_available():
        model_raw = ModelCompatibilityRefinement(modelDirectories[0], modelDirectories[1], modelDirectories[2], database.featureDim, Format.latentDim, database.poseDim)
        model = model_raw.to(device)
        discriminator = Discriminator(22 * 3 + 8, (database.sequenceLengthLong - training_config.warmup) * Format.deltaT )
        discriminator = discriminator.to(device)
    else:
        model = ModelCompatibilityRefinement(modelDirectories[0], modelDirectories[1], modelDirectories[2], database.featureDim, Format.latentDim, database.poseDim)
        discriminator = Discriminator(22 * 3 + 8, (database.sequenceLengthLong - training_config.warmup) * Format.deltaT )
    
    model.train()
    discriminator.train()

    params = list(model.HF.parameters()) + list(model.LF.parameters())

    optimizer_model = torch.optim.AdamW(params, lr = training_config.learn_rate)
    scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lambda ep: training_config.learn_rate*(training_config.learning_decay**ep) * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    optimizer_disc = torch.optim.AdamW(discriminator.parameters(), lr = training_config.learn_rate)
    scheduler_disc = torch.optim.lr_scheduler.LambdaLR(optimizer_disc, lr_lambda=lambda ep: training_config.learn_rate*(training_config.learning_decay**ep) * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    starter = 3
    warmup =  Format.deltaT * training_config.warmup
    
    print('Starting training...')  
    for ep in range(training_config.epochs):
        
        # TRAINING LOOP:

        losses_sum_weighted_training = 0.0
        losses_training = [0.0] * error_config.length()

        model.train()
        discriminator.train()
        for i, data in enumerate(database_loader):
            
            # split the data
            features, true_poses, true_latents = data

            if torch.cuda.is_available():
                features = features.cuda()
                true_poses = true_poses.cuda()
                true_latents = true_latents.cuda()
            
            true_poses = true_poses[:, warmup:]
            true_latents = true_latents[:, warmup:]
            
            # Initialisations
            Nbatch = features.size(0)
            frames = features.size(1)
            h1 = torch.zeros(Nbatch, model.HF.latent_red)
            h2 = torch.zeros(Nbatch, model.HF.latent_red)
            h3 = torch.zeros(Nbatch, model.HF.latent_red)
            last_feature = features[:,0]
            current_feature = features[:,0]
            last_latent = torch.zeros(Nbatch, Format.latentDim)
            next_latent = torch.zeros(Nbatch, Format.latentDim)
            pose = model.Decoder(torch.zeros(Nbatch, Format.latentDim))
            lastFeet, lastHands = getExtremeties(pose, database)
            lastFeet = lastFeet.detach()
            lastHands = lastHands.detach()

            poses = []
            latents = []
            nextLFposes = []

            for f in range(frames):

                current_feature, lastFeet, lastHands = updateFeature(features[:, f], pose, database, lastFeet, lastHands)

                current_feature = features[:,f]

                if (f % Format.deltaT ==0):
                    last_latent = next_latent
                    next_latent = model.LF(last_feature, current_feature, next_latent)[0]
                    last_feature = current_feature
                    nextLFposes.append(model.Decoder(next_latent).unsqueeze(1))

                currentLatent, h1, h2, h3 = model.HF.forward(current_feature, last_latent, next_latent, torch.ones(Nbatch, 1) * (f % Format.deltaT) / Format.deltaT, h1, h2, h3)
                
                latents.append(currentLatent.unsqueeze(1))
                pose = model.Decoder(currentLatent)
                poses.append(pose.unsqueeze(1))


            poses = torch.cat(poses, dim=1)[:, warmup:]
            nextLFposes = torch.cat(nextLFposes, dim=1)[:, training_config.warmup:]
            latents = torch.cat(latents, dim=1)[:, warmup:]
            
            # Discriminator
            discriminator.zero_grad()
            genLoss = torch.mean(torch.square(RunDiscriminator_pose(poses.detach(), discriminator, database)))
            discLoss = torch.mean(torch.square(RunDiscriminator_pose(true_poses, discriminator, database)-1))
            sumLosses = (genLoss+discLoss)/2
            if sumLosses >= 0.15:
                sumLosses.backward()
                optimizer_disc.step()
                #print("DiscriminatorLosses: Generated: {} \t \t True: {}".format(float(genLoss.item()), float(discLoss.item())))

            # Generator
            model.zero_grad()
            if ep>=starter+2:
                with torch.no_grad(): genLoss = torch.mean(torch.square(RunDiscriminator_pose(poses, discriminator, database)-1))
            else: genLoss = 0

            losses_here = LossFunction(latents, true_latents, poses, true_poses, nextLFposes, genLoss, database, frames - warmup)
            losses_here.applyWeights(error_config)
            # backward pass + optimisation
            totalLoss = sum(losses_here())
            totalLoss.backward()
            optimizer_model.step() 
            #print("Iteration {}: {}".format(i, float(totalLoss)))

             # update training_loss
            losses_sum_weighted_training = losses_sum_weighted_training + float(totalLoss) * (Nbatch/len(database))
            for j in range(len(losses_training)): losses_training[j] = losses_training[j] + losses_here.getUnweighted().makefloat()[j] * (Nbatch/len(database))

        # Add to tensorboard
        labels = error_config.getNames()
        writer.add_scalar("Refinement_Training/Total_Weighted", float(losses_sum_weighted_training), ep)
        writer.add_scalar("Refinement_Training/Total_Unweighted", float(sum(losses_training)), ep)
        for i in range(len(labels)):
            writer.add_scalar("Refinement_Training/"+labels[i], float(losses_training[i]), ep)

        if visual: motion_animation(getX_full(poses[0], database.poseDimDiv, Format.skeleton, Format.rotation), Format.skeleton._parents, ep, "ModelCompatibilityRefinement/Anim/")

        # Step the scheduler
        if ep>=starter: scheduler_disc.step()
        scheduler_model.step()

        # VALIDATION

        with torch.no_grad():

            losses_sum_weighted_validation = 0.0
            losses_validation = [0.0] * error_config.length()

            model.eval()
            discriminator.eval()
            for i, data in enumerate(database_validation_loader):
                
                # split the data
                features, true_poses, true_latents = data

                if torch.cuda.is_available():
                    features = features.cuda()
                    true_poses = true_poses.cuda()
                    true_latents = true_latents.cuda()
                
                true_poses = true_poses[:, warmup:]
                true_latents = true_latents[:, warmup:]
                
                # Initialisations
                Nbatch = features.size(0)
                frames = features.size(1)
                h1 = torch.zeros(Nbatch, model.HF.latent_red)
                h2 = torch.zeros(Nbatch, model.HF.latent_red)
                h3 = torch.zeros(Nbatch, model.HF.latent_red)
                last_feature = features[:,0]
                current_feature = features[:,0]
                last_latent = torch.zeros(Nbatch, Format.latentDim)
                next_latent = torch.zeros(Nbatch, Format.latentDim)
                pose = model.Decoder(torch.zeros(Nbatch, Format.latentDim))
                lastFeet, lastHands = getExtremeties(pose, database_validation)
                lastFeet = lastFeet.detach()
                lastHands = lastHands.detach()

                latents = []
                poses = []
                nextLFposes = []

                for f in range(frames):

                    current_feature, lastFeet, lastHands = updateFeature(features[:, f], pose, database_validation, lastFeet, lastHands)

                    current_feature = features[:,f]

                    if (f % Format.deltaT ==0):
                        last_latent = next_latent
                        next_latent = model.LF(last_feature, current_feature, next_latent)[0]
                        last_feature = current_feature
                        nextLFposes.append(model.Decoder(next_latent).unsqueeze(1))

                    currentLatent, h1, h2, h3 = model.HF.forward(current_feature, last_latent, next_latent, torch.ones(Nbatch, 1) * (f % Format.deltaT) / Format.deltaT, h1, h2, h3)
                    
                    pose = model.Decoder(currentLatent)
                    latents.append(currentLatent.unsqueeze(1))
                    poses.append(pose.unsqueeze(1))

                poses = torch.cat(poses, dim=1)[:, warmup:]
                latents = torch.cat(latents, dim=1)[:, warmup:]
                nextLFposes = torch.cat(nextLFposes, dim=1)[:, training_config.warmup:]
                
                # Discriminator
                discriminator.zero_grad()
                genLoss = torch.mean(torch.square(RunDiscriminator_pose(poses.detach(), discriminator, database_validation)))
                discLoss = torch.mean(torch.square(RunDiscriminator_pose(true_poses, discriminator, database_validation)-1))
                sumLosses = (genLoss+discLoss)/2
                #if sumLosses >= 0.15: print("DiscriminatorLosses: Generated: {} \t \t True: {}".format(float(genLoss.item()), float(discLoss.item())))

                # Generator
                if ep>=starter+2: genLoss = torch.mean(torch.square(RunDiscriminator_pose(poses, discriminator, database_validation)-1))
                else: genLoss = 0

                losses_here = LossFunction(latents, true_latents, poses, true_poses, nextLFposes, genLoss, database_validation, frames - warmup)
                losses_here.applyWeights(error_config)
                totalLoss = sum(losses_here())
                #print("Iteration {}: {}".format(i, float(totalLoss)))

                # update training_loss
                losses_sum_weighted_validation = losses_sum_weighted_validation + float(totalLoss) * (Nbatch/len(database_validation))
                for j in range(len(losses_validation)): losses_validation[j] = losses_validation[j] + losses_here.getUnweighted().makefloat()[j] * (Nbatch/len(database_validation))

            # Add to tensorboard
            labels = error_config.getNames()
            writer.add_scalar("Refinement_Validation/Total_Weighted", float(losses_sum_weighted_validation), ep)
            writer.add_scalar("Refinement_Validation/Total_Unweighted", float(sum(losses_validation)), ep)
            for i in range(len(labels)):
                writer.add_scalar("Refinement_Validation/"+labels[i], float(losses_validation[i]), ep)
            writer.flush()
        
        print("\nEpoch: " + str(ep) + "\t \t ErrorTr: " + str(losses_sum_weighted_training) + "\t \t ErrorVal: " + str(losses_sum_weighted_validation))


    model.export_to_onnx("ONNX_networks_refined")

    return model
