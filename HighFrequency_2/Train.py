from torch import optim
from ProcessData.Skeleton import Skeleton
from ProcessData.Utils import getX_full
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import typing

from HighFrequency_2.Database import DataBase
from HighFrequency_2.HighFrequency import HighFrequency
from HighFrequency_2.Discriminator import Discriminator
from HighFrequency_2.LossFunction import LossFunction
from HighFrequency_2.Vizualise import plotState

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

def Train(training_config:TrainingConfig, error_config:TrainingLoss, outFile:str, runName:str, database:DataBase, database_validation:DataBase, visual:bool= False):
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
        model_raw = HighFrequency(Format.latentDim, database.featureDim)
        model = model_raw.to(device)
    else:
        model = HighFrequency(Format.latentDim, database.featureDim)
    
    model.train()

    optimizer_model = torch.optim.AdamW(model.parameters(), lr = training_config.learn_rate )
    scheduler_model = torch.optim.lr_scheduler.LambdaLR(optimizer_model, lr_lambda=lambda ep: training_config.learning_decay**ep * (0.1 + np.cos(np.pi * ep/10)**2)/1.1)

    warmup =  Format.deltaT * training_config.warmup
    
    print('Starting training...')  
    for ep in range(training_config.epochs):
        
        # TRAINING LOOP:

        losses_sum_weighted_training = 0.0
        losses_training = [0.0] * error_config.length()

        inter_poses = []
        inter_poses_t = []

        model.train()
        for i, data in enumerate(database_loader):
            
            # split the data
            features, latents, true_data = data

            if torch.cuda.is_available():
                features = features.cuda()
                latents = latents.cuda()
                true_data = true_data.cuda()
                        
            # Initialisations
            Nbatch = features.size(0)
            frames = features.size(1)
            h1 = torch.zeros(Nbatch, model.latent_red)
            h2 = torch.zeros(Nbatch, model.latent_red)
            h3 = torch.zeros(Nbatch, model.latentDim)

            out_latent = []

            deltaT = 0
            counterT = 0
            last_lat = torch.tensor([])
            next_lat = torch.tensor([])
            breakearly = False

            for f in range(frames):

                if counterT == deltaT:
                    deltaT = np.random.randint(Format.epRange[0], Format.epRange[1])
                    if f + deltaT >= frames : break
                    last_lat = latents[:,f]
                    next_lat = latents[:,f+deltaT]
                    counterT = 0
                    inter_poses = []
                    inter_poses_t = []

                time = torch.cat((torch.ones(Nbatch, 1) * counterT / deltaT, torch.ones(Nbatch, 1) * deltaT),dim=1)
                currentLatent, h1, h2, h3 = model.forward(features[:,f], last_lat, next_lat, time, h1, h2, h3)
                
                out_latent.append(currentLatent.unsqueeze(1))
                pose = database.AE_network.decoder(currentLatent)

                inter_poses.append(pose[0].detach())
                inter_poses_t.append(true_data[0, f])
                
                counterT += 1

            out_latent = torch.cat(out_latent, dim=1)
            
            losses_here = LossFunction(out_latent[:, warmup:], latents[:, warmup:len(out_latent[0])], true_data[:, warmup:len(out_latent[0])], 0, database, len(out_latent[0]) - warmup)
            losses_here.applyWeights(error_config)
            # backward pass + optimisation
            totalLoss = sum(losses_here())
            totalLoss.backward()
            optimizer_model.step() 

            # update training_loss
            losses_sum_weighted_training = losses_sum_weighted_training + float(totalLoss) * (Nbatch/len(database))
            for j in range(len(losses_training)): losses_training[j] = losses_training[j] + losses_here.getUnweighted().makefloat()[j] * (Nbatch/len(database))
        
        # Step the scheduler
        scheduler_model.step()

        with torch.no_grad():
            # Add to tensorboard
            labels = error_config.getNames()
            writer.add_scalar("HF2_Training/Total_Weighted", losses_sum_weighted_training, ep)
            writer.add_scalar("HF2_Training/Total_Unweighted", sum(losses_training), ep)
            for i in range(len(labels)):
                writer.add_scalar("HF2_Training/"+labels[i], losses_training[i], ep)
            writer.flush()

        
        # Vizualise
        if visual:
            inter_poses = torch.vstack(inter_poses)
            inter_poses_t = torch.vstack(inter_poses_t)
            plotState(database.AE_network.decoder(last_lat)[0], database.AE_network.decoder(next_lat)[0], inter_poses_t, inter_poses, losses_here.makefloat(), database)


        
        # Validation LOOP:

        losses_sum_weighted_validation = 0.0
        losses_validation = [0.0] * error_config.length()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(database_validation_loader):
                
                # split the data
                features, latents, true_data = data

                if torch.cuda.is_available():
                    features = features.cuda()
                    latents = latents.cuda()
                    true_data = true_data.cuda()
                            
                # Initialisations
                Nbatch = features.size(0)
                frames = features.size(1)
                h1 = torch.zeros(Nbatch, model.latent_red)
                h2 = torch.zeros(Nbatch, model.latent_red)
                h3 = latents[:,0]

                poses = []
                out_latent = []

                deltaT = 0
                counterT = 0
                last_lat= torch.tensor([])
                next_lat = torch.tensor([])
                breakearly = False

                for f in range(frames):

                    if counterT == deltaT:
                        deltaT = np.random.randint(Format.epRange[0], Format.epRange[1])
                        if f + deltaT >= frames : breakearly = True
                        else:
                            last_lat = latents[:,f]
                            next_lat = latents[:,f+deltaT]
                            counterT = 0

                    if not breakearly:
                        time = torch.cat((torch.ones(Nbatch, 1) * counterT / deltaT, torch.ones(Nbatch, 1) * (deltaT - Format.epRange[0] + 1)),dim=1)
                        currentLatent, h1, h2, h3 = model.forward(features[:,f], last_lat, next_lat, time, h1, h2, h3)
                        out_latent.append(currentLatent.unsqueeze(1))
                        pose = database.AE_network.decoder(currentLatent)
                        poses.append(pose.unsqueeze(1))

                        counterT += 1


                out_latent = torch.cat(out_latent, dim=1)[:, warmup:]
                poses = torch.cat(poses, dim=1)[:, warmup:]
                
                losses_here = LossFunction(out_latent[:, warmup:], latents[:, warmup:len(out_latent[0])], true_data[:, warmup:len(out_latent[0])], 0, database_validation, len(out_latent[0]) - warmup)
                losses_here.applyWeights(error_config)
                # backward pass + optimisation
                totalLoss = sum(losses_here())

                # update validation_loss
                losses_sum_weighted_validation = losses_sum_weighted_validation + float(totalLoss) * (Nbatch/len(database_validation))
                for j in range(len(losses_validation)): losses_validation[j] = losses_validation[j] + losses_here.getUnweighted().makefloat()[j] * (Nbatch/len(database_validation))

            # Add to tensorboard
            labels = error_config.getNames()
            writer.add_scalar("HF2_Validation/Total_Weighted", losses_sum_weighted_validation, ep)
            writer.add_scalar("HF2_Validation/Total_Unweighted", sum(losses_validation), ep)
            for i in range(len(labels)):
                writer.add_scalar("HF2_Validation/"+labels[i], losses_validation[i], ep)
            writer.flush()
        
        print("\nEpoch: " + str(ep) + "\t \t ErrorTr: " + str(losses_sum_weighted_training)+ "\t \t ErrorEv: " + str(losses_sum_weighted_validation))

    
    torch.save(model.state_dict(), outFile)

    #model.export_to_onnx("ONNX_networks")

    return model
