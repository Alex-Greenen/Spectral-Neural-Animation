import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from LowFrequency_2.LowFrequency import LowFrequency
from LowFrequency_2.DataBase import DataBase
from LowFrequency_2.LossFunction import LossFunction
from LowFrequency_2.Vizualise import plotState
from ProcessData.TrainingLoss import TrainingLoss
import Format


class TrainingConfig:
    def __init__(self, batch_size , epochs, learn_rate, decay_rate):
            self.batch_size = batch_size
            self.epochs = epochs
            self.learn_rate = learn_rate
            self.decay_rate = decay_rate

def Train(training_config:TrainingConfig, error_config:TrainingLoss, outFile:str, runName:str, database:DataBase, database_validation:DataBase, visual:bool = False):
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
        model_raw = LowFrequency(database.featureDim, Format.latentDim)
        model = model_raw.to(device)
    else:
        model = LowFrequency(database.featureDim, Format.latentDim)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr = training_config.learn_rate )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda ep: training_config.decay_rate**ep * (0.1 + np.cos(np.pi * ep/10)**2)/1.1 )

    print('Starting training...')  
    for ep in range(training_config.epochs):

        
        ### TRAINING LOOP
        losses_sum_weighted_training = 0.0
        losses_training = [0.0] * error_config.length()
        model.train()
        for i, data in enumerate(database_loader):
            
            # split the data
            descriptor, latent_last, latent_goal, pose_goal, deltaT = data

            if torch.cuda.is_available():
                descriptor = descriptor.cuda()
                latent_last = latent_last.cuda()
                latent_goal = latent_goal.cuda()
                pose_goal = pose_goal.cuda()
                deltaT = deltaT.cuda()

            # set the parameter gradients to zero
            model.zero_grad()

            # forward pass and losses
            out_latent = model.forward(descriptor, latent_last, deltaT.float())[0]

            losses_here = LossFunction(out_latent, latent_goal, pose_goal, database)
            
            losses_here.applyWeights(error_config)
                
            #backward pass + optimisation
            totalLoss = sum(losses_here())

            totalLoss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            optimizer.step() 

            # update training_loss
            losses_sum_weighted_training = losses_sum_weighted_training + float(totalLoss) * (out_latent.size(0)/len(database))

            for j in range(len(losses_training)): losses_training[j] = losses_training[j] + losses_here.getUnweighted().makefloat()[j] * (out_latent.size(0)/len(database))
                    
        # Step the scheduler
        scheduler.step()

        # Add to tensorboard
        with torch.no_grad():
            
            labels = error_config.getNames()
            writer.add_scalar("LF_Training_2/Total_Weighted", losses_sum_weighted_training, ep)
            writer.add_scalar("LF_Training_2/Total_Unweighted", sum(losses_training), ep)
            for i in range(len(labels)):
                writer.add_scalar("LF_Training_2/"+labels[i], losses_training[i], ep)
            writer.flush()

            # Vizualise
            if visual: plotState(latent_last, out_latent, pose_goal, losses_here, database)
        
        ### VALIDATION LOOP
        with torch.no_grad():
            losses_sum_weighted_validation = 0.0
            losses_validation = [0.0] * error_config.length()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(database_loader_validation):

                    # split the data
                    descriptor, latent_last, latent_goal, pose_goal, deltaT = data

                    if torch.cuda.is_available():
                        descriptor = descriptor.cuda()
                        latent_last = latent_last.cuda()
                        latent_goal = latent_goal.cuda()
                        pose_goal = latent_goal.cuda()
                        deltaT = deltaT.cuda()

                    # forward pass and losses
                    out_latent = model.forward(descriptor, latent_last, deltaT.float())[0]

                    losses_here = LossFunction(out_latent, latent_goal, pose_goal, database)
                    
                    losses_here.applyWeights(error_config)
                    
                    # update training_loss
                    losses_sum_weighted_validation = losses_sum_weighted_validation + float(totalLoss) * (out_latent.size(0)/len(database_validation))

                    for j in range(len(losses_validation)): losses_validation[j] = losses_validation[j] + losses_here.getUnweighted().makefloat()[j] * (out_latent.size(0)/len(database_validation))

            # Add to tensorboard
            labels = error_config.getNames()
            writer.add_scalar("LF_Validation_2/Total_Weighted", losses_sum_weighted_validation, ep)
            writer.add_scalar("LF_Validation_2/Total_Unweighted", sum(losses_validation), ep)
            for i in range(len(labels)):
                writer.add_scalar("LF_Validation_2/"+labels[i], losses_validation[i], ep)
            writer.flush()

        # Pring
        print("\nEpoch: " + str(ep) + "\t \t ErrorTrain: " + str(losses_sum_weighted_training) + "\t \t ErrorVal: " + str(losses_sum_weighted_validation))

    torch.save(model.state_dict(), outFile)

    return model
