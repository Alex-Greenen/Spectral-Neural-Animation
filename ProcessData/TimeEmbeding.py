import torch


def create_time_embedding_adder(feature_dim:int, sequence_length:int, deltaT:int, basis:int = 0.5) -> torch.tensor: # basis is 10000 in paper
    
    proportions = torch.remainder(torch.arange((sequence_length-1)*deltaT),deltaT)/deltaT
    time_embedding = torch.zeros((len(proportions), feature_dim))

    for i in range(len(proportions)):
        for j in range(feature_dim):
            if j%2 ==0:
                time_embedding[i,j] = torch.sin(proportions[i]/basis**(2*j/feature_dim))
            else:
                time_embedding[i,j] = torch.cos(proportions[i]/basis**(2*j/feature_dim))
    
    return time_embedding

def time_embed_single_frame(frame: torch.tensor, timeStep:int, deltaT:int, basis:int = 0.5)->torch.tensor: 
    """ embeds a single frame given a timestep

    Args:
        frame (torch.tensor): shape (*, N_features)
        proportion (float): [description]

    Returns:
        torch.tensor: [description]
    """

    proportion = (timeStep % deltaT) / deltaT
    feature_size = int(frame.size(-1))
    adder = torch.zeros(feature_size)

    if timeStep % deltaT == 0:
        for j in range(feature_size):
            adder[j] = torch.sin(proportion/basis**(2*j/feature_size))
    else: 
        for j in range(feature_size):
            adder[j] = torch.cos(proportion/basis**(2*j/feature_size))
    
    return frame + adder

    
    
