import numpy as np

class LR():
    n_min = 0.0000001
    n_max = 0.1
    n_decay = 0.9
    wr_time = 5
    wr_factor = 1.75
    counter = 1
    counter_long = 1

    def __init__(self):
        pass
    
    def getstepLR():


        output = LR.n_min + 0.5 * (LR.n_max*(LR.n_decay**LR.counter_long) - LR.n_min) * (1 + np.cos(np.pi * LR.counter / LR.wr_time))


        if (LR.counter % LR.wr_time == 0):
            LR.wr_time = int(LR.wr_time * LR.wr_factor)
            LR.counter = 1
        else:
            LR.counter += 1

        LR.counter_long +=1
        
        return float(output)