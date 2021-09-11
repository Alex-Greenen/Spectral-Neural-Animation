from typing import List
from enum import Enum

class TrainingLoss:
    def __init__(self, lossTypeEnum:Enum, losses:List):
        """Manages the loss during the training process

        Args:
            lossTypeEnum (Enum): enum type that is used to keep track of the losses
            losses (List): list of losses to use
        """

        self.leng = len(list(lossTypeEnum))

        assert self.leng == len(losses)
        self.lossType = lossTypeEnum
        self.losses = losses

        self.weights = [1.] * self.leng

    def __call__(self):
        return self.losses

    def getValues(self): return self.losses
    
    def getNames(self): return [str(l.name) for l in list(self.lossType)]

    def length(self): return self.leng

    def __len__(self): return self.leng

    def makefloat(self): return [float(item) for item in self.losses]

    def applyWeights(self, weights):
        assert len(weights) == self.leng
        for i in range(self.leng):
            self.losses[i] *= weights()[i]
            self.weights[i] *= weights()[i]      
    
    def getUnweighted(self):
        return TrainingLoss(self.lossType, [((self.losses[i] / self.weights[i]) if self.weights[i] != 0 else 0) for i in range(self.leng)])    