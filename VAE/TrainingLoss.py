from enum import Enum, auto, unique

@unique
class lossType(Enum):
    Kld = auto()
    LatentSmooth = auto()
    DirectRecon = auto()
    DirectReconSmooth = auto()
    FkRecon = auto()
    FkReconSmooth = auto()
    PosVel = auto()
    PosVelSmooth = auto()
    RotVel = auto()
    RotVelSmooth = auto()
    Height = auto()
    HeightSmooth = auto()

class TrainingLoss:
    def __init__(self, Kld = None, LatentSmooth = None, DirectRecon = None, DirectReconSmooth = None, FkRecon = None, FkReconSmooth = None, PosVel = None, PosVelSmooth = None, RotVel = None, RotVelSmooth = None, Height = None, HeightSmooth = None):
        self.Kld = Kld
        self.LatentSmooth = LatentSmooth
        self.DirectRecon = DirectRecon
        self.DirectReconSmooth = DirectReconSmooth
        self.FkRecon = FkRecon
        self.FkReconSmooth = FkReconSmooth
        self.PosVel = PosVel
        self.PosVelSmooth = PosVelSmooth
        self.RotVel = RotVel
        self.RotVelSmooth = RotVelSmooth
        self.Height = Height
        self.HeightSmooth = HeightSmooth

        self.weights = None

    def __call__(self):
        return [self.Kld, self.LatentSmooth, self.DirectRecon, self.DirectReconSmooth, self.FkRecon, self.FkReconSmooth, self.PosVel, self.PosVelSmooth, self.RotVel, self.RotVelSmooth, self.Height, self.HeightSmooth]

    def getValues(self): return self.__call__()
    
    def getNames(): return [str(l.name) for l in list(lossType)]

    def length(): return len(TrainingLoss.getNames())

    def __len__(self): return TrainingLoss.length()

    def getNamesColor(): return [['green', 'red'][i%2] for i in range(len(lossType))]

    def makefloat(self):
      return [float(item) for item in self.__call__()]

    def applyWeights(self, weights):
      self.weights = weights
      
      self.Kld *= self.weights.Kld
      self.LatentSmooth *= self.weights.LatentSmooth
      self.DirectRecon *= self.weights.DirectRecon
      self.DirectReconSmooth *= self.weights.DirectReconSmooth
      self.FkRecon *= self.weights.FkRecon
      self.FkReconSmooth *= self.weights.FkReconSmooth
      self.PosVel *= self.weights.PosVel
      self.PosVelSmooth *= self.weights.PosVelSmooth
      self.RotVel *= self.weights.RotVel
      self.RotVelSmooth *= self.weights.RotVelSmooth
      self.Height *= self.weights.Height
      self.HeightSmooth *= self.weights.HeightSmooth
      
    
    def getUnweighted(self):
      if self.weights != None:
        return TrainingLoss(
          self.Kld / self.weights.Kld,
          self.LatentSmooth / self.weights.LatentSmooth,
          self.DirectRecon / self.weights.DirectRecon,
          self.DirectReconSmooth / self.weights.DirectReconSmooth,
          self.FkRecon / self.weights.FkRecon,
          self.FkReconSmooth / self.weights.FkReconSmooth,
          self.PosVel / self.weights.PosVel,
          self.PosVelSmooth / self.weights.PosVelSmooth,
          self.RotVel / self.weights.RotVel,
          self.RotVelSmooth / self.weights.RotVelSmooth,
          self.Height / self.weights.Height,
          self.HeightSmooth / self.weights.HeightSmooth)
      
      else: return None
    
