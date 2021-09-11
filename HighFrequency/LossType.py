from enum import Enum, auto, unique

@unique
class HighFrequencyLossType(Enum):
    Latent = auto()
    Discriminator = auto()

    Angular = auto()
    Fk = auto()
    PosVel = auto()
    RotHoriz = auto()
    RotVertVel = auto()
    Height = auto()
    Feet = auto()
    Contacts = auto()
    
    AngularSmooth = auto()
    FkSmooth = auto()
    PoseVelSmooth = auto()
    RotHorizSmooth = auto()
    RotVertVelSmooth = auto()
    HeightSmooth = auto()
    FeetSmooth = auto()

