from enum import Enum, auto, unique

@unique
class VAELossType(Enum):
    Kld = auto()
    Angular = auto()
    Fk = auto()
    PosVel = auto()
    RotHoriz = auto()
    RotVertVel = auto()
    Height = auto()
    Feet = auto()
    
    LatentSmooth = auto()
    AngularSmooth = auto()
    FkSmooth = auto()
    PoseVelSmooth = auto()
    RotHorizSmooth = auto()
    RotVertVelSmooth = auto()
    HeightSmooth = auto()
    FeetSmooth = auto()

    Contacts = auto()