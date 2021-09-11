from enum import Enum, auto, unique

@unique
class ModelCompatibilityRefinementLossType(Enum):
    discriminator = auto()
    latent = auto()
    LowFrequency = auto()

    angular = auto()
    fk = auto()
    posVel = auto()
    RotHoriz = auto()
    RotVertVel = auto()
    height = auto()
    feet = auto()
    contacts = auto()
    
    angularSmooth = auto()
    fkSmooth = auto()
    poseVelSmooth = auto()
    RotHorizSmooth = auto()
    RotVertVelSmooth = auto()
    heightSmooth = auto()
    feetSmooth = auto()
