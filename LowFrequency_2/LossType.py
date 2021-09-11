from enum import Enum, auto, unique

@unique
class LowFrequencyLossType(Enum):
    Latent = auto()
    Angular = auto()
    Fk = auto()
    PosVel = auto()
    RotHoriz = auto()
    RotVertVel = auto()
    Height = auto()
    Feet = auto()
    Contacts = auto()
