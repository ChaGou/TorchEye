from enum import Enum, unique

@unique
class LearningMode(Enum):
    Regression = 0 # Sun的value被设定为0
    Classification2LabelsOneHot = 1
    Classification1Label = 2
    Classification1LabelHeatMap = 3
class DataMode(Enum):
    OriginMode = 0
    DeltaMode = 1
    SquareMode = 2

OutputShape = [640, 480]
learnMode = LearningMode.Regression
dataMode = DataMode.SquareMode