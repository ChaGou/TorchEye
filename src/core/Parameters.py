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
dataMode = DataMode.OriginMode
picWidth = 1280
picHeight = 720
picDepth = 1000


cx=633.87982177734375000
cy=364.74035644531250000
fx=711.52478027343750000
fy=712.65344238281250000