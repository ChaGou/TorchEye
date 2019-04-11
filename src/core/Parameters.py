from enum import Enum, unique

@unique
class Mode(Enum):
    Regression = 0 # Sun的value被设定为0
    Classification2LabelsOneHot = 1
    Classification1Label = 2
    Classification1LabelHeatMap = 3


OutputShape = [640, 480]