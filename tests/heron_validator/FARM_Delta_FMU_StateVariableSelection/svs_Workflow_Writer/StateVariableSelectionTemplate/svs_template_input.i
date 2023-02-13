[fmuInfo]
# fmuFile: TODO: How to use #FARM# to specify relative path?
fmuFile = ../FMUs/BOP_SES_empty_MWinput.fmu
# inputVar: deliminate multiple input variables by comma (,)
inputVar = SES_Demand_MW, BOP_Demand_MW
# outputVar: deliminate each subgroup by semicolon (;), and deliminate each output variable of the same subgroupt by comma (,)
outputVar = SES_Electric_Power, SES_Firing_Temperature; BOP_Electric_Power, BOP_Turbine_Pressure


[simulationInfo]
# fmuStepSize: float number, measured in seconds
fmuStepSize = 10.0
# setpointShiftStep: number of time steps to shift adjacent setpoints. integer, minimum 2
setpointShiftStep = 2
# inputTransients: square wave transients, deliminate each profile by semicolon(;)
# within each profile: intialValueForInput1, secondValueForInput1, intialValueForInput2, secondValueForInput2, ... 
# The scheduling parameter is the combination of secondValueForInput*
# User can break this entry to multiple lines, using tab at the beginning of 2nd line and beyond. 
inputTransients = 
    17,24,1075,1086;
    17,24,1086,1097;
    17,24,1097,1108;
    17,24,1108,1119;
    17,24,1119,1130;
    17,24,1130,1075;
    24,31,1075,1086;
    24,31,1086,1097;
    24,31,1097,1108;
    24,31,1108,1119;
    24,31,1119,1130;
    24,31,1130,1075;
    31,38,1075,1086;
    31,38,1086,1097;
    31,38,1097,1108;
    31,38,1108,1119;
    31,38,1119,1130;
    31,38,1130,1075;
    38,45,1075,1086;
    38,45,1086,1097;
    38,45,1097,1108;
    38,45,1108,1119;
    38,45,1119,1130;
    38,45,1130,1075;
    45,52,1075,1086;
    45,52,1086,1097;
    45,52,1097,1108;
    45,52,1108,1119;
    45,52,1119,1130;
    45,52,1130,1075;
    52,17,1075,1086;
    52,17,1086,1097;
    52,17,1097,1108;
    52,17,1108,1119;
    52,17,1119,1130;
    52,17,1130,1075
# periodToChange: float, the seconds to change input values (usually each hour) 
periodToChange = 3600

[outputDataInfo]
# outputFolder: TODO: How to use #FARM# to specify relative path?
outputFolder = StateVariableSelectionTemplate/simulatedData
# outputTimeStart: float number, measured in seconds. All the data before this time value will be cropped out.
outputTimeStart = 1800
# outputTimeEnd: float number, measured in seconds. The simulation will end at this time value
outputTimeEnd = 18000

[featureSelectionInfo]
# maxParallelCores: will be written in <RunInfo> - <batchSize>. 
# integer, number of cores used when looping through all possible combinations.
maxParallelCores = 8
# maxFeaturesPerSubgroup: will be written in <Models> - <ROM> - <featureSelection> - <RFE> - <maxNumberFeatures>. 
# integer, number of starting features of each subgroup. Maximum number of candidate state variables for each IES component.
maxFeaturesPerSubgroup = 4
