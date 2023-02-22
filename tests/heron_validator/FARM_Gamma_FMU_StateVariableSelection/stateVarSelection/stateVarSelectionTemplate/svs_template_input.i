[fmuInfo]
# fmuFile: related to the "svs_maker.py"
# fmuFile = ../FMUs/BOP_empty_MWinput.fmu
fmuFile = ../FMUs/SES_empty_MWinput.fmu
# inputVar: delimit multiple input variables by comma (,)
# inputVar = BOP_Demand_MW
inputVar = SES_Demand_MW
# outputVar: delimit each subgroup by semicolon (;), and delimit each output variable of the same subgroupt by comma (,)
# outputVar = Electric_Power, Turbine_Pressure
outputVar = Electric_Power, Firing_Temperature

[simulationInfo]
# fmuStepSize: float number, measured in seconds
fmuStepSize = 10.0
# setpointShiftStep: number of time steps to shift adjacent setpoints. integer, minimum 2 (of course, it doen't matter for Single-Input system)
setpointShiftStep = 2
# inputTransients: square wave transients, deliminate each profile by semicolon(;)
# within each profile: intialValueForInput1, secondValueForInput1, intialValueForInput2, secondValueForInput2, ... 
# The scheduling parameter is the combination of secondValueForInput*
# User can break this entry to multiple lines, using tab at the beginning of 2nd line and beyond. 
# inputTransients = 
#     1075,1086;
#     1086,1097;
#     1097,1108;
#     1108,1119;
#     1119,1130;
#     1130,1075
inputTransients = 
    17,24;
    24,31;
    31,38;
    38,45;
    45,52;
    52,17
    
# periodToChange: float, the seconds to change input values (usually each hour) 
periodToChange = 3600

[outputDataInfo]
# outputTimeStart: float number, measured in seconds. All the data before this time value will be cropped out.
outputTimeStart = 1800
# outputTimeEnd: float number, measured in seconds. The simulation will end at this time value
outputTimeEnd = 5400
# outputFolder: TODO: How to use #FARM# to specify relative path?
# outputFolder = stateVarSelectionTemplate/simulatedDataBOP
outputFolder = stateVarSelectionTemplate/simulatedDataSES

[featureSelectionInfo]
# maxParallelCores: will be written in <RunInfo> - <batchSize>. 
# integer, number of cores used when looping through all possible combinations.
maxParallelCores = 8
# maxFeaturesPerSubgroup: will be written in <Models> - <ROM> - <featureSelection> - <RFE> - <maxNumberFeatures>. 
# integer, number of starting features of each subgroup. Maximum number of candidate state variables for each IES component.
maxFeaturesPerSubgroup = 4
