# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Created on Jan 26, 2023
@author: haoyuwang

Template interface for the state variable seletcion (svs) template input.
1. Load information from "svs_template_input.i"
2. Simulate the FMU using provided input transients, save data in csv
3. Create RAVEN input file for state variable selection application.
"""
from __future__ import print_function, unicode_literals
import os
import configparser
from collections import OrderedDict
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import numpy as np
import copy
import math
import csv

from StateVariableSelectionTemplate.svsTemplate import svsTemplate


"""1. Load information from 'svs_template_input.i' """
print('Loading input file ...')
config = configparser.ConfigParser()
# find the current folder containing this "svs_maker.py" file
current_folder = os.path.dirname(os.path.realpath(__file__))
# construct the full path of "svs_template_input.i"
file_path = os.path.join(current_folder,'StateVariableSelectionTemplate','svs_template_input.i')
# read the input file
config.read(file_path)
print('Loaded input file:', file_path, '\n')

# extract the section of "fmuInfo"
fmuInfo= {
  'fmuFile': config.get('fmuInfo', 'fmuFile'),
  'inputVar': list(x.strip() for x in config.get('fmuInfo', 'inputVar').split(',')),
  'outputVar': list(list(y.strip() for y in x.split(',')) for x in config.get('fmuInfo', 'outputVar').split(';'))
}
# extract the section of "simulationInfo"
simulationInfo = {
  'fmuStepSize': config.getfloat('simulationInfo', 'fmuStepSize'),
  'setpointShiftStep': config.getint('simulationInfo', 'setpointShiftStep'),
  'inputTransients': list(list(float(y.strip()) for y in x.split(',')) for x in config.get('simulationInfo', 'inputTransients').split(';')),
  'periodToChange': config.getfloat('simulationInfo', 'periodToChange')
}
# extract the section of "outputDataInfo"
outputDataInfo = {
  'outputFolder': config.get('outputDataInfo', 'outputFolder'),
  'outputTimeStart': config.getfloat('outputDataInfo', 'outputTimeStart'),
  'outputTimeEnd': config.getfloat('outputDataInfo', 'outputTimeEnd')
}
# extract the section of "featureSelectionInfo"
featureSelectionInfo = {
  'maxParallelCores': config.getint('featureSelectionInfo','maxParallelCores'),
  'maxFeaturesPerSubgroup': config.getint('featureSelectionInfo','maxFeaturesPerSubgroup')
}

# print to confirm
print(fmuInfo)
print(simulationInfo)
print(outputDataInfo)
print(featureSelectionInfo,'\n')

""" 2. Load FMU and prepare the variable name lists """
# FMU file name, MIMO version, using "os.path.normpath" to convert it to os-acceptable path pattern
fmu_filename = os.path.join(current_folder,os.path.normpath(fmuInfo['fmuFile']))

# FMU simulation start time (s) and step size (s)
T_delaystart = 0.; fmu_stepsize=simulationInfo['fmuStepSize']
# Input variables
inputVarNames = fmuInfo['inputVar'] # a list of 2 variables
# flattened Output variables. fmuInfo['outputVar'] is a list contains multiple subgroups
outputVarNames = [item for subgroup in fmuInfo['outputVar'] for item in subgroup] # a list of 4 variables

# read the model description
model_description = read_model_description(fmu_filename)
# collect all the process variables
stateVarNames = [variable.name for variable in model_description.modelVariables] # a list of 2287 process variables
for item in inputVarNames:
   stateVarNames.remove(item)
for item in outputVarNames:
   stateVarNames.remove(item) # now the input and output variables are removed, 2281 process variables left
# print(len(stateVarNames),stateVarNames)
# Dimensions of input (m), states (n) and output (p)
m=len(inputVarNames); p=len(outputVarNames); n=len(stateVarNames); 
# print(m,p)

# collect the value references
vrs = {}
for variable in model_description.modelVariables:
    vrs[variable.name] = variable.valueReference
# get the value references for the variables we want to get/set
# Input Power Setpoint (W)
vr_input = [vrs[item] for item in inputVarNames] # list
# State variables and dimension
vr_state = [vrs[item] for item in stateVarNames] # list
# Outputs: BOP_Electric_Power (W), BOP_Turbine_Pressure (Pa), SES_Electric_Power (W), SES_Firing_Temperature (K)
vr_output = [vrs[item] for item in outputVarNames] # list

# extract the FMU
unzipdir = extract(fmu_filename)
fmu = FMU2Slave(guid=model_description.guid,
                unzipDirectory=unzipdir,
                modelIdentifier=model_description.coSimulation.modelIdentifier,
                instanceName='instance1')

""" 3. Simulate the FMU using provided input transients, save data in csv """
setpoints_shift_step = simulationInfo['setpointShiftStep']
inputTransients = np.asarray(simulationInfo['inputTransients'])
# print(type(inputTransients),inputTransients.shape) # <class 'numpy.ndarray'> (3, 4)
Tr_Update = simulationInfo['periodToChange']
T_start = T_delaystart; 
T_end = outputDataInfo['outputTimeEnd']
# print(T_start,T_end)
tTran = np.arange(T_start+Tr_Update, T_end+Tr_Update, Tr_Update) # the moments when setpoint change [ 3600.  7200. 10800. 14400. 18000.]
t_ext = np.arange(T_start, T_end, fmu_stepsize) #[0, 10, 20, ... 17990]
# print(tTran) 
# print(t_ext) 
outputTimeStart = outputDataInfo['outputTimeStart']

# simulate according to each entry of "inputTransients"
for i_case in range(len(inputTransients)): # i_case = 0,1,2
  # Initialize the history arrays
  t_hist = []  # the history of time stamps
  r_hist = []  # the history of r signal (vector)
  x_hist = []  # the history of x signal (vector)
  y_hist = []  # the history of y signal (vector)

  # collecting the scheduling parameter, using the 2nd values
  transientData = inputTransients[i_case].reshape(-1,2).T
  # print(inputTransients)
  # print(transientData)
  # print(a)
  schedulingParameter = transientData[1]
  # print(i_case, schedulingParameter)
  
  # Initialize FMU
  fmu.instantiate()
  fmu.setupExperiment(startTime=T_delaystart)
  fmu.enterInitializationMode()
  fmu.exitInitializationMode()
  
  # Initial time and time index
  t = T_start
  i_hour = 0

  # simulate the 1st setpoint vector, no input shift
  while t < tTran[i_hour]:
    # find the current r_spec
    if np.mod(i_hour,2) == 0:
      r_spec = copy.deepcopy(transientData[0])
    else:
      r_spec = copy.deepcopy(transientData[1])
    
    # Find the current r value
    r = copy.deepcopy(r_spec)
    
    # print("i_case=", i_case, ", t=", t, ", r=",r)

    # fetch y
    y_fetch = np.asarray(fmu.getReal(vr_output)) # shape = (4,)
    # fetch r and x
    r_fetch = np.asarray(r).reshape(m,) # shape = (2,)
    x_fetch = np.asarray(fmu.getReal(vr_state)) # shape = (6,)
    # print("y_fetch:",type(y_fetch), y_fetch.shape, y_fetch)
    # print("v_fetch:",type(v_fetch), v_fetch.shape, v_fetch)
    # print("x_fetch:",type(x_fetch), x_fetch.shape, x_fetch)

    # Save time, r, v, and y
    if t >= outputTimeStart:
      t_hist.append(copy.deepcopy(t))         # Time
      r_hist.append(copy.deepcopy(r))         # reference setpoint r
      x_hist.append(copy.deepcopy(x_fetch))   # state x
      y_hist.append(copy.deepcopy(y_fetch))   # output y

    # set the input. The input / state / output in FMU are real-world values
    for i in range(len(vr_input)):
        fmu.setReal([vr_input[i]], [r[i]]) # Note: fmu.setReal must take two lists as input
    # perform one step
    fmu.doStep(currentCommunicationPoint=t, communicationStepSize=fmu_stepsize)

    # time increment
    t = t + fmu_stepsize
  
  i_hour += 1

  # simulate from the 2nd setpoint vector, apply input shift
  while i_hour < len(tTran):
    # find the current r_spec
    if np.mod(i_hour,2) == 0:
      r_spec = copy.deepcopy(transientData[0])
    else:
      r_spec = copy.deepcopy(transientData[1])

    # Shift input
    i_input = 0.
    while t < tTran[i_hour]: # shift setpoints at the beginning of 2nd hour
      
      if i_input < m:
            r[math.floor(i_input)]=copy.deepcopy(r_spec[math.floor(i_input)])
            i_input += 1/setpoints_shift_step
      
      # print("i_case=", i_case, ", t=", t, ", r=",r)

      # fetch y
      y_fetch = np.asarray(fmu.getReal(vr_output)) # shape = (4,)
      # fetch r and x
      r_fetch = np.asarray(r).reshape(m,) # shape = (2,)
      x_fetch = np.asarray(fmu.getReal(vr_state)) # shape = (6,)
      # print("y_fetch:",type(y_fetch), y_fetch.shape, y_fetch)
      # print("v_fetch:",type(v_fetch), v_fetch.shape, v_fetch)
      # print("x_fetch:",type(x_fetch), x_fetch.shape, x_fetch)

      # Save time, r, v, and y
      if t >= outputTimeStart:
        t_hist.append(copy.deepcopy(t))         # Time
        r_hist.append(copy.deepcopy(r))         # reference setpoint r
        x_hist.append(copy.deepcopy(x_fetch))   # state x
        y_hist.append(copy.deepcopy(y_fetch))   # output y

      # set the input. The input / state / output in FMU are real-world values
      for i in range(len(vr_input)):
          fmu.setReal([vr_input[i]], [r[i]]) # Note: fmu.setReal must take two lists as input
      # perform one step
      fmu.doStep(currentCommunicationPoint=t, communicationStepSize=fmu_stepsize)

      # time increment 
      t = t + fmu_stepsize
    
    i_hour += 1


  # concatenate t_hist, r_hist, y_hist and x_hist
  # print(np.asarray(t_hist).shape, np.asarray(r_hist).shape, np.asarray(y_hist).shape, np.asarray(x_hist).shape)
  data_array = np.concatenate((np.asarray(t_hist).reshape(-1,1), np.asarray(r_hist).reshape(-1,m), np.asarray(y_hist).reshape(-1,p), np.asarray(x_hist).reshape(-1,n)),axis=1)
  # print(data_array.shape)

  # save case data to csv
  folder_to_write = os.path.join(current_folder,os.path.normpath(outputDataInfo['outputFolder']))
  # create folder if not exist
  if not os.path.exists(folder_to_write):
    os.mkdir(folder_to_write)
  # assemble csv filename for data
  data_csv_filename = 'DataFile_%d.csv'%(i_case)
  data_csv_fullname = os.path.join(folder_to_write, data_csv_filename)
  # print(data_csv_fullname)
  # assemble header (list of strings) and data_str_list (with given format)
  csv_header = ['Time'] + inputVarNames + outputVarNames + stateVarNames # list of strings
  data_str_list = [['%.9e'%(item) for item in row] for row in data_array]
  # print(data_str_list)

  # write to individual csv file
  with open(data_csv_fullname,"w", newline='') as data_csv_file:  
    writer = csv.writer(data_csv_file, delimiter=',')
    writer.writerow(csv_header)
    writer.writerows(data_str_list)
    data_csv_file.close()

  # append to index csv file "outputData.csv"
  index_csv_fullname = os.path.join(folder_to_write,'outputData.csv')
  
  # components in header
  # 1. the list of scheduling parameter names
  schedulingParameterName = ['schedulingParameter_%d'%(i_sp) for i_sp in range(m)]
  # 2. miscellaneous var names required for index csv file
  miscVarName = ['PointProbability','ProbabilityWeight-mod','prefix','ProbabilityWeight','ProbabilityWeight-flow']
  # 3. initial var names
  initVarName = [item+'_init' for item in stateVarNames]
  # print("############", schedulingParameterName, schedulingParameter)
  # print("############", miscVarName)
  # print("############", initVarName)
  
  # if this is the first entry, create file and output headers
  if i_case == 0:
    # concantenate the header
    idx_header = schedulingParameterName + miscVarName + initVarName + ['filename']
    # print(idx_header)
    
    with open(index_csv_fullname, "w", newline='') as index_csv_file:
      writer = csv.writer(index_csv_file, delimiter=',')
      writer.writerow(idx_header)
      index_csv_file.close()

  # append the scheduling parameter, prefix, initial data values and filename to index csv file
  schedulingParameterLine = ['%.9e'%(item) for item in schedulingParameter]
  miscVarLine = ['%.9e'%(item) for item in [1, 1, int(i_case+1), 1, 1]]
  initVarLine = ['%.9e'%(item) for item in x_hist[0]]

  idx_line = schedulingParameterLine + miscVarLine + initVarLine + [data_csv_filename]
  with open(index_csv_fullname, "a", newline='') as index_csv_file:
    writer = csv.writer(index_csv_file, delimiter=',')
    writer.writerow(idx_line)
    index_csv_file.close()


# todo: read template, collect info, and write raven input file

# load template
print('Loading template ...')
temp = svsTemplate()
temp.loadTemplate('svs_inputFile_template.xml', "")
print(' ... template loaded')

# write to new file
print('Writing RAVEN file ...')
template = temp.createWorkflow(WorkingDir=outputDataInfo['outputFolder'], 
                               batchSize=featureSelectionInfo['maxParallelCores'], 
                               maxNumberFeatures=featureSelectionInfo['maxFeaturesPerSubgroup'], 
                               subGroup=fmuInfo['outputVar'], 
                               scheduling_paras=schedulingParameterName, 
                               actuator_variables=inputVarNames, 
                               output_variables=outputVarNames, 
                               state_variables=stateVarNames, 
                               state_variables_init=initVarName)
outputXMLpath = os.path.join(current_folder,"svs_new_inputFile.xml")
errors = temp.writeWorkflow(template, outputXMLpath, run=False)

# finish up
if errors == 0:
  print('\n\nSuccessfully wrote input "{}". Run it with RAVEN!\n'.format(outputXMLpath))
else:
  print('\n\nProblems occurred while running the code. See above.\n')
