
# Copyright 2020, Battelle Energy Alliance, LLC
# ALL RIGHTS RESERVED
"""
  Example class for validators.
"""
import numpy as np
# import pickle as pk
import os
import sys
import xml.etree.ElementTree as ET
import math
import scipy
from fmpy import read_model_description, extract
from fmpy.fmi2 import FMU2Slave
import copy
from sklearn import neighbors
# import cvxpy as cp
import matplotlib.pyplot as plt
import time
from datetime import datetime
from itertools import combinations
from scipy.spatial import ConvexHull
from scipy.optimize import fmin
from _utils import get_raven_loc, get_heron_loc, get_farm_loc
from validators.Validator import Validator
from scipy.io import savemat

import cvxpy as cp
import cvxopt
import quadprog
import qpsolvers

# set up raven path
raven_path = get_raven_loc()
sys.path.append(raven_path)
from ravenframework.utils import InputData, InputTypes

class FARM_Beta(Validator):
  """
    A FARM SISO Validator for dispatch decisions.(Dirty Implementation)
    Accepts parameterized A,B,C,D matrices from external XML file, and validate
    the dispatch power (BOP, SES & TES, unit=MW), and
    the next stored energy level (TES, unit=MWh)

    Haoyu Wang, ANL-NSE, Jan 6, 2022
  """
  # ---------------------------------------------
  # INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = Validator.get_input_specs()
    specs.name = 'FARM_Beta'
    specs.description = r"""Feasible Actuator Range Modifier, which uses a single-input-single-output
        reference governor validator to adjust the power setpoints issued to the components at the
        beginning of each dispatch interval (usually an hour), to ensure the the operational constraints
        were not violated during the following dispatch interval."""

    component = InputData.parameterInputFactory('ComponentForFARM', ordered=False, baseNode=None,
        descr=r"""The component whose power setpoint will be adjusted by FARM. The user need
        to provide the statespace matrices and operational constraints concerning this component,
        and optionally provide the initial states.""")
    component.addParam('name',param_type=InputTypes.StringType, required=True,
        descr=r"""The name by which this component should be referred within HERON. It should match
        the component's name in \xmlNode{Components}.""")

    component.addSub(InputData.parameterInputFactory('MatricesFile',contentType=InputTypes.StringType,
        descr=r"""The path to the Statespace representation matrices file of this component. Either absolute path
        or path relative to HERON root (starts with %HERON%/)will work. The matrices file can be generated from
        RAVEN DMDc or other sources."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsUpper',contentType=InputTypes.InterpretedListType,
        descr=r"""The upper bounds for the output variables of this component. It should be a list of
        floating numbers or integers."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsLower',contentType=InputTypes.InterpretedListType,
        descr=r"""The lower bounds for the output variables of this component. It should be a list of
        floating numbers or integers."""))
    component.addSub(InputData.parameterInputFactory('InitialState',contentType=InputTypes.InterpretedListType,
        descr=r"""The initial system state vector of this component. It should be a list of
        floating numbers or integers. This subnode is OPTIONAL in the HERON input file, and FARM will
        provide a default initial system state vector if \xmlNode{InitialState} is not present."""))

    specs.addSub(component)

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'
    self._tolerance = 1.003e-6

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    self._unitInfo = {}
    for component in inputs.subparts:
      name = component.parameterValues['name']
      xInit=[]
      for farmEntry in component.subparts:
        if farmEntry.getName() == "MatricesFile":
          matFile = farmEntry.value
          if matFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            matFile = os.path.abspath(matFile.replace('%HERON%', heron_path))
          elif matFile.startswith('%FARM%'):
            # magic word for "relative to HERON root"
            farm_path = get_farm_loc()
            matFile = os.path.abspath(matFile.replace('%FARM%', farm_path))
        if farmEntry.getName() == "OpConstraintsUpper":
          UpperBound = farmEntry.value
        if farmEntry.getName() == "OpConstraintsLower":
          LowerBound = farmEntry.value
        if farmEntry.getName() == "InitialState":
          xInit = farmEntry.value
      self._unitInfo.update({name:{'MatrixFile':matFile,'Targets_Max':UpperBound,'Targets_Min':LowerBound,'XInit':xInit,'v_hist':[],'y_hist':[]}})

  # ---------------------------------------------
  # API
  def validate(self, components, dispatch, times, meta):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, extra information pertaining to validation
      @ Out, errs, list, information about validation failures
    """
    # get start time
    startTime = copy.deepcopy(datetime.now())
    # errs will be returned to dispatcher. errs contains all the validation errors calculated in below
    errs = [] # TODO best format for this?

    # get time interval
    Tr_Update_hrs = float(times[1]-times[0])
    Tr_Update_sec = Tr_Update_hrs*3600.

    # loop through the <Component> items in HERON
    for comp, info in dispatch._resources.items():
      # e.g. comp= <HERON Component "SES""> <HERON Component "SES"">
      # loop through the items defined in the __init__ function
      for unit in self._unitInfo:
        # e.g. CompInfo, unit= SES
        # Identify the profile as defined in the __init__ function
        if str(unit) not in str(comp):
          # If the "unit" and "comp" do not match, go to the next "unit" in loop
          continue
        else: # when the str(unit) is in the str(comp) (e.g. "SES" in "<HERON Component "SES"">")
          self._unitInfo[unit]['v_hist']=[]; self._unitInfo[unit]['y_hist']=[]
          """ Read State Space XML file (generated by Raven parameterized DMDc) """
          MatrixFile = self._unitInfo[unit]['MatrixFile']
          Tss, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array, actuatorList, stateList, outputList = read_parameterized_XML(MatrixFile)

          """ MOAS steps Limit """
          g = int(Tr_Update_sec/Tss)+1 # numbers of steps to look forward, , type = <class 'int'>
          """ Keep only the profiles with YNorm within the [y_min, y_max] range """
          y_min = self._unitInfo[unit]['Targets_Min']
          y_max = self._unitInfo[unit]['Targets_Max']

          para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array = check_YNorm_within_Range(
            y_min, y_max, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array)

          if YNorm_list == []:
            print('ERROR:  No proper linearization point (YNorm) found in Matrix File. \n\tPlease provide a state space profile linearized within the [y_min, y_max] range\n')
            sys.exit('ERROR:  No proper linearization point (YNorm) found in Matrix File. \n\tPlease provide a state space profile linearized within the [y_min, y_max] range\n')
          max_eigA_id = eig_A_array.argmax()
          A_m = A_list[max_eigA_id]; B_m = B_list[max_eigA_id]; C_m = C_list[max_eigA_id]; D_m = np.zeros((p,m)) # all zero D matrix

          for tracker in comp.get_tracking_vars():
            # loop through the resources in info (only one resource here - electricity)
            for res in info:
              if str(res) == "electricity":
                # loop through the time index (tidx) and time in "times"
                if self._unitInfo[unit]['XInit']==[]:
                  x_sys = np.zeros(n)
                else:
                  x_sys = np.asarray(self._unitInfo[unit]['XInit'])-XNorm_list[0]


                for tidx, time in enumerate(times):
                  # Copy the system state variable
                  x_KF = x_sys
                  """ Get the r_value, original actuation value """
                  current = float(dispatch.get_activity(comp, tracker, res, times[tidx]))
                  # check if storage: power = (curr. MWh energy - prev. MWh energy)/interval Hrs

                  if comp.get_interaction().is_type('Storage') and tidx == 0:
                    init_level = comp.get_interaction().get_initial_level(meta)

                  # if storage:
                  if comp.get_interaction().is_type('Storage'):
                    # Initial_Level = float(self._unitInfo[unit]['Initial_Level'])
                    Initial_Level = float(init_level)
                    if tidx == 0: # for the first hour, use the initial level. charging yields to negative r_value
                      r_value = -(current - Initial_Level)/Tr_Update_hrs
                    else: # for the other hours
                      # r_value = -(current - float(dispatch.get_activity(comp, tracker, res, times[tidx-1])))/Tr_Update_hrs
                      r_value = -(current - Allowed_Level)/Tr_Update_hrs
                  else: # when not storage,
                    r_value = current # measured in MW

                  """ Find the correct profile according to r_value"""
                  profile_id = (np.abs(para_array - r_value)).argmin()

                  # Retrive the correct A, B, C matrices
                  A_d = A_list[profile_id]; B_d = B_list[profile_id]; C_d = C_list[profile_id]; D_d = np.zeros((p,m)) # all zero D matrix
                  # Retrive the correct y_0, r_0 and X
                  y_0 = YNorm_list[profile_id]; r_0 = float(UNorm_list[profile_id])

                  # Build the s, H and h for MOAS
                  s = [] # type == <class 'list'>
                  for i in range(0,p):
                    s.append([abs(y_max[i] - y_0[i])])
                    s.append([abs(y_0[i] - y_min[i])])

                  H, h = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g) # H and h, type = <class 'numpy.ndarray'>

                  # first v_RG: consider the step "0" - step "g"
                  v_RG = fun_RG_SISO(0, x_KF, r_value-r_0, H, h, p) # v_RG: type == <class 'numpy.ndarray'>

                  """ 2nd adjustment """
                  # MOAS for the steps "g+1" - step "2g"
                  Hm, hm = fun_MOAS_noinf(A_m, B_m, C_m, D_m, s, g)
                  # Calculate the max/min for v, ensuring the hm-Hxm*x(g+1) always positive for the next g steps.
                  v_max, v_min = fun_2nd_gstep_calc(x_KF, Hm, hm, A_m, B_m, g)

                  if v_RG < v_min:
                    v_RG = v_min
                  elif v_RG > v_max:
                    v_RG = v_max

                  # # Pretend there is no FARM intervention
                  # v_RG = np.asarray(r_value-r_0).flatten()

                  v_value = v_RG + r_0 # absolute value of electrical power (MW)
                  v_value = float(v_value)

                  # Update x_sys, and keep record in v_hist and yp_hist within this hour
                  for i in range(int(Tr_Update_sec/Tss)):
                    self._unitInfo[unit]['v_hist'].append(v_value)
                    y_sim = np.dot(C_d,x_sys)
                    self._unitInfo[unit]['y_hist'].append(y_sim+y_0)
                    x_sys = np.dot(A_d,x_sys)+np.dot(B_d,v_RG)

                  # Convert to V1:

                  # if storage
                  if comp.get_interaction().is_type('Storage'):
                    if tidx == 0: # for the first hour, use the initial level
                      Allowed_Level = Initial_Level - v_value*Tr_Update_hrs # Allowed_Level: predicted level due to v_value
                    else: # for the other hours
                      Allowed_Level = Allowed_Level - v_value*Tr_Update_hrs
                    V1 = Allowed_Level
                  else: # when not storage,
                    V1 = v_value

                  # print("Haoyu Debug, unit=",str(unit),", t=",time, ", curr= %.8g, V1= %.8g, delta=%.8g" %(current, V1, (V1-current)))

                  # Write up any violation to the errs:
                  if abs(current - V1) > self._tolerance*max(abs(current),abs(V1)):
                    # violation
                    errs.append({'msg': f'Reference Governor Violation',
                                'limit': V1,
                                'limit_type': 'lower' if (current < V1) else 'upper',
                                'component': comp,
                                'resource': res,
                                'time': time,
                                'time_index': tidx,
                                })
    # Get time for dispatching
    endTimeDispatch = copy.deepcopy(datetime.now())
    print('Haoyu t-debug, Time for this Dispatch is {}'.format(endTimeDispatch-startTime))


    if errs == []: # if no validation error:
      print(" ")
      print("*********************************************************************")
      print("*** Haoyu Debug, Validation Success, Print for offline processing ***")
      print("*********************************************************************")
      print(" ")

      for unit in self._unitInfo:
        t_hist = np.arange(0,len(self._unitInfo[unit]['v_hist'])*Tss,Tss) # t_hist.shape = (4320,)
        v_hist = np.array(self._unitInfo[unit]['v_hist']).T # v_hist.shape = (4320,)
        y_hist = np.array(self._unitInfo[unit]['y_hist']).T # y_hist.shape = (2, 4320)        
        # print(str(unit), t_hist.shape, v_hist.shape, y_hist.shape)

        for i in range(len(t_hist)):
          print("HaoyuDispatchSummary, " + str(unit) + 
          ", t, " + str(t_hist[i]) + ", vp, " + str(v_hist[i]) +
          ", y, " + ', '.join([str(num) for num in y_hist[:,i]]) + 
          ", ymin, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Min']]) +
          ", ymax, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Max']]))


    return errs

class FARM_Gamma_LTI(Validator):
  """
    A FARM SISO Validator for dispatch decisions.(Dirty Implementation)
    Accepts parameterized A,B,C,D matrices from external XML file and use the first set within constraints 
    as physics model, and validate
    the dispatch power (BOP, unit=MW)

    Haoyu Wang, ANL-NSE, March 21, 2022
  """
  # ---------------------------------------------
  # INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = Validator.get_input_specs()
    specs.name = 'FARM_Gamma_LTI'
    specs.description = r"""Feasible Actuator Range Modifier, which uses a single-input-single-output
        reference governor validator to adjust the power setpoints issued to the components at the
        beginning of each dispatch interval (usually an hour), to ensure the the operational constraints
        were not violated during the following dispatch interval. This version uses a Linear Time Invariant (LTI)
        model as a downstream high-fidelity model."""

    component = InputData.parameterInputFactory('ComponentForFARM', ordered=False, baseNode=None,
        descr=r"""The component whose power setpoint will be adjusted by FARM. The user need
        to provide the statespace matrices and operational constraints concerning this component,
        and optionally provide the initial states.""")
    component.addParam('name',param_type=InputTypes.StringType, required=True,
        descr=r"""The name by which this component should be referred within HERON. It should match
        the component's name in \xmlNode{Components}.""")

    component.addSub(InputData.parameterInputFactory('MatricesFile',contentType=InputTypes.StringType,
        descr=r"""The path to the parameterized LTI matrices file of this component. Either absolute path
        or path relative to HERON root (starts with %HERON%/)will work. The matrices file can be generated from
        RAVEN DMDc or other sources."""))
    component.addSub(InputData.parameterInputFactory('SystemProfile',contentType=InputTypes.IntegerType,
        descr=r"""The system profile index in the parameterized matrices file. It should be an integer."""))
    component.addSub(InputData.parameterInputFactory('LearningSetpoints',contentType=InputTypes.InterpretedListType,
        descr=r"""The learning setpoints are used to find the nominal value and first sets of ABCD matrices. 
        It should be a list of two or more floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('NeighborSetpointsThreshold',contentType=InputTypes.FloatType,
        descr=r"""The Neighbor Setpoints Threshold is used to create a new ABCD matrices profile. 
        If a new setpoint falls within an existing setpoint plus or minus this threshold, no new profile will be created.
        It should be a floating number."""))
    component.addSub(InputData.parameterInputFactory('RollingWindowWidth',contentType=InputTypes.IntegerType,
        descr=r"""The moving window duration for DMDc, with the unit of seconds. It should be an integer."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsUpper',contentType=InputTypes.InterpretedListType,
        descr=r"""The upper bounds for the output variables of this component. It should be a list of
        floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsLower',contentType=InputTypes.InterpretedListType,
        descr=r"""The lower bounds for the output variables of this component. It should be a list of
        floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('InitialState',contentType=InputTypes.InterpretedListType,
        descr=r"""The initial system state vector of this component. It should be a list of
        floating numbers or integers separated by comma. This subnode is OPTIONAL in the HERON input file, and FARM will
        provide a default initial system state vector if \xmlNode{InitialState} is not present."""))

    specs.addSub(component)

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'
    self._tolerance = 1.003e-6

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    self._unitInfo = {}
    for component in inputs.subparts:
      name = component.parameterValues['name']
      xInit=[]
      NeighborSetpointsThreshold = 1.0 # default threshold 1.0 MW
      for farmEntry in component.subparts:
        if farmEntry.getName() == "MatricesFile":
          matFile = farmEntry.value
          if matFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            matFile = os.path.abspath(matFile.replace('%HERON%', heron_path))
          elif matFile.startswith('%FARM%'):
            # magic word for "relative to HERON root"
            farm_path = get_farm_loc()
            matFile = os.path.abspath(matFile.replace('%FARM%', farm_path))
        if farmEntry.getName() == "SystemProfile":
          systemProfile = farmEntry.value
        if farmEntry.getName() == "LearningSetpoints":
          LearningSetpoints = farmEntry.value
          if len(LearningSetpoints) < 2:
            sys.exit('\nERROR: <LearningSetpoints> XML node needs to contain 2 or more floating or integer numbers.\n')   
          elif min(LearningSetpoints)==max(LearningSetpoints):
            exitMessage = """ERROR:  No transient found in <LearningSetpoints>. \n\tPlease modify the values in <LearningSetpoints>.\n"""
            sys.exit(exitMessage)        
        if farmEntry.getName() == "NeighborSetpointsThreshold":
          NeighborSetpointsThreshold = farmEntry.value
        if farmEntry.getName() == "RollingWindowWidth":
          RollingWindowWidth = farmEntry.value
        if farmEntry.getName() == "OpConstraintsUpper":
          UpperBound = farmEntry.value
        if farmEntry.getName() == "OpConstraintsLower":
          LowerBound = farmEntry.value
        if farmEntry.getName() == "InitialState":
          xInit = farmEntry.value
      self._unitInfo.update(
        {name:{
          'MatrixFile':matFile,
          'systemProfile':systemProfile,
          'LearningSetpoints':LearningSetpoints,
          'NeighborSetpointsThreshold':NeighborSetpointsThreshold,
          'RollingWindowWidth':RollingWindowWidth,
          'Targets_Max':UpperBound,
          'Targets_Min':LowerBound,
          'XInit':xInit,
          't_hist_sl':[],
          'v_hist_sl':[],
          'x_hist_sl':[],
          'y_hist_sl':[],
          'v_0_sl': None,
          'x_0_sl': None,
          'y_0_sl': None,
          't_idx_sl': None,
          'A_list_sl':[],
          'B_list_sl':[],
          'C_list_sl':[],
          'eig_A_list_sl':[],
          'para_list_sl':[],
          'tTran_list_sl':[],
          't_hist':[],
          'v_hist':[],
          'x_hist':[],
          'y_hist':[],
          'A_list':[],
          'B_list':[],
          'C_list':[],
          'eig_A_list':[],
          'para_list':[],
          'tTran_list':[]}})
    print('\n',self._unitInfo,'\n')

  # ---------------------------------------------
  # API
  def validate(self, components, dispatch, times, meta):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, extra information pertaining to validation
      @ Out, errs, list, information about validation failures
    """
    # errs will be returned to dispatcher. errs contains all the validation errors calculated in below
    errs = [] # TODO best format for this?

    """ get time interval"""
    Tr_Update_hrs = float(times[1]-times[0])
    Tr_Update_sec = Tr_Update_hrs*3600.

    # loop through the <Component> items in HERON
    for comp, info in dispatch._resources.items():
      # e.g. comp= <HERON Component "SES""> <HERON Component "SES"">
      # loop through the items defined in the __init__ function
      for unit in self._unitInfo:
        # e.g. CompInfo, unit= SES
        # Identify the profile as defined in the __init__ function
        if str(unit) not in str(comp):
          # If the "unit" and "comp" do not match, go to the next "unit" in loop
          continue
        else: # when the str(unit) is in the str(comp) (e.g. "SES" in "<HERON Component "SES"">")
          # get start time
          startTime = copy.deepcopy(datetime.now())
          """ 1. Constraints information, Set-point trajectory, and Moving window width """
          # Constraints
          y_min = np.asarray(self._unitInfo[unit]['Targets_Min'])
          y_max = np.asarray(self._unitInfo[unit]['Targets_Max'])

          # The width of moving window (seconds, centered at transient edge, for moving window DMDc)
          Moving_Window_Width = self._unitInfo[unit]['RollingWindowWidth']; #Tr_Update

          # Load the self learning hists, if any
          self._unitInfo[unit]['t_hist']=copy.deepcopy(self._unitInfo[unit]['t_hist_sl'])
          self._unitInfo[unit]['v_hist']=copy.deepcopy(self._unitInfo[unit]['v_hist_sl'])
          self._unitInfo[unit]['x_hist']=copy.deepcopy(self._unitInfo[unit]['x_hist_sl'])
          self._unitInfo[unit]['y_hist']=copy.deepcopy(self._unitInfo[unit]['y_hist_sl'])
          # empty the A_list, B_list, C_list, eig_A_list, para_list, tTran_list
          self._unitInfo[unit]['A_list']=copy.deepcopy(self._unitInfo[unit]['A_list_sl'])
          self._unitInfo[unit]['B_list']=copy.deepcopy(self._unitInfo[unit]['B_list_sl'])
          self._unitInfo[unit]['C_list']=copy.deepcopy(self._unitInfo[unit]['C_list_sl'])
          self._unitInfo[unit]['eig_A_list']=copy.deepcopy(self._unitInfo[unit]['eig_A_list_sl'])
          self._unitInfo[unit]['para_list']=copy.deepcopy(self._unitInfo[unit]['para_list_sl'])
          self._unitInfo[unit]['tTran_list']=copy.deepcopy(self._unitInfo[unit]['tTran_list_sl'])
           
          
          """ 2. Read State Space XML file (generated by Raven parameterized DMDc) and generate the physical model"""
          MatrixFile = self._unitInfo[unit]['MatrixFile']
          Tss, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array, actuatorList, stateList, outputList = read_parameterized_XML(MatrixFile)
          # use the 8th profile as physical model
          systemProfile = self._unitInfo[unit]['systemProfile']
          A_sys = A_list[systemProfile]; B_sys = B_list[systemProfile]; C_sys = C_list[systemProfile]; 
          U_0_sys = UNorm_list[systemProfile].reshape(m,-1); 
          X_0_sys = XNorm_list[systemProfile].reshape(n,-1); 
          Y_0_sys = YNorm_list[systemProfile].reshape(p,-1)

          T_delaystart = 0.

          """ 3 & 4. simulate the 1st setpoint, to get the steady state output """
          LearningSetpoints = self._unitInfo[unit]['LearningSetpoints']
          window = int(Moving_Window_Width/Tss) # window width for DMDc
          if len(self._unitInfo[unit]['t_hist']) == 0: # if self-learning was never run before  # Initialize linear model
            # Initialize linear model
            x_sys_internal = np.zeros(n).reshape(n,-1) # x_sys type == <class 'numpy.ndarray'>
            t = -Tr_Update_sec*len(LearningSetpoints) # t = -7200 s
            t_idx = 0
          
            # Do the step-by-step simulation, from beginning to the first transient
            while t < -Tr_Update_sec*(len(LearningSetpoints)-1): # only the steady state value
              # Find the current r value
              
              r_value = copy.deepcopy(float(LearningSetpoints[t_idx]))
              # print("t_idx=", t_idx, "t=", t, "r=", r_value)
              # print(type(r_value))
              
              # No reference governor for the first setpoint value yet
              v_RG = copy.deepcopy(r_value)
              # print("v_RG:", type(v_RG))
              
              # fetch y
              y_sim_internal = np.dot(C_sys,x_sys_internal).reshape(p,-1)
              y_fetch = (y_sim_internal + Y_0_sys).reshape(p,)
              
              # fetch v and x
              v_fetch = np.asarray(v_RG).reshape(m,)
              x_fetch = (x_sys_internal + X_0_sys).reshape(n,)

              self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))  # input v
              self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))  # input v
              self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))  # state x
              self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))  # output y

              # step update x
              x_sys_internal = np.dot(A_sys,x_sys_internal)+np.dot(B_sys,v_RG-float(U_0_sys))
              # print("x_sys_internal:",type(x_sys_internal), x_sys_internal.shape, x_sys_internal)
              
              # time increment
              t = t + Tss
            # fetch the steady-state y variables
            v_0 = copy.deepcopy(v_fetch.reshape(m,-1))
            x_0 = copy.deepcopy(x_fetch.reshape(n,-1))
            y_0 = copy.deepcopy(y_fetch.reshape(p,-1))

            # store v_0, x_0 and y_0 into self._unitInfo
            self._unitInfo[unit]['v_0_sl'] = copy.deepcopy(v_0)
            self._unitInfo[unit]['x_0_sl'] = copy.deepcopy(x_0)
            self._unitInfo[unit]['y_0_sl'] = copy.deepcopy(y_0)

            t_idx += 1

            # check if steady-state y is within the [ymin, ymax]
            for i in range(len(y_0)):
              if y_0[i][0]>y_max[i]:
                exitMessage = """\n\tERROR:  Steady state output y_STEADY[{:d}] is {:.2f} HIGHER than y upper constraints. \n
                \tFYI:      Unit = {};
                \tFYI:  y_STEADY = {};
                \tFYI: y_maximum = {};
                \tFYI: y_minimum = {}.\n
                \tPlease modify the steady state setpoint in <LearningSetpoints>, Item #0.\n""".format(i, 
                y_0[i][0]-y_max[i], str(unit), 
                np.array2string(y_0.flatten(), formatter={'float_kind':lambda x: "%.4e" % x}),
                np.array2string(y_max, formatter={'float_kind':lambda x: "%.4e" % x}),
                np.array2string(y_min, formatter={'float_kind':lambda x: "%.4e" % x}),
                )
                print(exitMessage)
                sys.exit(exitMessage)
              elif y_0[i][0]<y_min[i]:
                exitMessage = """\n\tERROR:  Steady state output y_STEADY[{:d}] is {:.2f} LOWER than y lower constraints. \n
                \tFYI:      Unit = {};
                \tFYI: y_maximum = {};
                \tFYI: y_minimum = {};
                \tFYI:  y_STEADY = {}.\n
                \tPlease modify the steady state setpoint in <LearningSetpoints>, Item #0.\n""".format(i, 
                y_min[i]-y_0[i][0], str(unit), 
                np.array2string(y_max, formatter={'float_kind':lambda x: "%.4e" % x}),
                np.array2string(y_min, formatter={'float_kind':lambda x: "%.4e" % x}),
                np.array2string(y_0.flatten(), formatter={'float_kind':lambda x: "%.4e" % x}),
                )
                print(exitMessage)
                sys.exit(exitMessage)
            print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
            print("^^^ Steady State Summary Start ^^^")
            print("Unit =", str(unit), ", t =", t - Tss, "\nv_0 =\n", float(v_0), "\nx_0 = \n",x_0,"\ny_0 = \n",y_0)
            print("^^^^ Steady State Summary End ^^^^")
            print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
            # print("v_hist of ",str(unit), "=\n",len(self._unitInfo[unit]['v_hist']),self._unitInfo[unit]['v_hist'])
            # print(" y_hist of ",str(unit), "=\n",len(self._unitInfo[unit]['y_hist']),self._unitInfo[unit]['y_hist'])

            """ 5. Simulate the using the second r_ext value, to get the first guess of ABCD matrices """
            
            while t_idx < len(LearningSetpoints):
            
              # Do the step-by-step simulation, from beginning to the first transient
              while t < -Tr_Update_sec*(len(LearningSetpoints)-1-t_idx): # only the steady state value
                # Find the current r value
                
                r_value = copy.deepcopy(float(LearningSetpoints[t_idx]))
                # print("t_idx=", t_idx, "t=", t, "r=", r_value)
                # print(type(r_value))
                
                # No reference governor for the first setpoint value yet
                v_RG = copy.deepcopy(r_value)
                
                # fetch y
                y_sim_internal = np.dot(C_sys,x_sys_internal).reshape(p,-1)
                y_fetch = (y_sim_internal + Y_0_sys).reshape(p,)
                
                # fetch v and x
                v_fetch = np.asarray(v_RG).reshape(m,)
                x_fetch = (x_sys_internal + X_0_sys).reshape(n,)

                self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))  # input v
                self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))  # input v
                self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))  # state x
                self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))  # output y

                # step update x
                x_sys_internal = np.dot(A_sys,x_sys_internal)+np.dot(B_sys,v_RG-float(U_0_sys))
                # print("x_sys_internal:",type(x_sys_internal), x_sys_internal.shape, x_sys_internal)
                # time increment
                t = t + Tss
              # Collect data for DMDc
              t_window = np.asarray(self._unitInfo[unit]['t_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).reshape(1,-1)
              v_window = np.asarray(self._unitInfo[unit]['v_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).reshape(m,-1)
              x_window = np.asarray(self._unitInfo[unit]['x_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).T
              y_window = np.asarray(self._unitInfo[unit]['y_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).T  
              # print(t_window.shape) # (1, 180)
              # print(v_window.shape) # (1, 180)
              # print(x_window.shape) # (1, 180)
              # print(y_window.shape) # (2, 180)

              # Do the DMDc, and return ABCD matrices
              U1 = v_window[:,0:-1]-v_0; X1 = x_window[:, 0:-1]-x_0; X2 = x_window[:, 1:]-x_0; Y1 = y_window[:, 0:-1]-y_0
              Do_DMDc = False
              if abs(np.max(U1)-np.min(U1))>self._unitInfo[unit]['NeighborSetpointsThreshold']: # if transient found within this window
                if len(self._unitInfo[unit]['para_list'])==0: 
                  # if para_list is empty, do DMDc
                  Do_DMDc = True
                elif np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) > self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                  # if the nearest parameter is more than 1 MW apart, do DMDc
                  Do_DMDc = True

              if Do_DMDc:
                Ad_Dc, Bd_Dc, Cd_Dc= fun_DMDc(X1, X2, U1, Y1, -1)
                # Dd_Dc = np.zeros((p,m))
                # append the A,B,C,D matrices to an list
                self._unitInfo[unit]['A_list'].append(Ad_Dc); 
                self._unitInfo[unit]['B_list'].append(Bd_Dc); 
                self._unitInfo[unit]['C_list'].append(Cd_Dc); 
                self._unitInfo[unit]['para_list'].append(float(U1[:,-1]+v_0)); 
                self._unitInfo[unit]['eig_A_list'].append(np.max(np.linalg.eig(Ad_Dc)[0]))
                self._unitInfo[unit]['tTran_list'].append(t-Tr_Update_sec)
                print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                print("&&& DMDc summary Start &&&")
                print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                print("A_list=\n",self._unitInfo[unit]['A_list'])
                print("B_list=\n",self._unitInfo[unit]['B_list'])
                print("C_list=\n",self._unitInfo[unit]['C_list'])
                print("para_list=\n",self._unitInfo[unit]['para_list'])
                print("eig_A_list=\n",self._unitInfo[unit]['eig_A_list'])
                print("tTran_list=\n",self._unitInfo[unit]['tTran_list'])
                print("&&&& DMDc summary End &&&&")
                print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                # print(a)
              else:
                print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                print("&&& DMDc was not done. &&&")
                print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                if abs(np.max(U1)-np.min(U1))<=self._unitInfo[unit]['NeighborSetpointsThreshold']:
                  print("Reason: Transient is too small. v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                elif np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) <= self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                  print("Reason: New parameter is too close to existing parameter [{}].".format(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1]).argmin()))
                  print("New parameter =", v_window[:,-1], "Para_list =",self._unitInfo[unit]['para_list'])
                print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
              t_idx += 1
            
            # save the histories to _sl hist
            self._unitInfo[unit]['t_hist_sl']=copy.deepcopy(self._unitInfo[unit]['t_hist'])
            self._unitInfo[unit]['v_hist_sl']=copy.deepcopy(self._unitInfo[unit]['v_hist'])
            self._unitInfo[unit]['x_hist_sl']=copy.deepcopy(self._unitInfo[unit]['x_hist'])
            self._unitInfo[unit]['y_hist_sl']=copy.deepcopy(self._unitInfo[unit]['y_hist'])
            # empty the A_list, B_list, C_list, eig_A_list, para_list, tTran_list
            self._unitInfo[unit]['A_list_sl']=copy.deepcopy(self._unitInfo[unit]['A_list'])
            self._unitInfo[unit]['B_list_sl']=copy.deepcopy(self._unitInfo[unit]['B_list'])
            self._unitInfo[unit]['C_list_sl']=copy.deepcopy(self._unitInfo[unit]['C_list'])
            self._unitInfo[unit]['eig_A_list_sl']=copy.deepcopy(self._unitInfo[unit]['eig_A_list'])
            self._unitInfo[unit]['para_list_sl']=copy.deepcopy(self._unitInfo[unit]['para_list'])
            self._unitInfo[unit]['tTran_list_sl']=copy.deepcopy(self._unitInfo[unit]['tTran_list'])
            self._unitInfo[unit]['t_idx_sl']=copy.deepcopy(t_idx)

            # Get time for self-learning
            endTimeSL = copy.deepcopy(datetime.now())
            print('Haoyu t-debug, Time for {} Self_Learning is {}'.format(str(unit),endTimeSL-startTime))
          
          """ 6. Simulate from the third r_ext value using RG, and update the ABCD matrices as it goes """
          # Initialization of time, and retrieve of norminal values
          t = 0
          # store v_0, x_0 and y_0 into self._unitInfo
          v_0 = copy.deepcopy(self._unitInfo[unit]['v_0_sl'])
          x_0 = copy.deepcopy(self._unitInfo[unit]['x_0_sl'])
          y_0 = copy.deepcopy(self._unitInfo[unit]['y_0_sl'])
          t_idx = copy.deepcopy(self._unitInfo[unit]['t_idx_sl'])
          # MOAS steps Limit 
          g = int(Tr_Update_sec/Tss)+1 # numbers of steps to look forward, , type = <class 'int'>
          # Calculate s for Maximal Output Admissible Set (MOAS)
          s = [] # type == <class 'list'>
          for i in range(0,p):
            s.append(abs(y_max[i] - y_0[i]))
            s.append(abs(y_0[i] - y_min[i]))
          s = np.asarray(s).tolist()
          # print(s)

          for tracker in comp.get_tracking_vars():
            # loop through the resources in info (only one resource here - electricity)
            for res in info:
              if str(res) == "electricity":
                # Initiate the linear system
                if self._unitInfo[unit]['XInit']==[]:
                  x_sys_internal = np.zeros(n)
                else:
                  x_sys_internal = np.asarray(self._unitInfo[unit]['XInit'])-x_0[0]
                  # print("Step 6, x_sys_internal=",x_sys_internal)

                # loop through the time index (tidx) and time in "times"
                # t_idx = t_idx+1
                for tidx, time in enumerate(times):
                  # Copy the system state variable
                  x_KF = x_sys_internal
                  """ Get the r_value, original actuation value """
                  current = float(dispatch.get_activity(comp, tracker, res, times[tidx]))
                  # check if storage: power = (curr. MWh energy - prev. MWh energy)/interval Hrs

                  if comp.get_interaction().is_type('Storage') and tidx == 0:
                    init_level = comp.get_interaction().get_initial_level(meta)

                  if comp.get_interaction().is_type('Storage'):
                    # Initial_Level = float(self._unitInfo[unit]['Initial_Level'])
                    Initial_Level = float(init_level)
                    if tidx == 0: # for the first hour, use the initial level. charging yields to negative r_value
                      r_value = -(current - Initial_Level)/Tr_Update_hrs
                    else: # for the other hours
                      # r_value = -(current - float(dispatch.get_activity(comp, tracker, res, times[tidx-1])))/Tr_Update_hrs
                      r_value = -(current - Allowed_Level)/Tr_Update_hrs
                  else: # when not storage,
                    r_value = current # measured in MW

                  """ Find the correct profile according to r_value"""
                  profile_id = (np.abs(np.asarray(self._unitInfo[unit]['para_list']) - r_value)).argmin()
                  # print("t_idx=",t_idx, "t=",t)

                  # Retrive the correct A, B, C matrices
                  A_d = self._unitInfo[unit]['A_list'][profile_id]
                  B_d = self._unitInfo[unit]['B_list'][profile_id]
                  C_d = self._unitInfo[unit]['C_list'][profile_id]
                  D_d = np.zeros((p,m)) # all zero D matrix
                  
                  # Build the s, H and h for MOAS
                  
                  H_DMDc, h_DMDc = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g)  # H and h, type = <class 'numpy.ndarray'>
              
                  # first v_RG: consider the step "0" - step "g"
                  v_RG = fun_RG_SISO(0, x_KF, r_value-v_0, H_DMDc, h_DMDc, p) # v_RG: type == <class 'numpy.ndarray'>

                  # find the profile with max eigenvalue of A
                  max_eigA_id = np.asarray(self._unitInfo[unit]['eig_A_list']).argmax()
                  A_m = self._unitInfo[unit]['A_list'][max_eigA_id]
                  B_m = self._unitInfo[unit]['B_list'][max_eigA_id]
                  C_m = self._unitInfo[unit]['C_list'][max_eigA_id]
                  D_m = np.zeros((p,m)) # all zero D matrix

                  """ 2nd adjustment """
                  # MOAS for the steps "g+1" - step "2g"
                  Hm, hm = fun_MOAS_noinf(A_m, B_m, C_m, D_m, s, g)
                  # Calculate the max/min for v, ensuring the hm-Hxm*x(g+1) always positive for the next g steps.
                  v_max, v_min = fun_2nd_gstep_calc(x_KF, Hm, hm, A_m, B_m, g)

                  if v_RG < v_min:
                    v_RG = v_min
                  elif v_RG > v_max:
                    v_RG = v_max

                  # # Pretend there is no FARM intervention
                  # v_RG = np.asarray(r_value-r_0).flatten()

                  v_RG = float(v_RG)+float(v_0) # absolute value of electrical power (MW)
                  print("\n**************************", "\n**** RG summary Start ****","\nUnit = ", str(unit),", t = ", t, "\nr = ", r_value, "\nProfile Selected = ", profile_id, "\nv_RG = ", v_RG, "\n***** RG summary End *****","\n**************************\n")
                  print("HaoyuRGSummary,", str(unit), ",t,", t, ",Profile,", profile_id, 
                    ",r0,", r_value, ",v0,", v_RG, ",v_min,", float(v_min)+float(v_0), ",v_max,", float(v_max)+float(v_0) )
              
                  # Update x_sys_internal, and keep record in v_hist and yp_hist within this hour
                  for i in range(int(Tr_Update_sec/Tss)):
                    # fetch y
                    y_sim_internal = np.dot(C_sys,x_sys_internal).reshape(p,-1)
                    y_fetch = (y_sim_internal + Y_0_sys).reshape(p,) 
                    
                    # fetch v and x
                    v_fetch = np.asarray(v_RG).reshape(m,)
                    x_fetch = (x_sys_internal + X_0_sys).reshape(n,)
                    
                    self._unitInfo[unit]['t_hist'].append(t)  # input v
                    self._unitInfo[unit]['v_hist'].append(v_fetch)  # input v
                    self._unitInfo[unit]['x_hist'].append(x_fetch)  # state x
                    self._unitInfo[unit]['y_hist'].append(y_fetch)  # output y

                    # step update x
                    x_sys_internal = np.dot(A_sys,x_sys_internal)+np.dot(B_sys,v_RG-float(U_0_sys))
                    t = t + Tss

                  # Convert to V1:

                  # if storage
                  if comp.get_interaction().is_type('Storage'):
                    if tidx == 0: # for the first hour, use the initial level
                      Allowed_Level = Initial_Level - v_RG*Tr_Update_hrs # Allowed_Level: predicted level due to v_value
                    else: # for the other hours
                      Allowed_Level = Allowed_Level - v_RG*Tr_Update_hrs
                    V1 = Allowed_Level
                  else: # when not storage,
                    V1 = v_RG

                  # print("Haoyu Debug, unit=",str(unit),", t=",time, ", curr= %.8g, V1= %.8g, delta=%.8g" %(current, V1, (V1-current)))

                  # Collect data for DMDc
                  t_window = np.asarray(self._unitInfo[unit]['t_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).reshape(1,-1)
                  v_window = np.asarray(self._unitInfo[unit]['v_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).reshape(m,-1)
                  x_window = np.asarray(self._unitInfo[unit]['x_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).T
                  y_window = np.asarray(self._unitInfo[unit]['y_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).T  

                  # Do the DMDc, and return ABCD matrices
                  U1 = v_window[:,0:-1]-v_0; X1 = x_window[:, 0:-1]-x_0; X2 = x_window[:, 1:]-x_0; Y1 = y_window[:, 0:-1]-y_0
                  Do_DMDc = False
                  if abs(np.max(U1)-np.min(U1))>self._unitInfo[unit]['NeighborSetpointsThreshold'] and t_idx!=len(LearningSetpoints): # if there is transient, DMDc can be done: # if transient found within this window
                    if np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) > self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                      # if the nearest parameter is more than 1 MW apart, do DMDc
                      Do_DMDc = True

                  if Do_DMDc:  # print(U1.shape)
                    Ad_Dc, Bd_Dc, Cd_Dc= fun_DMDc(X1, X2, U1, Y1, -1)
                    # Dd_Dc = np.zeros((p,m))
                    
                    # append the A,B,C,D matrices to an list
                    self._unitInfo[unit]['A_list'].append(Ad_Dc); 
                    self._unitInfo[unit]['B_list'].append(Bd_Dc); 
                    self._unitInfo[unit]['C_list'].append(Cd_Dc); 
                    self._unitInfo[unit]['para_list'].append(float(U1[:,-1]+v_0)); 
                    self._unitInfo[unit]['eig_A_list'].append(np.max(np.linalg.eig(Ad_Dc)[0]))
                    self._unitInfo[unit]['tTran_list'].append(t-Tr_Update_sec)
                    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    print("&&& DMDc summary Start &&&")
                    print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                    print("A_list=\n",self._unitInfo[unit]['A_list'])
                    print("B_list=\n",self._unitInfo[unit]['B_list'])
                    print("C_list=\n",self._unitInfo[unit]['C_list'])
                    print("para_list=\n",self._unitInfo[unit]['para_list'])
                    print("eig_A_list=\n",self._unitInfo[unit]['eig_A_list'])
                    print("tTran_list=\n",self._unitInfo[unit]['tTran_list'])
                    print("&&&& DMDc summary End &&&&")
                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                    # print(a)
                  else:
                    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    print("&&& DMDc was not done. &&&")
                    print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                    if abs(np.max(U1)-np.min(U1))<=self._unitInfo[unit]['NeighborSetpointsThreshold'] or t_idx==len(LearningSetpoints):
                      if abs(np.max(U1)-np.min(U1))<=self._unitInfo[unit]['NeighborSetpointsThreshold']:
                        print("Reason: Transient is too small. v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                      if t_idx==len(LearningSetpoints):
                        print("Reason: System is initialized at t =",t-Tr_Update_sec)
                    elif np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) <= self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                      print("Reason: New parameter is too close to existing parameter [{}].".format(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1]).argmin()))
                      print("New parameter =", v_window[:,-1], "Para_list =",self._unitInfo[unit]['para_list'])
                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                  t_idx = t_idx+1

                  # Write up any violation to the errs:
                  if abs(current - V1) > self._tolerance*max(abs(current),abs(V1)):
                    # violation
                    errs.append({'msg': f'Reference Governor Violation',
                                'limit': V1,
                                'limit_type': 'lower' if (current < V1) else 'upper',
                                'component': comp,
                                'resource': res,
                                'time': time,
                                'time_index': tidx,
                                })

          # Get time for dispatching
          endTimeDispatch = copy.deepcopy(datetime.now())
          print('Haoyu t-debug, Time for this {} Dispatch is {}'.format(str(unit),endTimeDispatch-startTime))

    if errs == []: # if no validation error:
      print(" ")
      print("*********************************************************************")
      print("*** Haoyu Debug, Validation Success, Print for offline processing ***")
      print("*********************************************************************")
      print(" ")

      for unit in self._unitInfo:
        t_hist = np.array(self._unitInfo[unit]['t_hist']).T # t_hist.shape = (5040,)
        v_hist = np.array(self._unitInfo[unit]['v_hist']).T # v_hist.shape = (1, 5040)
        y_hist = np.array(self._unitInfo[unit]['y_hist']).T # y_hist.shape = (2, 5040)
        # print(str(unit), t_hist.shape, v_hist.shape, y_hist.shape)

        for i in range(len(t_hist)):
          print("HaoyuDispatchSummary, " + str(unit) + 
          ", t, " + str(t_hist[i]) + ", vp, " + str(v_hist[0][i]) +
          ", y, " + ', '.join([str(num) for num in y_hist[:,i]]) + 
          ", ymin, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Min']]) +
          ", ymax, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Max']]))

    return errs

class FARM_SISO(Validator):
  """
    A FARM SISO Validator for dispatch decisions.(Dirty Implementation)
    Accepts parameterized A,B,C,D matrices from external XML file and use the first set within constraints 
    as physics model, and validate
    the dispatch power (BOP, unit=MW)

    Haoyu Wang, ANL-NSE, March 28, 2022
  """
  # ---------------------------------------------
  # INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = Validator.get_input_specs()
    specs.name = 'FARM_SISO'
    specs.description = r"""Feasible Actuator Range Modifier, which uses a single-input-single-output
        reference governor validator to adjust the power setpoints issued to the components at the
        beginning of each dispatch interval (usually an hour), to ensure the the operational constraints
        were not violated during the following dispatch interval. This version uses a Functional Mock-up 
        Unit (FMU) as a downstream high-fidelity model."""

    component = InputData.parameterInputFactory('ComponentForFARM', ordered=False, baseNode=None,
        descr=r"""The component whose power setpoint will be adjusted by FARM. The user need
        to provide the statespace matrices and operational constraints concerning this component,
        and optionally provide the initial states.""")
    component.addParam('name',param_type=InputTypes.StringType, required=True,
        descr=r"""The name by which this component should be referred within HERON. It should match
        the component's name in \xmlNode{Components}.""")

    component.addSub(InputData.parameterInputFactory('FMUFile',contentType=InputTypes.StringType,
        descr=r"""The path to the FMU file of this component. Either absolute path
        or path relative to HERON root (starts with %HERON%/)will work. The matrices file can be generated from
        RAVEN DMDc or other sources."""))
    component.addSub(InputData.parameterInputFactory('StateVariableSelectionXML',contentType=InputTypes.StringType,
        descr=r"""The path to the XML file generated by state variable selection algorithm. Either absolute path
        or path relative to HERON / FARM root (starts with %HERON% / %FARM%) will work. NOTE: Once this file was provided,
        FARM will not read the following nodes: \xmlNode{FMUSimulationStep}, \xmlNode{InputVarNames}, 
        \xmlNode{StateVarNames}, \xmlNode{OutputVarNames}, \xmlNode{LearningSetpoints}, """))
    
    component.addSub(InputData.parameterInputFactory('FMUSimulationStep',contentType=InputTypes.FloatType,
        descr=r"""The step length of FMU simulation. It should be a floating number or an integer."""))
    component.addSub(InputData.parameterInputFactory('InputVarNames',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of FMU input variables. It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('StateVarNames',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of FMU state variables. It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('OutputVarNames',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of FMU output variables. It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('LearningSetpoints',contentType=InputTypes.InterpretedListType,
        descr=r"""The learning setpoints are used to find the nominal value and first sets of ABCD matrices. 
        It should be a list of two or more floating numbers or integers separated by comma."""))
    
    component.addSub(InputData.parameterInputFactory('NeighborSetpointsThreshold',contentType=InputTypes.FloatType,
        descr=r"""The Neighbor Setpoints Threshold is used to create a new ABCD matrices profile. 
        If a new setpoint falls within an existing setpoint plus or minus this threshold, no new profile will be created.
        It should be a floating number."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsUpper',contentType=InputTypes.InterpretedListType,
        descr=r"""The upper bounds for the output variables of this component. It should be a list of
        floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsLower',contentType=InputTypes.InterpretedListType,
        descr=r"""The lower bounds for the output variables of this component. It should be a list of
        floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('RollingWindowWidth',contentType=InputTypes.IntegerType,
        descr=r"""The moving window duration for DMDc, with the unit of seconds. It should be an integer."""))

    specs.addSub(component)

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'
    self._tolerance = 1.003e-3

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    self._unitInfo = {}
    for component in inputs.subparts:
      name = component.parameterValues['name']

      NeighborSetpointsThreshold = 1.0 # default threshold 1.0 MW
      svsXMLFile = None
      FMUSimulationStep = None
      InputVarNames = None
      StateVarNames = None
      OutputVarNames = None
      LearningSetpoints = None
      for farmEntry in component.subparts:
        if farmEntry.getName() == "FMUFile":
          fmuFile = farmEntry.value
          if fmuFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            fmuFile = os.path.abspath(fmuFile.replace('%HERON%', heron_path))
          elif fmuFile.startswith('%FARM%'):
            # magic word for "relative to HERON root"
            farm_path = get_farm_loc()
            fmuFile = os.path.abspath(fmuFile.replace('%FARM%', farm_path))
        if farmEntry.getName() == "FMUSimulationStep":
          FMUSimulationStep = farmEntry.value
        if farmEntry.getName() == "setpointsShiftStep":
          setpointsShiftStep = farmEntry.value  
        if farmEntry.getName() == "StateVariableSelectionXML":
          svsXMLFile = farmEntry.value
          if svsXMLFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            svsXMLFile = os.path.abspath(svsXMLFile.replace('%HERON%', heron_path))
          elif svsXMLFile.startswith('%FARM%'):
            # magic word for "relative to HERON root"
            farm_path = get_farm_loc()
            svsXMLFile = os.path.abspath(svsXMLFile.replace('%FARM%', farm_path))
        if farmEntry.getName() == "InputVarNames":
          InputVarNames = farmEntry.value
        if farmEntry.getName() == "StateVarNames":
          StateVarNames = farmEntry.value
        if farmEntry.getName() == "OutputVarNames":
          OutputVarNames = farmEntry.value
        if farmEntry.getName() == "LearningSetpoints":
          LearningSetpoints = farmEntry.value
          if len(LearningSetpoints) < 2:
            sys.exit('\nERROR: <LearningSetpoints> XML node needs to contain 2 or more floating or integer numbers.\n')   
          elif min(LearningSetpoints)==max(LearningSetpoints):
            exitMessage = """ERROR:  No transient found in <LearningSetpoints>. \n\tPlease modify the values in <LearningSetpoints>.\n"""
            sys.exit(exitMessage)        
        if farmEntry.getName() == "NeighborSetpointsThreshold":
          NeighborSetpointsThreshold = farmEntry.value
        if farmEntry.getName() == "RollingWindowWidth":
          RollingWindowWidth = farmEntry.value
        if farmEntry.getName() == "OpConstraintsUpper":
          UpperBound = farmEntry.value
        if farmEntry.getName() == "OpConstraintsLower":
          LowerBound = farmEntry.value

      self._unitInfo.update(
        {name:{
          'FMUFile':fmuFile,
          'FMUSimulationStep':FMUSimulationStep,
          'StateVariableSelectionXML':svsXMLFile,
          'InputVarNames':InputVarNames,
          'StateVarNames':StateVarNames,
          'OutputVarNames':OutputVarNames,
          'LearningSetpoints':LearningSetpoints,
          'NeighborSetpointsThreshold':NeighborSetpointsThreshold,
          'RollingWindowWidth':RollingWindowWidth,
          'Targets_Max':UpperBound,
          'Targets_Min':LowerBound,
          't_hist_sl':[],
          'v_hist_sl':[],
          'x_hist_sl':[],
          'y_hist_sl':[],
          'v_0_sl': None,
          'x_0_sl': None,
          'y_0_sl': None,
          't_idx_sl': None,
          'A_list_sl':[],
          'B_list_sl':[],
          'C_list_sl':[],
          'eig_A_list_sl':[],
          'para_list_sl':[],
          'tTran_list_sl':[],
          't_hist':[],
          'v_hist':[],
          'x_hist':[],
          'y_hist':[],
          'A_list':[],
          'B_list':[],
          'C_list':[],
          'eig_A_list':[],
          'para_list':[],
          'tTran_list':[]}})
    print('\n',self._unitInfo,'\n')

  # ---------------------------------------------
  # API
  def validate(self, components, dispatch, times, meta):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, extra information pertaining to validation
      @ Out, errs, list, information about validation failures
    """
    # errs will be returned to dispatcher. errs contains all the validation errors calculated in below
    errs = [] # TODO best format for this?

    """ get time interval"""
    Tr_Update_hrs = float(times[1]-times[0])
    Tr_Update_sec = Tr_Update_hrs*3600.

    # loop through the <Component> items in HERON
    for comp, info in dispatch._resources.items():
      # e.g. comp= <HERON Component "SES""> <HERON Component "SES"">
      # loop through the items defined in the __init__ function
      for unit in self._unitInfo:
        # e.g. CompInfo, unit= SES
        # Identify the profile as defined in the __init__ function
        if str(unit) not in str(comp):
          # If the "unit" and "comp" do not match, go to the next "unit" in loop
          continue
        else: # when the str(unit) is in the str(comp) (e.g. "SES" in "<HERON Component "SES"">")
          # get start time
          startTime = copy.deepcopy(datetime.now())
          """ 1. Constraints information, and Moving window width """
          # Constraints
          y_min = np.asarray(self._unitInfo[unit]['Targets_Min'])
          y_max = np.asarray(self._unitInfo[unit]['Targets_Max'])

          # The width of moving window (seconds, centered at transient edge, for moving window DMDc)
          Moving_Window_Width = self._unitInfo[unit]['RollingWindowWidth']; #Tr_Update

          # Load the self learning hists, if any
          self._unitInfo[unit]['t_hist']=copy.deepcopy(self._unitInfo[unit]['t_hist_sl'])
          self._unitInfo[unit]['v_hist']=copy.deepcopy(self._unitInfo[unit]['v_hist_sl'])
          self._unitInfo[unit]['x_hist']=copy.deepcopy(self._unitInfo[unit]['x_hist_sl'])
          self._unitInfo[unit]['y_hist']=copy.deepcopy(self._unitInfo[unit]['y_hist_sl'])
          # empty the A_list, B_list, C_list, eig_A_list, para_list, tTran_list
          self._unitInfo[unit]['A_list']=copy.deepcopy(self._unitInfo[unit]['A_list_sl'])
          self._unitInfo[unit]['B_list']=copy.deepcopy(self._unitInfo[unit]['B_list_sl'])
          self._unitInfo[unit]['C_list']=copy.deepcopy(self._unitInfo[unit]['C_list_sl'])
          self._unitInfo[unit]['eig_A_list']=copy.deepcopy(self._unitInfo[unit]['eig_A_list_sl'])
          self._unitInfo[unit]['para_list']=copy.deepcopy(self._unitInfo[unit]['para_list_sl'])
          self._unitInfo[unit]['tTran_list']=copy.deepcopy(self._unitInfo[unit]['tTran_list_sl'])
           
          
          """ 2. Read FMU file to specify the physical model"""
          fmu_filename = self._unitInfo[unit]['FMUFile']

          # if state variable selection is not provided
          if self._unitInfo[unit]['StateVariableSelectionXML'] is None:
            print("******, Will do Self learning ******")
            inputVarNames = self._unitInfo[unit]['InputVarNames']
            stateVarNames = self._unitInfo[unit]['StateVarNames']
            outputVarNames = self._unitInfo[unit]['OutputVarNames']
            Tss = self._unitInfo[unit]['FMUSimulationStep']
            # Dimensions of input (m), states (n) and output (p)
            m=len(inputVarNames); n=len(stateVarNames); p=len(outputVarNames)
          else: # if state variable selection is provided
            print("******, Will not do Self learning, reading everything from state variable selection XML ******")
            # read xml here, and assign values to the three VarNames and Tss above
            """ Read State Space XML file (generated by Raven parameterized DMDc) """
            MatrixFile = self._unitInfo[unit]['StateVariableSelectionXML']
            Tss, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array, actuatorList, stateList, outputList= read_parameterized_XML(MatrixFile)
            # convert para_array to para_list (single parameter)
            para_list = [float(item) for item in para_array]
            eig_A_list = eig_A_array.tolist()

            # print("Tss", Tss, type(Tss)) # <class 'numpy.float64'>
            # print("n", n, type(n)) # <class 'int'>
            # print("m", m, type(m)) # <class 'int'>
            # print("p", p, type(p)) # <class 'int'>
            # print("para_array", para_array, type(para_array)) # <class 'numpy.ndarray'>
            # print("para_list", para_list, type(para_list)) # 
            
            # print("UNorm_list",UNorm_list, type(UNorm_list)) # <class 'list'>
            # print("XNorm_list",XNorm_list, type(XNorm_list)) # <class 'list'>
            # print("YNorm_list",YNorm_list, type(YNorm_list)) # <class 'list'>
            # # print("A_list",A_list)
            # # print("B_list",B_list)
            # # print("C_list",C_list)
            # print("actuatorList",actuatorList, type(actuatorList)) # <class 'list'>
            # print("stateList",stateList, type(stateList)) # <class 'list'>
            # print("outputList",outputList, type(outputList)) # <class 'list'>
            # print(a)
            inputVarNames = actuatorList
            stateVarNames = stateList
            outputVarNames = outputList

          # read the model description
          model_description = read_model_description(fmu_filename)
          
          # collect the value references
          vrs = {}
          for variable in model_description.modelVariables:
              vrs[variable.name] = variable.valueReference
          
          # get the value references for the variables we want to get/set
          # Input Power Setpoint (W)
          vr_input = [vrs[item] for item in inputVarNames]
          # State variables and dimension
          vr_state = [vrs[item] for item in stateVarNames]
          # Outputs: Power Generated (W), Turbine Pressure (Pa)
          vr_output = [vrs[item] for item in outputVarNames]
          
          # extract the FMU
          unzipdir = extract(fmu_filename)
          fmu = FMU2Slave(guid=model_description.guid,
                          unzipDirectory=unzipdir,
                          modelIdentifier=model_description.coSimulation.modelIdentifier,
                          instanceName='instance1')
          
          # Initialize FMU
          T_delaystart = 0.
          fmu.instantiate()
          fmu.setupExperiment(startTime=T_delaystart)
          fmu.enterInitializationMode()
          fmu.exitInitializationMode()

          # print(a)
          window = int(Moving_Window_Width/Tss) # window width for DMDc

          if self._unitInfo[unit]['StateVariableSelectionXML'] is None:
            print("******, Self learning Starts ******")            

            """ 3 & 4. simulate the 1st setpoint, to get the steady state output """
            LearningSetpoints = self._unitInfo[unit]['LearningSetpoints']
            
            if len(self._unitInfo[unit]['t_hist']) == 0: # if self-learning was never run before
              t = -Tr_Update_sec*len(LearningSetpoints) # t = -7200 s
              t_idx = 0
            
              # Do the step-by-step simulation, from beginning to the first transient
              while t < -Tr_Update_sec*(len(LearningSetpoints)-1): # only the steady state value
                # Find the current r value
                
                r_value = copy.deepcopy(float(LearningSetpoints[t_idx]))
                # print("t_idx=", t_idx, "t=", t, "r=", r_value)
                # print(type(r_value))
                
                # No reference governor for the first setpoint value yet
                v_RG = copy.deepcopy(r_value)
                # print("v_RG:", type(v_RG))
                
                # fetch y
                y_fetch = np.asarray(fmu.getReal(vr_output))

                # fetch v and x
                v_fetch = np.asarray(v_RG).reshape(m,)
                x_fetch = np.asarray(fmu.getReal(vr_state))

                # set the input. The input / state / output in FMU are real-world values
                fmu.setReal(vr_input, [v_RG])
                # perform one step
                fmu.doStep(currentCommunicationPoint=t, communicationStepSize=Tss)

                self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))  # input v
                self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))  # input v
                self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))  # state x
                self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))  # output y

                # time increment
                t = t + Tss
              # fetch the steady-state y variables
              v_0 = copy.deepcopy(v_fetch.reshape(m,-1))
              x_0 = copy.deepcopy(x_fetch.reshape(n,-1))
              y_0 = copy.deepcopy(y_fetch.reshape(p,-1))

              # store v_0, x_0 and y_0 into self._unitInfo
              self._unitInfo[unit]['v_0_sl'] = copy.deepcopy(v_0)
              self._unitInfo[unit]['x_0_sl'] = copy.deepcopy(x_0)
              self._unitInfo[unit]['y_0_sl'] = copy.deepcopy(y_0)

              t_idx += 1
          
              # check if steady-state y is within the [ymin, ymax]
              for i in range(len(y_0)):
                if y_0[i][0]>y_max[i]:
                  exitMessage = """\n\tERROR:  Steady state output y_STEADY[{:d}] is {:.2f} HIGHER than y upper constraints. \n
                  \tFYI:      Unit = {};
                  \tFYI:  y_STEADY = {};
                  \tFYI: y_maximum = {};
                  \tFYI: y_minimum = {}.\n
                  \tPlease modify the steady state setpoint in <LearningSetpoints>, Item #0.\n""".format(i, 
                  y_0[i][0]-y_max[i], str(unit), 
                  np.array2string(y_0.flatten(), formatter={'float_kind':lambda x: "%.4e" % x}),
                  np.array2string(y_max, formatter={'float_kind':lambda x: "%.4e" % x}),
                  np.array2string(y_min, formatter={'float_kind':lambda x: "%.4e" % x}),
                  )
                  print(exitMessage)
                  sys.exit(exitMessage)
                elif y_0[i][0]<y_min[i]:
                  exitMessage = """\n\tERROR:  Steady state output y_STEADY[{:d}] is {:.2f} LOWER than y lower constraints. \n
                  \tFYI:      Unit = {};
                  \tFYI: y_maximum = {};
                  \tFYI: y_minimum = {};
                  \tFYI:  y_STEADY = {}.\n
                  \tPlease modify the steady state setpoint in <LearningSetpoints>, Item #0.\n""".format(i, 
                  y_min[i]-y_0[i][0], str(unit), 
                  np.array2string(y_max, formatter={'float_kind':lambda x: "%.4e" % x}),
                  np.array2string(y_min, formatter={'float_kind':lambda x: "%.4e" % x}),
                  np.array2string(y_0.flatten(), formatter={'float_kind':lambda x: "%.4e" % x}),
                  )
                  print(exitMessage)
                  sys.exit(exitMessage)
              print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
              print("^^^ Steady State Summary Start ^^^")
              print("Unit =", str(unit), ", t =", t - Tss, "\nv_0 =\n", float(v_0), "\nx_0 = \n",x_0,"\ny_0 = \n",y_0)
              print("^^^^ Steady State Summary End ^^^^")
              print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
              # print("v_hist of ",str(unit), "=\n",len(self._unitInfo[unit]['v_hist']),self._unitInfo[unit]['v_hist'])
              # print(" y_hist of ",str(unit), "=\n",len(self._unitInfo[unit]['y_hist']),self._unitInfo[unit]['y_hist'])

              """ 5. Simulate the using the second r_ext value, to get the first guess of ABCD matrices """
              
              while t_idx < len(LearningSetpoints):

                # Do the step-by-step simulation, from beginning to the first transient
                while t < -Tr_Update_sec*(len(LearningSetpoints)-1-t_idx): # only the steady state value
                  # Find the current r value
                  
                  r_value = copy.deepcopy(float(LearningSetpoints[t_idx]))
                  # print("t_idx=", t_idx, "t=", t, "r=", r_value)
                  # print(type(r_value))
                  
                  # No reference governor for the first setpoint value yet
                  v_RG = copy.deepcopy(r_value)
                  
                  # fetch y
                  y_fetch = np.asarray(fmu.getReal(vr_output))

                  # fetch v and x
                  v_fetch = np.asarray(v_RG).reshape(m,)
                  x_fetch = np.asarray(fmu.getReal(vr_state))

                  # set the input. The input / state / output in FMU are real-world values
                  fmu.setReal(vr_input, [v_RG])
                  # perform one step
                  fmu.doStep(currentCommunicationPoint=t, communicationStepSize=Tss)

                  self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))  # input v
                  self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))  # input v
                  self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))  # state x
                  self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))  # output y

                  # time increment
                  t = t + Tss

                # Collect data for DMDc
                t_window = np.asarray(self._unitInfo[unit]['t_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).reshape(1,-1)
                v_window = np.asarray(self._unitInfo[unit]['v_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).reshape(m,-1)
                x_window = np.asarray(self._unitInfo[unit]['x_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).T
                y_window = np.asarray(self._unitInfo[unit]['y_hist'][(t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2)):(t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2))]).T  
                # print(t_window.shape) # (1, 180)
                # print(v_window.shape) # (1, 180)
                # print(x_window.shape) # (1, 180)
                # print(y_window.shape) # (2, 180)

                # Do the DMDc, and return ABCD matrices
                U1 = v_window[:,0:-1]-v_0; X1 = x_window[:, 0:-1]-x_0; X2 = x_window[:, 1:]-x_0; Y1 = y_window[:, 0:-1]-y_0
                Do_DMDc = False
                if abs(np.max(U1)-np.min(U1))>self._unitInfo[unit]['NeighborSetpointsThreshold']: # if transient found within this window
                  if len(self._unitInfo[unit]['para_list'])==0: 
                    # if para_list is empty, do DMDc
                    Do_DMDc = True
                  elif np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) > self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                    # if the nearest parameter is more than 1 MW apart, do DMDc
                    Do_DMDc = True

                if Do_DMDc:
                  Ad_Dc, Bd_Dc, Cd_Dc= fun_DMDc(X1, X2, U1, Y1, -1)
                  # Dd_Dc = np.zeros((p,m))
                  # append the A,B,C,D matrices to an list
                  self._unitInfo[unit]['A_list'].append(Ad_Dc); 
                  self._unitInfo[unit]['B_list'].append(Bd_Dc); 
                  self._unitInfo[unit]['C_list'].append(Cd_Dc); 
                  self._unitInfo[unit]['para_list'].append(float(U1[:,-1]+v_0)); 
                  self._unitInfo[unit]['eig_A_list'].append(np.max(np.linalg.eig(Ad_Dc)[0]))
                  self._unitInfo[unit]['tTran_list'].append(t-Tr_Update_sec)
                  print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                  print("&&& DMDc summary Start &&&")
                  print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                  print("A_list=\n",self._unitInfo[unit]['A_list'])
                  print("B_list=\n",self._unitInfo[unit]['B_list'])
                  print("C_list=\n",self._unitInfo[unit]['C_list'])
                  print("para_list=\n",self._unitInfo[unit]['para_list'])
                  print("eig_A_list=\n",self._unitInfo[unit]['eig_A_list'])
                  print("tTran_list=\n",self._unitInfo[unit]['tTran_list'])
                  print("&&&& DMDc summary End &&&&")
                  print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                  # print(a)
                else:
                  print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                  print("&&& DMDc was not done. &&&")
                  print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                  if abs(np.max(U1)-np.min(U1))<=self._unitInfo[unit]['NeighborSetpointsThreshold']:
                    print("Reason: Transient is too small. v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                  elif np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) <= self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                    print("Reason: New parameter is too close to existing parameter [{}].".format(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1]).argmin()))
                    print("New parameter =", v_window[:,-1], "Para_list =",self._unitInfo[unit]['para_list'])
                  print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                t_idx += 1

              # save the histories to _sl hist
              self._unitInfo[unit]['t_hist_sl']=copy.deepcopy(self._unitInfo[unit]['t_hist'])
              self._unitInfo[unit]['v_hist_sl']=copy.deepcopy(self._unitInfo[unit]['v_hist'])
              self._unitInfo[unit]['x_hist_sl']=copy.deepcopy(self._unitInfo[unit]['x_hist'])
              self._unitInfo[unit]['y_hist_sl']=copy.deepcopy(self._unitInfo[unit]['y_hist'])
              # empty the A_list, B_list, C_list, eig_A_list, para_list, tTran_list
              self._unitInfo[unit]['A_list_sl']=copy.deepcopy(self._unitInfo[unit]['A_list'])
              self._unitInfo[unit]['B_list_sl']=copy.deepcopy(self._unitInfo[unit]['B_list'])
              self._unitInfo[unit]['C_list_sl']=copy.deepcopy(self._unitInfo[unit]['C_list'])
              self._unitInfo[unit]['eig_A_list_sl']=copy.deepcopy(self._unitInfo[unit]['eig_A_list'])
              self._unitInfo[unit]['para_list_sl']=copy.deepcopy(self._unitInfo[unit]['para_list'])
              self._unitInfo[unit]['tTran_list_sl']=copy.deepcopy(self._unitInfo[unit]['tTran_list'])
              self._unitInfo[unit]['t_idx_sl']=copy.deepcopy(t_idx)

              # Get time for self-learning
              endTimeSL = copy.deepcopy(datetime.now())
              print('Haoyu t-debug, Time for {} Self_Learning is {}'.format(str(unit),endTimeSL-startTime))
          
          else: # if state variable selection XML is provided
            # 1. find proper nominal values where y_0 is within the bound
            for i_para in range(len(para_list)):
              # examine the y monimal values, item # i_para
              y_0_temp = YNorm_list[i_para]
              # examine if every element is within the constrant
              inConstraint = True
              for i_output in range(p):
                if y_0_temp[i_output] <= y_max[i_output] and y_0_temp[i_output] >= y_min[i_output]:
                  inConstraint = inConstraint and True
                else:
                  inConstraint = inConstraint and False
              
              # check the result
              if inConstraint:
                v_0 = copy.deepcopy(UNorm_list[i_para].reshape(m,-1))
                x_0 = copy.deepcopy(XNorm_list[i_para].reshape(n,-1))
                y_0 = copy.deepcopy(YNorm_list[i_para].reshape(p,-1))

                self._unitInfo[unit]['v_0_sl'] = copy.deepcopy(v_0)
                self._unitInfo[unit]['x_0_sl'] = copy.deepcopy(x_0)
                self._unitInfo[unit]['y_0_sl'] = copy.deepcopy(y_0)
                break
            
            if not inConstraint:
              exitMessage = """\n\tERROR:  No proper steady state output is provided in the state variable selection XML file. \n"""

            # 2. save the A_list, B_list, C_list, para_list
            self._unitInfo[unit]['A_list'] = copy.deepcopy(A_list)
            self._unitInfo[unit]['B_list'] = copy.deepcopy(B_list)
            self._unitInfo[unit]['C_list'] = copy.deepcopy(C_list)
            self._unitInfo[unit]['para_list'] = copy.deepcopy(para_list)
            self._unitInfo[unit]['eig_A_list'] = copy.deepcopy(eig_A_list)
            self._unitInfo[unit]['t_idx_sl'] = 0
      
          
          """ 6. Simulate from the third r_ext value using RG, and update the ABCD matrices as it goes """
          # Initialization of time, and retrieve of norminal values
          t = 0
          # store v_0, x_0 and y_0 into self._unitInfo
          v_0 = copy.deepcopy(self._unitInfo[unit]['v_0_sl'])
          x_0 = copy.deepcopy(self._unitInfo[unit]['x_0_sl'])
          y_0 = copy.deepcopy(self._unitInfo[unit]['y_0_sl'])
          t_idx = copy.deepcopy(self._unitInfo[unit]['t_idx_sl'])
          # MOAS steps Limit 
          g = int(Tr_Update_sec/Tss)+1 # numbers of steps to look forward, , type = <class 'int'>
          # Calculate s for Maximal Output Admissible Set (MOAS)
          s = [] # type == <class 'list'>
          for i in range(0,p):
            s.append(abs(y_max[i] - y_0[i]))
            s.append(abs(y_0[i] - y_min[i]))
          s = np.asarray(s).tolist()
          # print(s)

          for tracker in comp.get_tracking_vars():
            # loop through the resources in info (only one resource here - electricity)
            for res in info:
              if str(res) == "electricity":
                # Initialize FMU
                fmu.instantiate()
                fmu.setupExperiment(startTime=T_delaystart)
                fmu.enterInitializationMode()
                fmu.exitInitializationMode()
                
                # loop through the time index (tidx) and time in "times"
                # t_idx = t_idx+1
                for tidx, time in enumerate(times):
                  # Copy the system state variable
                  x_KF = np.asarray(fmu.getReal(vr_state))-x_0.reshape(n,)
                  # print("time=",time,", x_KF=",x_KF)
                  # print("x_0 reshape=",x_0.reshape(n,))
                  """ Get the r_value, original actuation value """
                  current = float(dispatch.get_activity(comp, tracker, res, times[tidx]))
                  # check if storage: power = (curr. MWh energy - prev. MWh energy)/interval Hrs

                  if comp.get_interaction().is_type('Storage') and tidx == 0:
                    init_level = comp.get_interaction().get_initial_level(meta)

                  if comp.get_interaction().is_type('Storage'):
                    # Initial_Level = float(self._unitInfo[unit]['Initial_Level'])
                    Initial_Level = float(init_level)
                    if tidx == 0: # for the first hour, use the initial level. charging yields to negative r_value
                      r_value = -(current - Initial_Level)/Tr_Update_hrs
                    else: # for the other hours
                      # r_value = -(current - float(dispatch.get_activity(comp, tracker, res, times[tidx-1])))/Tr_Update_hrs
                      r_value = -(current - Allowed_Level)/Tr_Update_hrs
                  else: # when not storage,
                    r_value = current # measured in MW

                  """ Find the correct profile according to r_value"""
                  profile_id = (np.abs(np.asarray(self._unitInfo[unit]['para_list']) - r_value)).argmin()
                  # print("t_idx=",t_idx, "t=",t)

                  # Retrive the correct A, B, C matrices
                  A_d = self._unitInfo[unit]['A_list'][profile_id]
                  B_d = self._unitInfo[unit]['B_list'][profile_id]
                  C_d = self._unitInfo[unit]['C_list'][profile_id]
                  D_d = np.zeros((p,m)) # all zero D matrix
                  
                  # Build the s, H and h for MOAS
                  
                  H_DMDc, h_DMDc = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g)  # H and h, type = <class 'numpy.ndarray'>
              
                  # first v_RG: consider the step "0" - step "g"
                  v_RG = fun_RG_SISO(0, x_KF, r_value-v_0, H_DMDc, h_DMDc, p) # v_RG: type == <class 'numpy.ndarray'>

                  # find the profile with max eigenvalue of A
                  max_eigA_id = np.asarray(self._unitInfo[unit]['eig_A_list']).argmax()
                  A_m = self._unitInfo[unit]['A_list'][max_eigA_id]
                  B_m = self._unitInfo[unit]['B_list'][max_eigA_id]
                  C_m = self._unitInfo[unit]['C_list'][max_eigA_id]
                  D_m = np.zeros((p,m)) # all zero D matrix

                  """ 2nd adjustment """
                  # MOAS for the steps "g+1" - step "2g"
                  Hm, hm = fun_MOAS_noinf(A_m, B_m, C_m, D_m, s, g)
                  # Calculate the max/min for v, ensuring the hm-Hxm*x(g+1) always positive for the next g steps.
                  v_max, v_min = fun_2nd_gstep_calc(x_KF, Hm, hm, A_m, B_m, g)

                  if v_RG < v_min:
                    v_RG = v_min
                  elif v_RG > v_max:
                    v_RG = v_max

                  # # Pretend there is no FARM intervention
                  # v_RG = np.asarray(r_value-r_0).flatten()

                  v_RG = float(v_RG)+float(v_0) # absolute value of electrical power (MW)
                  print("\n**************************", "\n**** RG summary Start ****","\nUnit = ", str(unit),", t = ", t, "\nr = ", r_value, "\nProfile Selected = ", profile_id, "\nv_RG = ", v_RG, "\n***** RG summary End *****","\n**************************\n")
                  print("HaoyuRGSummary,", str(unit), ",t,", t, ",Profile,", profile_id, 
                    ",r0,", r_value, ",v0,", v_RG, ",v_min,", float(v_min)+float(v_0), ",v_max,", float(v_max)+float(v_0) )
                  
                  # Update x_sys_internal, and keep record in v_hist and yp_hist within this hour
                  for i in range(int(Tr_Update_sec/Tss)):
                    # fetch y
                    y_fetch = np.asarray(fmu.getReal(vr_output))

                    # fetch v and x
                    v_fetch = np.asarray(v_RG).reshape(m,)
                    x_fetch = np.asarray(fmu.getReal(vr_state))
                    
                    # set the input. The input / state / output in FMU are real-world values
                    fmu.setReal(vr_input, [v_RG])
                    # perform one step
                    fmu.doStep(currentCommunicationPoint=t, communicationStepSize=Tss)

                    self._unitInfo[unit]['t_hist'].append(t)  # input v
                    self._unitInfo[unit]['v_hist'].append(v_fetch)  # input v
                    self._unitInfo[unit]['x_hist'].append(x_fetch)  # state x
                    self._unitInfo[unit]['y_hist'].append(y_fetch)  # output y

                    # time increment
                    t = t + Tss

                  # Convert to V1:

                  # if storage
                  if comp.get_interaction().is_type('Storage'):
                    if tidx == 0: # for the first hour, use the initial level
                      Allowed_Level = Initial_Level - v_RG*Tr_Update_hrs # Allowed_Level: predicted level due to v_value
                    else: # for the other hours
                      Allowed_Level = Allowed_Level - v_RG*Tr_Update_hrs
                    V1 = Allowed_Level
                  else: # when not storage,
                    V1 = v_RG

                  # print("Haoyu Debug, unit=",str(unit),", t=",time, ", curr= %.8g, V1= %.8g, delta=%.8g" %(current, V1, (V1-current)))

                  # Collect data for DMDc
                  winIdxStart = max(0, t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2))
                  winIdxEnd = t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2)
                  t_window = np.asarray(self._unitInfo[unit]['t_hist'][winIdxStart:winIdxEnd]).reshape(1,-1)
                  v_window = np.asarray(self._unitInfo[unit]['v_hist'][winIdxStart:winIdxEnd]).reshape(m,-1)
                  x_window = np.asarray(self._unitInfo[unit]['x_hist'][winIdxStart:winIdxEnd]).T
                  y_window = np.asarray(self._unitInfo[unit]['y_hist'][winIdxStart:winIdxEnd]).T  

                  # Do the DMDc, and return ABCD matrices
                  U1 = v_window[:,0:-1]-v_0; X1 = x_window[:, 0:-1]-x_0; X2 = x_window[:, 1:]-x_0; Y1 = y_window[:, 0:-1]-y_0
                  
                  # if U1 has any dimension unchanged within the window, DMDc will return A matrix with eigenvalue of 1.
                  # Skip DMDc if no U1 has constant components.
                  flagNoTransient = False
                  if abs(np.max(U1)-np.min(U1)) < self._unitInfo[unit]['NeighborSetpointsThreshold']: # if there is constant setpoint, cannot run DMDc
                    flagNoTransient = True

                  # Skip DMDc if a close by scheduling parameter already exists
                  CloseParaExist = False
                  if np.min(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1])) < self._unitInfo[unit]['NeighborSetpointsThreshold']: 
                    CloseParaExist = True

                  # the condition related to initialization:
                  initializationCond = False
                  if self._unitInfo[unit]['StateVariableSelectionXML'] is None:
                    if t_idx == len(LearningSetpoints):
                      initializationCond = True
                  else:
                    if t_idx == 0:
                      initializationCond = True

                  Do_DMDc = False
                  if not flagNoTransient and not CloseParaExist and not initializationCond: # if transient found within this window
                    Do_DMDc = True

                  if Do_DMDc:  # print(U1.shape)
                    Ad_Dc, Bd_Dc, Cd_Dc= fun_DMDc(X1, X2, U1, Y1, -1)
                    # Dd_Dc = np.zeros((p,m))
                    
                    # append the A,B,C,D matrices to an list
                    self._unitInfo[unit]['A_list'].append(Ad_Dc); 
                    self._unitInfo[unit]['B_list'].append(Bd_Dc); 
                    self._unitInfo[unit]['C_list'].append(Cd_Dc); 
                    self._unitInfo[unit]['para_list'].append(float(U1[:,-1]+v_0)); 
                    self._unitInfo[unit]['eig_A_list'].append(np.max(np.linalg.eig(Ad_Dc)[0]))
                    self._unitInfo[unit]['tTran_list'].append(t-Tr_Update_sec)
                    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    print("&&& DMDc summary Start &&&")
                    print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                    print("A_list=\n",self._unitInfo[unit]['A_list'])
                    print("B_list=\n",self._unitInfo[unit]['B_list'])
                    print("C_list=\n",self._unitInfo[unit]['C_list'])
                    print("para_list=\n",self._unitInfo[unit]['para_list'])
                    print("eig_A_list=\n",self._unitInfo[unit]['eig_A_list'])
                    print("tTran_list=\n",self._unitInfo[unit]['tTran_list'])
                    print("&&&& DMDc summary End &&&&")
                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                    # print(a)
                  else:
                    print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
                    print("&&& DMDc was not done. &&&")
                    print("Unit =", str(unit), ", t = ", t-Tr_Update_sec, ", v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                    if flagNoTransient:
                      print("Reason: Transient is too small. v_window[0] =", v_window[0][0], ", v_window[-1] =", v_window[0][-1])
                    if initializationCond:
                      print("Reason: System is initialized at t =",t-Tr_Update_sec)
                    if CloseParaExist: 
                      print("Reason: New parameter is too close to existing parameter [{}].".format(np.abs(np.asarray(self._unitInfo[unit]['para_list']) - v_window[:,-1]).argmin()))
                      print("New parameter =", v_window[:,-1], "Para_list =",self._unitInfo[unit]['para_list'])
                    print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
                  t_idx = t_idx+1

                  # Write up any violation to the errs:
                  if abs(current - V1) > self._tolerance*max(abs(current),abs(V1)):
                    # violation
                    errs.append({'msg': f'Reference Governor Violation',
                                'limit': V1,
                                'limit_type': 'lower' if (current < V1) else 'upper',
                                'component': comp,
                                'resource': res,
                                'time': time,
                                'time_index': tidx,
                                })

          # Get time for dispatching
          endTimeDispatch = copy.deepcopy(datetime.now())
          print('Haoyu t-debug, Time for this {} Dispatch is {}'.format(str(unit),endTimeDispatch-startTime))

    if errs == []: # if no validation error:
      print(" ")
      print("*********************************************************************")
      print("*** Haoyu Debug, Validation Success, Print for offline processing ***")
      print("*********************************************************************")
      print(" ")

      for unit in self._unitInfo:
        t_hist = np.array(self._unitInfo[unit]['t_hist']).T # t_hist.shape = (5040,)
        v_hist = np.array(self._unitInfo[unit]['v_hist']).T # v_hist.shape = (1, 5040)
        y_hist = np.array(self._unitInfo[unit]['y_hist']).T # y_hist.shape = (2, 5040)
        print(str(unit), t_hist.shape, v_hist.shape, y_hist.shape)

        for i in range(len(t_hist)):
          print("HaoyuDispatchSummary, " + str(unit) + 
          ", t, " + str(t_hist[i]) + ", vp, " + str(v_hist[0][i]) +
          ", y, " + ', '.join([str(num) for num in y_hist[:,i]]) + 
          ", ymin, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Min']]) +
          ", ymax, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Max']]))

    return errs

class FARM_MIMO(Validator):
  """
    A FARM SISO Validator for dispatch decisions.(Dirty Implementation)
    Accepts parameterized A,B,C,D matrices from external XML file and use the first set within constraints 
    as physics model, and validate
    the dispatch power (BOP, unit=MW)

    Haoyu Wang, ANL-NSE, March 28, 2022
  """
  # ---------------------------------------------
  # INITIALIZATION
  @classmethod
  def get_input_specs(cls):
    """
      Set acceptable input specifications.
      @ In, None
      @ Out, specs, InputData, specs
    """
    specs = Validator.get_input_specs()
    specs.name = 'FARM_MIMO'
    specs.description = r"""Feasible Actuator Range Modifier, which uses a Multi-input-Multi-output
        command governor validator to adjust the power setpoints issued to the components at the
        beginning of each dispatch interval (usually an hour), to ensure the the operational constraints
        were not violated during the following dispatch interval. This version uses a MIMO Functional Mock-up 
        Unit (FMU) as a downstream high-fidelity model."""

    component = InputData.parameterInputFactory('ComponentForFARM', ordered=False, baseNode=None,
        descr=r"""The MIMO component whose power setpoints will be adjusted by FARM. The user need
        to provide the file name, simulation step, component name mapping, Input, State, Output variables,
        Learning setpoints and operational constraints concerning this component.""")
    component.addParam('name',param_type=InputTypes.StringType, required=True,
        descr=r"""The name by which this component should be referred within HERON. It is nor required to match
        the component's name in \xmlNode{Components}.""")

    component.addSub(InputData.parameterInputFactory('StateVariableSelectionXML',contentType=InputTypes.StringType,
        descr=r"""The path to the XML file generated by state variable selection algorithm. Either absolute path
        or path relative to HERON / FARM root (starts with %HERON% / %FARM%) will work. NOTE: Once this file was provided,
        FARM will not read the following nodes: \xmlNode{FMUSimulationStep}, \xmlNode{InputVarNames}, 
        \xmlNode{StateVarNames}, \xmlNode{OutputVarNames}, \xmlNode{LearningSetpoints}, """))
    
    component.addSub(InputData.parameterInputFactory('FMUFile',contentType=InputTypes.StringType,
        descr=r"""The path to the FMU file of this component. Either absolute path
        or path relative to HERON / FARM root (starts with %HERON% / %FARM%) will work."""))
    component.addSub(InputData.parameterInputFactory('ComponentNamesForFMUInput',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of components contained in this FMU. The sequence should line up with the items in
         \xmlNode{InputVarNames} and the names should match the component's name in \xmlNode{Components}.
        It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('setpointsShiftStep',contentType=InputTypes.IntegerType,
        descr=r"""The time shift between adjacent setpoint leading edges, measured in steps. It should be an integer."""))
    
    component.addSub(InputData.parameterInputFactory('FMUSimulationStep',contentType=InputTypes.FloatType,
        descr=r"""The step length of FMU simulation, measured in seconds. It should be a floating number or an integer."""))
    component.addSub(InputData.parameterInputFactory('InputVarNames',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of FMU input variables. It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('StateVarNames',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of FMU state variables. It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('OutputVarNames',contentType=InputTypes.InterpretedListType,
        descr=r"""The names of FMU output variables. It should be a list of strings separated by comma."""))
    component.addSub(InputData.parameterInputFactory('LearningSetpoints',contentType=InputTypes.InterpretedListType,
        descr=r"""The learning setpoints are used to find the nominal value and first sets of ABCD matrices. 
        It should be a list of two or more floating numbers for component 1, then same amount of floating numbers for 
        component 2, etc. All numbers should be in one row and separated by comma."""))
    
    component.addSub(InputData.parameterInputFactory('OnlineMatrixUpdate',contentType=InputTypes.BoolType,
        descr=r"""The Online Matrix Update option is used to run FMU during power dispatch and update the matrix database
        for un-explored transients. It should be a boolean variable True/False."""))
    component.addSub(InputData.parameterInputFactory('NeighborSetpointsThreshold',contentType=InputTypes.FloatType,
        descr=r"""The Neighbor Setpoints Threshold is used to create a new ABCD matrices profile. 
        If a new setpoint falls within an existing setpoint plus or minus this threshold, no new profile will be created.
        It should be a floating number."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsUpper',contentType=InputTypes.InterpretedListType,
        descr=r"""The upper bounds for the output variables of this component. It should be a list of
        floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('OpConstraintsLower',contentType=InputTypes.InterpretedListType,
        descr=r"""The lower bounds for the output variables of this component. It should be a list of
        floating numbers or integers separated by comma."""))
    component.addSub(InputData.parameterInputFactory('RollingWindowWidth',contentType=InputTypes.IntegerType,
        descr=r"""The moving window duration for DMDc, with the unit of seconds. It should be an integer."""))

    specs.addSub(component)

    return specs

  def __init__(self):
    """
      Constructor.
      @ In, None
      @ Out, None
    """
    self.name = 'BaseValidator'
    self.rel_tolerance = 1.003e-2
    self.abs_tolerance = 0.01

  def read_input(self, inputs):
    """
      Loads settings based on provided inputs
      @ In, inputs, InputData.InputSpecs, input specifications
      @ Out, None
    """
    self._unitInfo = {}
    for component in inputs.subparts:
      name = component.parameterValues['name']

      # Default values
      svsXMLFile = None
      fmuFile = None
      ComponentNamesForFMUInput = None
      setpointsShiftStep = 0
      FMUSimulationStep = None
      InputVarNames = None
      StateVarNames = None
      OutputVarNames = None
      LearningSetpoints = None
      OnlineMatrixUpdate = False

      NeighborSetpointsThreshold = 1.0 # default threshold 1.0 MW
      for farmEntry in component.subparts:
        if farmEntry.getName() == "StateVariableSelectionXML":
          # fileName = farmEntry.value
          # print(os.path.abspath(__file__))
          # print(self._workingDir)

          svsXMLFile = farmEntry.value
          if svsXMLFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            svsXMLFile = os.path.abspath(svsXMLFile.replace('%HERON%', heron_path))
          elif svsXMLFile.startswith('%FARM%'):
            # magic word for "relative to FARM root"
            farm_path = get_farm_loc()
            svsXMLFile = os.path.abspath(svsXMLFile.replace('%FARM%', farm_path))
          else:
            svsXMLFile = os.path.abspath(svsXMLFile)
          
          if not os.path.isfile(svsXMLFile):
            raise RuntimeError(f'File not found for "{svsXMLFile}".')
        
        if farmEntry.getName() == "FMUFile":
          fmuFile = farmEntry.value
          if fmuFile.startswith('%HERON%'):
            # magic word for "relative to HERON root"
            heron_path = get_heron_loc()
            fmuFile = os.path.abspath(fmuFile.replace('%HERON%', heron_path))
          elif fmuFile.startswith('%FARM%'):
            # magic word for "relative to FARM root"
            farm_path = get_farm_loc()
            fmuFile = os.path.abspath(fmuFile.replace('%FARM%', farm_path))
        
        if farmEntry.getName() == "ComponentNamesForFMUInput":
          ComponentNamesForFMUInput = farmEntry.value
        
        if farmEntry.getName() == "setpointsShiftStep":
          setpointsShiftStep = farmEntry.value  

        if farmEntry.getName() == "FMUSimulationStep":
          FMUSimulationStep = farmEntry.value
        
        if farmEntry.getName() == "InputVarNames":
          InputVarNames = farmEntry.value
          Num_Input = len(InputVarNames)
        if farmEntry.getName() == "StateVarNames":
          StateVarNames = farmEntry.value
        if farmEntry.getName() == "OutputVarNames":
          OutputVarNames = farmEntry.value
          
        if farmEntry.getName() == "LearningSetpoints":
          LearningSetpoints = farmEntry.value
          if len(LearningSetpoints) < 2*Num_Input:
            sys.exit('\nERROR: <LearningSetpoints> XML node needs to contain 2 or more floating or integer numbers for each input variable.\n')   
          elif len(LearningSetpoints) % Num_Input != 0:
            sys.exit('\nERROR: <LearningSetpoints> XML node needs to contain same amount of numbers for each input variable.\n')
          else:
            # print("** Haoyu Debug, type(LearningSetpoints)=", type(LearningSetpoints))
            LearningSetpoints = np.asarray(LearningSetpoints).reshape(Num_Input,-1)
            for setPointSeries in LearningSetpoints: # check the constantness of each series of setpoints in learning stage
              if min(setPointSeries)==max(setPointSeries):
                exitMessage = """ERROR:  No transient found in <LearningSetpoints>. \n\tPlease modify the values in <LearningSetpoints>.\n"""
                sys.exit(exitMessage)  

        if farmEntry.getName() == "OnlineMatrixUpdate":
          OnlineMatrixUpdate = farmEntry.value
          
        if farmEntry.getName() == "NeighborSetpointsThreshold":
          NeighborSetpointsThreshold = farmEntry.value
        
        if farmEntry.getName() == "RollingWindowWidth":
          RollingWindowWidth = farmEntry.value
        if farmEntry.getName() == "OpConstraintsUpper":
          UpperBound = farmEntry.value
        if farmEntry.getName() == "OpConstraintsLower":
          LowerBound = farmEntry.value

      self._unitInfo.update(
        {name:{
          'StateVariableSelectionXML':svsXMLFile,
          'FMUFile':fmuFile,
          'ComponentNamesForFMUInput':ComponentNamesForFMUInput,
          'setpointsShiftStep':setpointsShiftStep,
          'FMUSimulationStep':FMUSimulationStep,
          'InputVarNames':InputVarNames,
          'StateVarNames':StateVarNames,
          'OutputVarNames':OutputVarNames,
          'LearningSetpoints':LearningSetpoints,
          'OnlineMatrixUpdate':OnlineMatrixUpdate,
          'NeighborSetpointsThreshold':NeighborSetpointsThreshold,
          'RollingWindowWidth':RollingWindowWidth,
          'Targets_Max':UpperBound,
          'Targets_Min':LowerBound,
          't_hist_sl':[],
          'v_hist_sl':[],
          'x_hist_sl':[],
          'y_hist_sl':[],
          'v_0_sl': None,
          'x_0_sl': None,
          'y_0_sl': None,
          't_idx_sl': None,
          'A_list_sl':[],
          'B_list_sl':[],
          'C_list_sl':[],
          'eig_A_list_sl':[],
          'para_list_sl':[],
          'tTran_list_sl':[],
          't_hist':[],
          'v_hist':[],
          'x_hist':[],
          'y_hist':[],
          'A_list':[],
          'B_list':[],
          'C_list':[],
          'eig_A_list':[],
          'para_list':[],
          'tTran_list':[]}})
    print('\n self._unitInfo = \n',self._unitInfo,'\n')
    # print some error message here:
    print('FARM-Delta Validator Summary:\n')
    if svsXMLFile is None:
      print("Self-Learning: Yes\n")
      if fmuFile is None:
        raise RuntimeError("Need to specify <FMUFile>.\n")
      if ComponentNamesForFMUInput is None:
        raise RuntimeError("Need to specify <ComponentNamesForFMUInput>.\n")
      if setpointsShiftStep <= 0:
        raise RuntimeError("Need a positive number in <setpointsShiftStep>.\n")

    else:
      print("Self-Learning: No\n")


    # perform the self-learning stage here

  # ---------------------------------------------
  # API
  def validate(self, components, dispatch, times, meta):
    """
      Method to validate a dispatch activity.
      @ In, components, list, HERON components whose cashflows should be evaluated
      @ In, activity, DispatchState instance, activity by component/resources/time
      @ In, times, np.array(float), time values to evaluate; may be length 1 or longer
      @ In, meta, dict, extra information pertaining to validation
      @ Out, errs, list, information about validation failures
    """
    # get start time
    startTime = copy.deepcopy(datetime.now())
    # errs will be returned to dispatcher. errs contains all the validation errors calculated in below
    errs = [] # TODO best format for this?

    """ get time interval"""
    Tr_Update_hrs = float(times[1]-times[0])
    Tr_Update_sec = Tr_Update_hrs*3600.

    # get all information from the only "unit" in self._unitInfo
    for unit in self._unitInfo:
      """ 1. Constraints information, and Moving window width """
      setpoints_shift_step = self._unitInfo[unit]['setpointsShiftStep']
      # Constraints
      y_min = np.asarray(self._unitInfo[unit]['Targets_Min'])
      y_max = np.asarray(self._unitInfo[unit]['Targets_Max'])
      # The width of moving window (seconds, centered at transient edge, for moving window DMDc)
      Moving_Window_Width = self._unitInfo[unit]['RollingWindowWidth']; #Tr_Update
      # Load the self learning hists, if any
      self._unitInfo[unit]['t_hist']=copy.deepcopy(self._unitInfo[unit]['t_hist_sl'])
      self._unitInfo[unit]['v_hist']=copy.deepcopy(self._unitInfo[unit]['v_hist_sl'])
      self._unitInfo[unit]['x_hist']=copy.deepcopy(self._unitInfo[unit]['x_hist_sl'])
      self._unitInfo[unit]['y_hist']=copy.deepcopy(self._unitInfo[unit]['y_hist_sl'])
      # empty the A_list, B_list, C_list, eig_A_list, para_list, tTran_list
      self._unitInfo[unit]['A_list']=copy.deepcopy(self._unitInfo[unit]['A_list_sl'])
      self._unitInfo[unit]['B_list']=copy.deepcopy(self._unitInfo[unit]['B_list_sl'])
      self._unitInfo[unit]['C_list']=copy.deepcopy(self._unitInfo[unit]['C_list_sl'])
      self._unitInfo[unit]['eig_A_list']=copy.deepcopy(self._unitInfo[unit]['eig_A_list_sl'])
      self._unitInfo[unit]['para_list']=copy.deepcopy(self._unitInfo[unit]['para_list_sl'])
      self._unitInfo[unit]['tTran_list']=copy.deepcopy(self._unitInfo[unit]['tTran_list_sl'])

      """ 2. Read FMU file to specify the physical model"""
      # The name of FMU file, and the name of sub-systems 
      fmu_filename = self._unitInfo[unit]['FMUFile']
      inputNameSeq = self._unitInfo[unit]['ComponentNamesForFMUInput'] 

      # Input / State / Output variable names, dimensions, and Time step (Tss)
      if self._unitInfo[unit]['StateVariableSelectionXML'] is None: # if state variable selection is not provided, 
        print("******, Will do Self learning ******")
        # fetch variable names from self._unitInfo[unit]
        inputVarNames = self._unitInfo[unit]['InputVarNames']
        stateVarNames = self._unitInfo[unit]['StateVarNames']
        outputVarNames = self._unitInfo[unit]['OutputVarNames']
        Tss = self._unitInfo[unit]['FMUSimulationStep']
        # Dimensions of input (m), states (n) and output (p)
        m=len(inputVarNames); n=len(stateVarNames); p=len(outputVarNames)
      else: # if state variable selection is provided
        print("******, Will not do Self learning, reading everything from state variable selection XML ******")
        # read xml here, and assign values to the three VarNames and Tss above
        """ Read State Space XML file (generated by Raven parameterized DMDc) """
        MatrixFile = self._unitInfo[unit]['StateVariableSelectionXML']
        Tss, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array, actuatorList, stateList, outputList= read_parameterized_XML(MatrixFile)
        # convert para_array to para_list
        para_list = [(item.reshape(m,)).tolist() for item in para_array]
        eig_A_list = eig_A_array.tolist()
        # fetch variable names from XML file
        inputVarNames = actuatorList
        stateVarNames = stateList
        outputVarNames = outputList
      
      if fmu_filename is not None:
        print("******, Initiating FMU ******")
        # Read the model description
        model_description = read_model_description(fmu_filename)
        # Collect all the value references (handles) from FMU
        vrs = {}
        for variable in model_description.modelVariables:
            vrs[variable.name] = variable.valueReference
        
        # Get the value references (handle) for the variables we want to get/set
        # Inputs: Power Setpoint
        vr_input = [vrs[item] for item in inputVarNames]
        # States: 
        vr_state = [vrs[item] for item in stateVarNames]
        # Outputs: 
        vr_output = [vrs[item] for item in outputVarNames]
        
        # Extract the FMU
        unzipdir = extract(fmu_filename)
        fmu = FMU2Slave(guid=model_description.guid,
                        unzipDirectory=unzipdir,
                        modelIdentifier=model_description.coSimulation.modelIdentifier,
                        instanceName='instance1')
        
        # Initialize FMU for self-learning stage
        T_delaystart = 0.
        fmu.instantiate()
        fmu.setupExperiment(startTime=T_delaystart)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()

      if ((self._unitInfo[unit]['StateVariableSelectionXML'] is None) or (self._unitInfo[unit]['OnlineMatrixUpdate'])):
        # Calculate the integer window width for DMDc
        window = int(Moving_Window_Width/Tss) 

      if self._unitInfo[unit]['StateVariableSelectionXML'] is None: # if the self learning was not done during SVS
        print("******, Self learning Starts ******")
        
        """ 3. Simulate FMU using the 1st setpoint, to get the steady state output """
        LearningSetpoints = self._unitInfo[unit]['LearningSetpoints']
        
        if len(self._unitInfo[unit]['t_hist']) == 0: # if self-learning was never run before  
          t = -Tr_Update_sec*len(LearningSetpoints[0]) # negative values means self-learning stage
          t_idx = 0 # this is the index for "LearningSetpoints"

          # Do the step-by-step simulation, from beginning to the first transient
          while t < -Tr_Update_sec*(len(LearningSetpoints[0])-1): # only the steady state value
            # Find the current r value
            r = copy.deepcopy(LearningSetpoints[:,t_idx]) # <class 'numpy.ndarray'>, r.shape = (m,)
            
            # No reference governor for the first setpoint value, direct copy value from r
            v = copy.deepcopy(r) # <class 'numpy.ndarray'>, v.shape = (m,)
            
            # fetch v 
            v_fetch = np.asarray(v).reshape(m,)
            
            # fetch y, x 
            y_fetch = np.asarray(fmu.getReal(vr_output)) # <class 'numpy.ndarray'>, y_fetch.shape = (p,)
            x_fetch = np.asarray(fmu.getReal(vr_state)) # <class 'numpy.ndarray'>, x_fetch.shape = (n,)
                                    
            # Save time, r, v, and y
            self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))         # Time
            self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))   # input v
            self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))   # state x
            self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))   # output y

            # Set the input. The input / state / output in FMU are real-world values (before nominal subtraction)
            for i in range(len(vr_input)):
              fmu.setReal([vr_input[i]], [v[i]]) # Note: fmu.setReal must take two lists as input
            # perform one step
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=Tss)
            # print("Wall Time = {}, ".format(datetime.now().strftime("%Y_%m_%d_%H:%M:%S:%f")),"FMU t = ",t)

            # time increment
            t = t + Tss

          # fetch the steady-state y variables, and save as 2-dimensional ndarray with the shape of (m/n/p, 1)
          v_0 = copy.deepcopy(v_fetch.reshape(m,-1))
          x_0 = copy.deepcopy(x_fetch.reshape(n,-1))
          y_0 = copy.deepcopy(y_fetch.reshape(p,-1))

          # store v_0, x_0 and y_0 into self._unitInfo
          self._unitInfo[unit]['v_0_sl'] = copy.deepcopy(v_0)
          self._unitInfo[unit]['x_0_sl'] = copy.deepcopy(x_0)
          self._unitInfo[unit]['y_0_sl'] = copy.deepcopy(y_0)
          # time index increment
          t_idx += 1
          
          # check if steady-state y is within the [ymin, ymax]
          for i in range(len(y_0)):
            if y_0[i][0]>y_max[i]:
              exitMessage = """\n\tERROR:  Steady state output y_STEADY[{:d}] is {:.2f} HIGHER than y upper constraints. \n
              \tFYI:  y_STEADY = {};
              \tFYI: y_maximum = {};
              \tFYI: y_minimum = {}.\n
              \tPlease modify the steady state setpoint in learningSetpoints[:,0].\n""".format(i, y_0[i][0]-y_max[i],
              np.array2string(y_0.flatten(), formatter={'float_kind':lambda x: "%.4e" % x}),
              np.array2string(y_max, formatter={'float_kind':lambda x: "%.4e" % x}),
              np.array2string(y_min, formatter={'float_kind':lambda x: "%.4e" % x}),
              )
              sys.exit(exitMessage)
            elif y_0[i]<y_min[i]:
              print(y_min[i], y_0[i][0])
              exitMessage = """\n\tERROR:  Steady state output y_STEADY[{:d}] is {:.2f} LOWER than y lower constraints. \n
              \tFYI: y_maximum = {};
              \tFYI: y_minimum = {};
              \tFYI:  y_STEADY = {}.\n
              \tPlease modify the steady state setpoint in learningSetpoints[:,0].\n""".format(i, y_min[i]-y_0[i][0],
              np.array2string(y_max, formatter={'float_kind':lambda x: "%.4e" % x}),
              np.array2string(y_min, formatter={'float_kind':lambda x: "%.4e" % x}),
              np.array2string(y_0.flatten(), formatter={'float_kind':lambda x: "%.4e" % x}),
              )
              sys.exit(exitMessage)
          print("\n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
          print("^^^ Steady State Summary Start ^^^")
          print("t =", t - Tss, "\nv_0 =\n", v_0, "\nx_0 = \n",x_0,"\ny_0 = \n",y_0)
          print("^^^^ Steady State Summary End ^^^^")
          print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n")
          # Summary: at the end of Step 4, the nominal values are stored in u_0, x_0 and y_0. 


          """ 4. Simulate the using the rest LearningSetpoints values, to get the ABCD matrices through self-learning """                      
          while t_idx < LearningSetpoints[0].size:
            # extract the desired r vector
            r_spec = LearningSetpoints[:,t_idx] # <class 'numpy.ndarray'>, r.shape = (m,)            

            # Shift input
            i_input = 0.
            while t < -Tr_Update_sec*(len(LearningSetpoints[0])-1-t_idx): # the 2nd value in LearningSetpoints
              # Find the current r value              
              r = copy.deepcopy(r_spec)
              
              # No reference governor during self-learning stage
              if i_input < m:
                v[math.floor(i_input)]=copy.deepcopy(r[math.floor(i_input)])
                i_input += 1/setpoints_shift_step
              
              # fetch y
              y_fetch = np.asarray(fmu.getReal(vr_output)) # <class 'numpy.ndarray'>, y_fetch.shape = (p,)

              # fetch v and x
              v_fetch = np.asarray(v).reshape(m,)
              x_fetch = np.asarray(fmu.getReal(vr_state)) # <class 'numpy.ndarray'>, x_fetch.shape = (n,)

              # Save time, r, v, and y
              self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))         # Time
              self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))   # input v
              self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))   # state x
              self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))   # output y
              
              # Set the input. The input / state / output in FMU are real-world values (before nominal subtraction)
              for i in range(len(vr_input)):
                fmu.setReal([vr_input[i]], [v[i]]) # Note: fmu.setReal must take two lists as input
              # perform one step
              fmu.doStep(currentCommunicationPoint=t, communicationStepSize=Tss)
              # print("Wall Time = {}, ".format(datetime.now().strftime("%Y_%m_%d_%H:%M:%S:%f")),"FMU t = ",t)

              # time increment
              t = t + Tss

            # Collect data for DMDc
            winIdxStart = max(0, t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2))
            winIdxEnd = t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2)
            t_window = np.asarray(self._unitInfo[unit]['t_hist'][winIdxStart:winIdxEnd]).reshape(1,-1)    # (1, window)
            v_window = np.asarray(self._unitInfo[unit]['v_hist'][winIdxStart:winIdxEnd]).reshape(-1,m).T  # (m, window)
            x_window = np.asarray(self._unitInfo[unit]['x_hist'][winIdxStart:winIdxEnd]).T                # (n, window)
            y_window = np.asarray(self._unitInfo[unit]['y_hist'][winIdxStart:winIdxEnd]).T                # (p, window)
            # print("Window Start = {}, Window End = {}".format(winIdxStart, winIdxEnd))
            # print(a)
            
            # Do the DMDc, and return ABCD matrices during self-learning stage
            U1 = v_window[:,0:-1]-v_0; X1 = x_window[:, 0:-1]-x_0; X2 = x_window[:, 1:]-x_0; Y1 = y_window[:, 0:-1]-y_0

            # Skip DMDc if no U1 has constant components. (Otherwise DMDc will have singulatiry error)
            flagNoTransient = False # initialize the flag
            for item in U1: 
              if abs(np.max(item) - np.min(item)) < self._unitInfo[unit]['NeighborSetpointsThreshold']: # if there is constant setpoint, cannot run DMDc
                flagNoTransient = True

            Do_DMDc = False # initialize the flag for DMDc
            if not flagNoTransient: # if transient found within this window
              if len(self._unitInfo[unit]['para_list'])==0: # if para list is empty (DMDc never done before), do DMDc
                Do_DMDc = True
              else: # if para list is not empty (DMDc done before)
                CloseParaExist = False; nearest_para=[]
                for item in self._unitInfo[unit]['para_list']: # check each entry of para list and see if it's too close to current v
                  if np.linalg.norm(np.asarray(item).reshape(-1,)-v_window[:,-1].reshape(-1,)) < self._unitInfo[unit]['NeighborSetpointsThreshold']:
                    CloseParaExist = True
                    nearest_para = item

                if not CloseParaExist: # if no close parameter exists, do DMDc
                  Do_DMDc = True

            if Do_DMDc:
              Ad_Dc, Bd_Dc, Cd_Dc= fun_DMDc(X1, X2, U1, Y1, -1)
              # Dd_Dc = np.zeros((p,m))
              # append the A,B,C,D matrices to an list
              self._unitInfo[unit]['A_list'].append(Ad_Dc); 
              self._unitInfo[unit]['B_list'].append(Bd_Dc); 
              self._unitInfo[unit]['C_list'].append(Cd_Dc); 
              self._unitInfo[unit]['para_list'].append((U1[:,-1]+v_0.reshape(m,)).tolist()); 
              self._unitInfo[unit]['eig_A_list'].append(np.max(np.linalg.eig(Ad_Dc)[0]))
              self._unitInfo[unit]['tTran_list'].append(t-Tr_Update_sec)
              print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
              print("&&& DMDc summary Start &&&")
              print("Unit =", str(unit), ", t_idx = ", t_idx, ", t = ", t-Tr_Update_sec, ", v_window[:,0] =", v_window[:,0], ", v_window[:,-1] =", v_window[:,-1])
              print("A_list=\n",self._unitInfo[unit]['A_list'])
              print("B_list=\n",self._unitInfo[unit]['B_list'])
              print("C_list=\n",self._unitInfo[unit]['C_list'])
              print("para_list=\n",self._unitInfo[unit]['para_list'])
              print("eig_A_list=\n",self._unitInfo[unit]['eig_A_list'])
              print("tTran_list=\n",self._unitInfo[unit]['tTran_list'])
              print("&&&& DMDc summary End &&&&")
              print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
              # print(a)
            else:
              print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
              print("&&& DMDc was not done. &&&")
              print("t_idx = ", t_idx, ", t = ", t-Tr_Update_sec, ", v_window[:,0] =", v_window[:,0], ", v_window[:,-1] =", v_window[:,-1])
              if flagNoTransient:
                print("Reason: Transient is too small. v_window[:,0] =", v_window[:,0], ", v_window[:,-1] =", v_window[:,-1])
              elif CloseParaExist: 
                print("Reason: New parameter is too close to existing parameter [{}].".format(nearest_para))
                print("New parameter =", v_window[:,-1].tolist(), "Para_list =",self._unitInfo[unit]['para_list'])
              print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
            
            t_idx += 1

          if len(self._unitInfo[unit]['A_list']) == 0:
            sys.exit('ERROR:  No proper transient found in "LearningSetpoints". \n\tPlease modify the values in "LearningSetpoints" to ensure all the set-points changed at least one moment.\n')
          # print(a)

          # save the histories to _sl hist
          self._unitInfo[unit]['t_hist_sl']=copy.deepcopy(self._unitInfo[unit]['t_hist'])
          self._unitInfo[unit]['v_hist_sl']=copy.deepcopy(self._unitInfo[unit]['v_hist'])
          self._unitInfo[unit]['x_hist_sl']=copy.deepcopy(self._unitInfo[unit]['x_hist'])
          self._unitInfo[unit]['y_hist_sl']=copy.deepcopy(self._unitInfo[unit]['y_hist'])
          # empty the A_list, B_list, C_list, eig_A_list, para_list, tTran_list
          self._unitInfo[unit]['A_list_sl']=copy.deepcopy(self._unitInfo[unit]['A_list'])
          self._unitInfo[unit]['B_list_sl']=copy.deepcopy(self._unitInfo[unit]['B_list'])
          self._unitInfo[unit]['C_list_sl']=copy.deepcopy(self._unitInfo[unit]['C_list'])
          self._unitInfo[unit]['eig_A_list_sl']=copy.deepcopy(self._unitInfo[unit]['eig_A_list'])
          self._unitInfo[unit]['para_list_sl']=copy.deepcopy(self._unitInfo[unit]['para_list'])
          self._unitInfo[unit]['tTran_list_sl']=copy.deepcopy(self._unitInfo[unit]['tTran_list'])
          self._unitInfo[unit]['t_idx_sl']=copy.deepcopy(t_idx)

          # Get time for self-learning
          endTimeSL = copy.deepcopy(datetime.now())
          print('Haoyu t-debug, Time for Self_Learning is {}'.format(endTimeSL-startTime))
      else: # if state variable selection XML is provided, No self-learning
        # 3. find proper nominal values where y_0 is within the bound
        for i_para in range(len(para_list)):
          # examine the y monimal values, item # i_para
          y_0_temp = YNorm_list[i_para]
          # examine if every element is within the constrant
          inConstraint = True
          for i_output in range(p):
            if y_0_temp[i_output] <= y_max[i_output] and y_0_temp[i_output] >= y_min[i_output]:
              inConstraint = inConstraint and True
            else:
              inConstraint = inConstraint and False
          
          # check the result
          if inConstraint:
            v_0 = copy.deepcopy(UNorm_list[i_para].reshape(m,-1))
            x_0 = copy.deepcopy(XNorm_list[i_para].reshape(n,-1))
            y_0 = copy.deepcopy(YNorm_list[i_para].reshape(p,-1))

            self._unitInfo[unit]['v_0_sl'] = copy.deepcopy(v_0)
            self._unitInfo[unit]['x_0_sl'] = copy.deepcopy(x_0)
            self._unitInfo[unit]['y_0_sl'] = copy.deepcopy(y_0)
            break

        if not inConstraint:
          exitMessage = """\n\tERROR:  No proper steady state output is provided in the state variable selection XML file. \n"""

        # 4. save the A_list, B_list, C_list, para_list
        self._unitInfo[unit]['A_list'] = copy.deepcopy(A_list)
        self._unitInfo[unit]['B_list'] = copy.deepcopy(B_list)
        self._unitInfo[unit]['C_list'] = copy.deepcopy(C_list)
        self._unitInfo[unit]['para_list'] = copy.deepcopy(para_list)
        self._unitInfo[unit]['eig_A_list'] = copy.deepcopy(eig_A_list)
        self._unitInfo[unit]['t_idx_sl'] = 0
      

      """ 5. Simulate the Dispatch stage using CG, and update the ABCD matrices as it goes """
      # Initialization of time, and retrieve of norminal values
      t = 0
      # store v_0, x_0 and y_0 into self._unitInfo
      v_0 = copy.deepcopy(self._unitInfo[unit]['v_0_sl'])
      x_0 = copy.deepcopy(self._unitInfo[unit]['x_0_sl'])
      y_0 = copy.deepcopy(self._unitInfo[unit]['y_0_sl'])
      t_idx = copy.deepcopy(self._unitInfo[unit]['t_idx_sl'])
      # MOAS steps Limit 
      g = 2*int(Tr_Update_sec/Tss) + max(2,m*setpoints_shift_step) # numbers of steps to look forward, , type = <class 'int'>
      # Calculate s for Maximal Output Admissible Set (MOAS)
      s = [] # type == <class 'list'>
      for i in range(0,p):
        s.append(abs(y_max[i] - y_0[i]))
        s.append(abs(y_0[i] - y_min[i]))
      s = np.asarray(s).tolist()
      # print(s)

      # if to run FMU during dispatch to update matrix online
      if self._unitInfo[unit]['OnlineMatrixUpdate']:
        # Initialize FMU
        fmu.instantiate()
        fmu.setupExperiment(startTime=T_delaystart)
        fmu.enterInitializationMode()
        fmu.exitInitializationMode()

        # fetch the initial state
        x_fetch = np.asarray(fmu.getReal(vr_state)) # shape = (m,)
        y_fetch = np.asarray(fmu.getReal(vr_output)) # <class 'numpy.ndarray'>, y_fetch.shape = (p,)
      # When not needing online matrix update, 
      else:
        x_fetch = x_0.reshape(-1,)     
        y_fetch = y_0.reshape(-1,)

      # loop through the time index (tidx) and time in "times", extract the dispatched setpoint
      for tidx, time in enumerate(times):
        # create an empty r array, and the empty compdict dictionary
        r_spec = np.zeros((m,)); allowedLevel = np.zeros((m,))

        # loop through the <Component> items in HERON
        for comp, info in dispatch._resources.items():
          # e.g. comp= <HERON Component "BOP""> <HERON Component "SES"">

          for tracker in comp.get_tracking_vars():
            # loop through the resources in info (only one resource here - electricity)
            for res in info:
              if str(res) == "electricity":
                """ Get the r_comp, original actuation value """
                current = float(dispatch.get_activity(comp, tracker, res, times[tidx]))
                
                # check if storage: power = (curr. MWh energy - prev. MWh energy)/interval Hrs
                if comp.get_interaction().is_type('Storage'):
                  if tidx == 0: # for the first hour, use the initial level. charging yields to negative r_comp
                    init_level = comp.get_interaction().get_initial_level(meta)
                    Initial_Level = float(init_level)
                    
                    for i in range(m):
                      # line up the r_comp values to the r array, also save the initial value
                      if inputNameSeq[i] in str(comp):
                        r_comp = -(current - Initial_Level)/Tr_Update_hrs
                        r_spec[i] = r_comp
                        allowedLevel[i] = Initial_Level
                  else: # for the other hours
                    for i in range(m):
                      # line up the r_comp values to the r array, also save the initial value
                      if inputNameSeq[i] in str(comp):
                        Allowed_Level = allowedLevel[i]
                        r_comp = -(current - Allowed_Level)/Tr_Update_hrs
                        r_spec[i] = r_comp
                else: # if not storage
                  r_comp = current # measured in MW
                  for i in range(m):
                    # line up the r_comp values to the r array, also save the initial value
                    if inputNameSeq[i] in str(comp):
                      r_spec[i] = r_comp


        print("\n**************************")
        print("          CG info         ")
        print("tidx = {}, time = {}, r_spec = {}, allowedlevel = {}".format(tidx, time, r_spec, allowedLevel))

        # Do CG for this transient
        
        # Find the current r value
        r = copy.deepcopy(r_spec)

        # build an nearest neighbor classifier
        neigh_classifier = neighbors.KNeighborsRegressor(n_neighbors=1)
        neigh_classifier.fit(np.asarray(self._unitInfo[unit]['para_list']), np.asarray(range(len(self._unitInfo[unit]['para_list']))))
        # Find the correct profile according to r_spec
        profile_id = neigh_classifier.predict(r_spec.reshape(1,-1)).astype(int)[0]
        print("Profile Selected from r = ", profile_id)

        # Retrive the correct A, B, C matrices
        A_d = copy.deepcopy(self._unitInfo[unit]['A_list'][profile_id]); 
        B_d = copy.deepcopy(self._unitInfo[unit]['B_list'][profile_id]); 
        C_d = copy.deepcopy(self._unitInfo[unit]['C_list'][profile_id]); 
        D_d = np.zeros((p,m)) # all zero D matrix

        # Calculate the H and h in MOAS: H_DMDc * [x.T v.T].T < h_DMDc
        if tidx == 0: # The 1st production set-point: use regular MOAS
          H_DMDc, h_DMDc = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g, m*setpoints_shift_step)  # H and h, type = <class 'numpy.ndarray'>
        else: # from the 2nd production set-point: use shifted MOAS
          H_DMDc, h_DMDc = fun_MOAS_MIMO_Setpoint_Shift(A_d, B_d, C_d, D_d, s, g, copy.deepcopy(v_CG), setpoints_shift_step, m*setpoints_shift_step)
        
        # remove the redundant linear constraints in Hx_DMDc * x + Hv_DMDc * v <= hv_DMDc
        Hx_DMDc = H_DMDc[:, 0:n]; Hv_DMDc = H_DMDc[:, n:]
        hv_DMDc = (h_DMDc - np.dot(Hx_DMDc,x_fetch.reshape(n,-1)-x_0)).reshape(-1,) # hv is the system remaining vector. We need to obey Hv * v_RG < hv
        print("Hv_DMDc={}, hv_DMDc={}".format(Hv_DMDc.shape, hv_DMDc.shape))

        Hv_Redund = Hv_DMDc; hv_Redund = hv_DMDc
        # Hv_noRedund = Hv_DMDc; hv_noRedund = hv_DMDc
        Idx_Redund = np.arange(1,hv_Redund.shape[0]+1).reshape(-1,)
        # print(Idx_Redund, type(Idx_Redund))
        Hv_noRedund, hv_noRedund, Idx_noRedund = noRedund(Hv_Redund, hv_Redund, Idx_Redund)
        i=0
        print('Trial #{}, {} constraints left.'.format(i,hv_noRedund.shape[0]))
        while hv_Redund.shape[0] != hv_noRedund.shape[0]:
          Hv_Redund = Hv_noRedund; hv_Redund = hv_noRedund; Idx_Redund = Idx_noRedund
          Hv_noRedund, hv_noRedund, Idx_noRedund = noRedund(Hv_Redund, hv_Redund, Idx_Redund)
          i+=1
          print('Trial #{}, {} constraints left.'.format(i,hv_noRedund.shape[0]))

        print("Hv_noRedund={}, hv_noRedund={}".format(Hv_noRedund.shape, hv_noRedund.shape))

        timeString = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save Hv, hv, r, v_0 as .mat file for offline analysis
        mdic = {
          "Hv_orig":Hv_DMDc, "hv_orig":hv_DMDc.reshape(-1,1), 
          "Hv_noRedund":Hv_noRedund, "hv_noRedund":hv_noRedund.reshape(-1,1), "Idx_noRedund":Idx_noRedund.reshape(-1,1),
          "r":(r.reshape(m,-1)-v_0).reshape(-1,1), "v_0":v_0.reshape(-1,1), "y":y_fetch.reshape(-1,1)}
        savemat('FeasibleRegion_tidx_{:03}_Clock_{}.mat'.format(tidx, timeString),mdic)
        
        # calculate v_RG using command governor
        # print("Tentative r_value =\n",r_value)    
        v_CG = fun_CG_MIMO(r.reshape(m,-1)-v_0, Hv_noRedund, hv_noRedund) # 
        print("v = ",(v_CG+v_0).reshape(-1,))
        profile_id_r = copy.deepcopy(profile_id)

        # re-calculate the profile ID using v_CG
        profile_id = neigh_classifier.predict((v_CG+v_0).reshape(1,-1)).astype(int)[0]
        print("Profile Selected from v_CG= ", profile_id)

        # if profile_id and profile_id_r are different
        # while profile_id != profile_id_r:
        #   print('\nConflict. Profile #{} for r, but Profile #{} for v. Revalidating using Profile #{}...'.format(profile_id_r, profile_id, profile_id))
        #   # Retrive the correct A, B, C matrices
        #   A_d = copy.deepcopy(self._unitInfo[unit]['A_list'][profile_id]); 
        #   B_d = copy.deepcopy(self._unitInfo[unit]['B_list'][profile_id]); 
        #   C_d = copy.deepcopy(self._unitInfo[unit]['C_list'][profile_id]); 
        #   D_d = np.zeros((p,m)) # all zero D matrix

        #   # Calculate the H and h in MOAS: H_DMDc * [x.T v.T].T < h_DMDc
        #   if tidx == 0: # The 1st production set-point: use regular MOAS
        #     H_DMDc, h_DMDc = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g, m*setpoints_shift_step)  # H and h, type = <class 'numpy.ndarray'>
        #   else: # from the 2nd production set-point: use shifted MOAS
        #     H_DMDc, h_DMDc = fun_MOAS_MIMO_Setpoint_Shift(A_d, B_d, C_d, D_d, s, g, copy.deepcopy(v_CG), setpoints_shift_step, m*setpoints_shift_step)
          
        #   # remove the redundant linear constraints in Hx_DMDc * x + Hv_DMDc * v <= hv_DMDc
        #   Hx_DMDc = H_DMDc[:, 0:n]; Hv_DMDc = H_DMDc[:, n:]
        #   hv_DMDc = (h_DMDc - np.dot(Hx_DMDc,x_fetch.reshape(n,-1)-x_0)).reshape(-1,) # hv is the system remaining vector. We need to obey Hv * v_RG < hv
        #   print("Hv_DMDc={}, hv_DMDc={}".format(Hv_DMDc.shape, hv_DMDc.shape))

        #   Hv_Redund = Hv_DMDc; hv_Redund = hv_DMDc
        #   # Hv_noRedund = Hv_DMDc; hv_noRedund = hv_DMDc
        #   Hv_noRedund, hv_noRedund = noRedund(Hv_Redund, hv_Redund)
        #   i=0
        #   print('Trial #{}, {} constraints left.'.format(i,hv_noRedund.shape[0]))
        #   while hv_Redund.shape[0] != hv_noRedund.shape[0]:
        #     Hv_Redund = Hv_noRedund; hv_Redund = hv_noRedund
        #     Hv_noRedund, hv_noRedund = noRedund(Hv_Redund, hv_Redund)
        #     i+=1
        #     print('Trial #{}, {} constraints left.'.format(i,hv_noRedund.shape[0]))

        #   print("Hv_noRedund={}, hv_noRedund={}".format(Hv_noRedund.shape, hv_noRedund.shape))

        #   # timeString = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
          
        #   # Save Hv, hv, r, v_0 as .mat file for offline analysis
        #   mdic = {
        #     "Hv_orig":Hv_DMDc, "hv_orig":hv_DMDc.reshape(-1,1), 
        #     "Hv_noRedund":Hv_noRedund, "hv_noRedund":hv_noRedund.reshape(-1,1), 
        #     "r":(r.reshape(m,-1)-v_0).reshape(-1,1), "v_0":v_0.reshape(-1,1)}
        #   savemat('FeasibleRegion_tidx_{:03}_Clock_{}.mat'.format(tidx, timeString),mdic)
          
        #   # calculate v_RG using command governor
        #   # print("Tentative r_value =\n",r_value)    
        #   v_CG = fun_CG_MIMO(r.reshape(m,-1)-v_0, Hv_noRedund, hv_noRedund) # 
        #   print("v = ",(v_CG+v_0).reshape(-1,))
        #   profile_id_r = copy.deepcopy(profile_id)

        #   # re-calculate the profile ID using v_CG
        #   profile_id = neigh_classifier.predict((v_CG+v_0).reshape(1,-1)).astype(int)[0]
        #   print("Profile Selected from v_CG= ", profile_id)

        print("        CG info End       ")
        print("**************************\n")
        
        # Save Hv, hv, r, v_0, v as .mat file, and overwrite the previous one
        mdic = {
          "Hv_orig":Hv_DMDc, "hv_orig":hv_DMDc.reshape(-1,1), 
          "Hv_noRedund":Hv_noRedund, "hv_noRedund":hv_noRedund.reshape(-1,1), "Idx_noRedund":Idx_noRedund.reshape(-1,1),
          "r":(r.reshape(m,-1)-v_0).reshape(-1,1), "v":v_CG.reshape(-1,1), "v_0":v_0.reshape(-1,1), "y":y_fetch.reshape(-1,1)}
        savemat('FeasibleRegion_tidx_{:03}_Clock_{}.mat'.format(tidx, timeString),mdic)

        v_adj = (v_CG + v_0).reshape(m,)
        print("\n**************************", "\n**** CG summary Start ****","\nUnit = ", str(unit),", t = ", t, "\nProfile Selected = ", profile_id, "\nr = ", r, "\nv = ", v_adj, "\n***** CG summary End *****","\n**************************\n")
        print("HaoyuCGSummary, " + str(unit) + ", t, " + str(t) + ", Profile, " + str(profile_id) + 
        ", r, " + ', '.join([str(num) for num in r]) + ", v, " + ', '.join([str(num) for num in v_adj]))

        # # TODO: replace this line with the option from input file
        # simulateFMUDuringDispatch = True

        # Simulate the FMU using this v_adj, Update x_sys_internal, and keep record in v_hist and yp_hist within this hour
        # Shift input
        i_input = 0.
        for i in range(int(Tr_Update_sec/Tss)):
          # Shift the v components
          if (tidx>0) and (i_input < m) and (setpoints_shift_step >= 1) :
              v[math.floor(i_input)]=copy.deepcopy(v_adj[math.floor(i_input)])
              i_input += 1/setpoints_shift_step
          else:
            v = copy.deepcopy(v_adj)            
          
          # fetch v: real world values
          v_fetch = np.asarray(v).reshape(m,)
          
          # fetch y, x: real world values
          if self._unitInfo[unit]['OnlineMatrixUpdate']:
            y_fetch = np.asarray(fmu.getReal(vr_output)) # <class 'numpy.ndarray'>, y_fetch.shape = (p,)
            x_fetch = np.asarray(fmu.getReal(vr_state)) # <class 'numpy.ndarray'>, x_fetch.shape = (n,)
          else:
            y_fetch = (np.dot(C_d, x_fetch.reshape(-1,1)-x_0)+y_0).reshape(-1,)
            x_fetch = x_fetch
          # print("t=", t, "v_fetch", v_fetch.shape, v_fetch)
          # print("t=", t, "x_fetch", x_fetch.shape, x_fetch)
          # print("t=", t, "y_fetch", y_fetch.shape, y_fetch)
          # print("\n")
          
          # Save time, r, v, and y
          self._unitInfo[unit]['t_hist'].append(copy.deepcopy(t))         # Time
          self._unitInfo[unit]['v_hist'].append(copy.deepcopy(v_fetch))   # input v
          self._unitInfo[unit]['x_hist'].append(copy.deepcopy(x_fetch))   # state x
          self._unitInfo[unit]['y_hist'].append(copy.deepcopy(y_fetch))   # output y
          
          if self._unitInfo[unit]['OnlineMatrixUpdate']:
            # set the input. The input / state / output in FMU are real-world values
            for i in range(len(vr_input)):
              fmu.setReal([vr_input[i]], [v[i]]) # Note: fmu.setReal must take two lists as input
            # perform one step
            fmu.doStep(currentCommunicationPoint=t, communicationStepSize=Tss)
            # print("Wall Time = {}, ".format(datetime.now().strftime("%Y_%m_%d_%H:%M:%S:%f")),"FMU t = ",t)
          else:
            x_fetch = (np.dot(A_d, x_fetch.reshape(-1,1)-x_0) + np.dot(B_d, v_fetch.reshape(-1,1)-v_0)+x_0).reshape(-1,)
          
          # time increment
          t = t + Tss
        
        if self._unitInfo[unit]['OnlineMatrixUpdate']:
          # Collect data for DMDc
          winIdxStart = max(0, t_idx*int(Tr_Update_sec/Tss)-math.floor(window/2))
          winIdxEnd = t_idx*int(Tr_Update_sec/Tss)+math.floor(window/2)
          t_window = np.asarray(self._unitInfo[unit]['t_hist'][winIdxStart:winIdxEnd]).reshape(1,-1)
          v_window = np.asarray(self._unitInfo[unit]['v_hist'][winIdxStart:winIdxEnd]).reshape(-1,m).T
          x_window = np.asarray(self._unitInfo[unit]['x_hist'][winIdxStart:winIdxEnd]).T
          y_window = np.asarray(self._unitInfo[unit]['y_hist'][winIdxStart:winIdxEnd]).T 
          # print("Window Start = {}, Window End = {}".format(winIdxStart, winIdxEnd))
          # print("t_window=",t_window.shape,"\n", t_window) # (2, 180)
          # print("v_window=",v_window.shape,"\n", v_window) # (2, 180)
          # print("x_window=",x_window.shape,"\n", x_window) # (2, 180)
          # print("y_window=",y_window.shape,"\n", y_window) # (2, 180)
          # print(a)

        
          # Do the DMDc, and return ABCD matrices
          U1 = v_window[:,0:-1]-v_0; X1 = x_window[:, 0:-1]-x_0; X2 = x_window[:, 1:]-x_0; Y1 = y_window[:, 0:-1]-y_0

          # if U1 has any dimension unchanged within the window, DMDc will return A matrix with eigenvalue of 1.
          # Skip DMDc if no U1 has constant components.
          flagNoTransient = False
          for item in U1: 
            if abs(np.max(item) - np.min(item)) < self._unitInfo[unit]['NeighborSetpointsThreshold']: # if there is constant setpoint, cannot run DMDc
              flagNoTransient = True

          # Skip DMDc if a close by scheduling parameter already exists
          CloseParaExist = False; nearest_para=-1
          for item in self._unitInfo[unit]['para_list']:
            if np.linalg.norm(np.asarray(item).reshape(-1,)-v_window[:,-1].reshape(-1,)) < self._unitInfo[unit]['NeighborSetpointsThreshold']:
              CloseParaExist = True
              nearest_para = item

          # the condition related to initialization:
          initializationCond = False
          if self._unitInfo[unit]['StateVariableSelectionXML'] is None:
            if t_idx == len(LearningSetpoints[0]):
              initializationCond = True
          else:
            if t_idx == 0:
              initializationCond = True

          Do_DMDc = False
          if not flagNoTransient and not CloseParaExist and not initializationCond and self._unitInfo[unit]['OnlineMatrixUpdate']: # if transient found within this window
            Do_DMDc = True

          if Do_DMDc:
            Ad_Dc, Bd_Dc, Cd_Dc= fun_DMDc(X1, X2, U1, Y1, -1)
            # Dd_Dc = np.zeros((p,m))
            # append the A,B,C,D matrices to an list
            self._unitInfo[unit]['A_list'].append(Ad_Dc); 
            self._unitInfo[unit]['B_list'].append(Bd_Dc); 
            self._unitInfo[unit]['C_list'].append(Cd_Dc); 
            self._unitInfo[unit]['para_list'].append((U1[:,-1]+v_0.reshape(m,)).tolist()); 
            self._unitInfo[unit]['eig_A_list'].append(np.max(np.linalg.eig(Ad_Dc)[0]))
            self._unitInfo[unit]['tTran_list'].append(t-Tr_Update_sec)
            print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("&&& DMDc summary Start &&&")
            print("Unit =", str(unit), ", t_idx = ", t_idx, ", t = ", t-Tr_Update_sec, ", v_window[:,0] =", v_window[:,0], ", v_window[:,-1] =", v_window[:,-1])
            print("A_list=\n",self._unitInfo[unit]['A_list'])
            print("B_list=\n",self._unitInfo[unit]['B_list'])
            print("C_list=\n",self._unitInfo[unit]['C_list'])
            print("para_list=\n",self._unitInfo[unit]['para_list'])
            print("eig_A_list=\n",self._unitInfo[unit]['eig_A_list'])
            print("tTran_list=\n",self._unitInfo[unit]['tTran_list'])
            print("&&&& DMDc summary End &&&&")
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
            # print(a)
          else:
            print("\n&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("&&& DMDc was not done. &&&")
            print("t_idx = ", t_idx, ", t = ", t-Tr_Update_sec, ", v_window[:,0] =", v_window[:,0], ", v_window[:,-1] =", v_window[:,-1])
            if flagNoTransient:
              print("Reason: Transient is too small. v_window[:,0] =", v_window[:,0], ", v_window[:,-1] =", v_window[:,-1])
            if initializationCond:
              print("Reason: System is initialized at t =",t-Tr_Update_sec)
            if CloseParaExist: 
              print("Reason: New parameter is too close to existing parameter [{}].".format(nearest_para))
              print("New parameter =", v_window[:,-1].tolist(), "Para_list =",self._unitInfo[unit]['para_list'])
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
        
        t_idx = t_idx + 1

        # Initialize an "err_TF" for this time step
        err_TF = False
        
        # loop through the <Component> items in HERON, looking for errors
        for comp, info in dispatch._resources.items():
          # e.g. comp= <HERON Component "BOP""> <HERON Component "SES"">

          for tracker in comp.get_tracking_vars():
            # loop through the resources in info (only one resource here - electricity)
            for res in info:
              if str(res) == "electricity":
                for i in range(m):
                  # find the index in v_adj for this comp
                  if inputNameSeq[i] in str(comp):
                    r_comp = r_spec[i]
                    
                    if comp.get_interaction().is_type('Storage'):
                      Allowed_Level = allowedLevel[i] - v_adj[i]*Tr_Update_hrs # Allowed_Level: predicted level due to v_value
                      allowedLevel[i] = Allowed_Level
                      V1 = float(Allowed_Level)
                      
                    else: # not storage
                      V1 = float(v_adj[i])

                    # Write up any violation to the errs:
                    if abs(r_comp - V1) > max(self.abs_tolerance, self.rel_tolerance*max(abs(r_comp),abs(V1))):
                      # violation
                      err_TF = True

        # loop through the <Component> items in HERON, logging errors
        for comp, info in dispatch._resources.items():
          # e.g. comp= <HERON Component "BOP""> <HERON Component "SES"">

          for tracker in comp.get_tracking_vars():
            # loop through the resources in info (only one resource here - electricity)
            for res in info:
              if str(res) == "electricity":
                for i in range(m):
                  # find the index in v_adj for this comp
                  if inputNameSeq[i] in str(comp):
                    r_comp = r_spec[i]
                    
                    if comp.get_interaction().is_type('Storage'):
                      Allowed_Level = allowedLevel[i] - v_adj[i]*Tr_Update_hrs # Allowed_Level: predicted level due to v_value
                      allowedLevel[i] = Allowed_Level
                      V1 = float(Allowed_Level)
                      
                    else: # not storage
                      V1 = float(v_adj[i])

                    # Write up any violation to the errs:
                    if err_TF:
                      # violation, log errors for all the components
                      errs.append({'msg': f'Reference Governor Violation',
                                  'limit': V1,
                                  'limit_type': 'lower' if (r_comp < V1) else 'upper',
                                  'component': comp,
                                  'resource': res,
                                  'time': time,
                                  'time_index': tidx,
                                  })
      # Get time for dispatching
      endTimeDispatch = copy.deepcopy(datetime.now())
      print('Haoyu t-debug, Time for this Dispatch is {}'.format(endTimeDispatch-startTime))
    print(errs)

    if errs == []: # if no validation error:
      print(" ")
      print("*********************************************************************")
      print("*** Haoyu Debug, Validation Success, Print for offline processing ***")
      print("*********************************************************************")
      print(" ")

      for unit in self._unitInfo:
        # print(self._unitInfo[unit]['t_hist'])
        # print(self._unitInfo[unit]['v_hist'])
        # print(self._unitInfo[unit]['x_hist'])
        # print(len(self._unitInfo[unit]['y_hist']))
        t_hist = np.array(self._unitInfo[unit]['t_hist']).T
        v_hist = np.array(self._unitInfo[unit]['v_hist']).T
        x_hist = np.array(self._unitInfo[unit]['x_hist']).T
        y_hist = np.array(self._unitInfo[unit]['y_hist']).T
        print(str(unit), t_hist.shape, v_hist.shape, x_hist.shape, y_hist.shape)

        # for i in range(len(t_hist)):
        #   print("HaoyuDispatchSummary,", str(unit), ",t,", t_hist[i], 
        #   ",v0,", v_hist[0][i], ",v1,", v_hist[1][i], ",v2,", v_hist[2][i], 
        #   ",x0,", x_hist[0][i], ",x1,", x_hist[1][i], ",x2,", x_hist[2][i], 
        #   ",y0,",y_hist[0][i], ",y0min,",self._unitInfo[unit]['Targets_Min'][0],",y0max,",self._unitInfo[unit]['Targets_Max'][0], 
        #   ",y1,",y_hist[1][i], ",y1min,",self._unitInfo[unit]['Targets_Min'][1],",y1max,",self._unitInfo[unit]['Targets_Max'][1], 
        #   ",y2,",y_hist[2][i], ",y2min,",self._unitInfo[unit]['Targets_Min'][2],",y2max,",self._unitInfo[unit]['Targets_Max'][2], 
        #   ",y3,",y_hist[3][i], ",y3min,",self._unitInfo[unit]['Targets_Min'][3],",y3max,",self._unitInfo[unit]['Targets_Max'][3], 
        #   ",y4,",y_hist[4][i], ",y4min,",self._unitInfo[unit]['Targets_Min'][4],",y4max,",self._unitInfo[unit]['Targets_Max'][4], 
        #   ",y5,",y_hist[5][i], ",y5min,",self._unitInfo[unit]['Targets_Min'][5],",y5max,",self._unitInfo[unit]['Targets_Max'][5])

        for i in range(len(t_hist)):
          print("HaoyuDispatchSummary, " + str(unit) + 
          ", t, " + str(t_hist[i]) + 
          ", v, " + ', '.join([str(num) for num in v_hist[:,i]]) +
          ", x, " + ', '.join([str(num) for num in x_hist[:,i]]) +                                                           
          ", y, " + ', '.join([str(num) for num in y_hist[:,i]]) + 
          ", ymin, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Min']]) +
          ", ymax, " + ', '.join([str(num) for num in self._unitInfo[unit]['Targets_Max']]))

    return errs

def read_parameterized_XML(MatrixFileName):
  tree = ET.parse(MatrixFileName)
  root = tree.getroot()
  para_array = []; UNorm_list = []; XNorm_list = []; XLast_list = []; YNorm_list =[]
  A_Re_list = []; B_Re_list = []; C_Re_list = []; A_Im_list = []; B_Im_list = []; C_Im_list = []
  for child1 in root:
    for child2 in child1:
      for child3 in child2:
        if child3.tag == 'acturators':
          actuatorList = child3.text.split(' ')
        if child3.tag == 'states' or child3.tag == 'stateVariables':
          stateList = child3.text.split(' ')
        if child3.tag == 'outputs':
          outputList = child3.text.split(' ')
        

        if child3.tag == 'dmdTimeScale':
          Temp_txtlist = child3.text.split(' ')
          Temp_floatlist = [float(item) for item in Temp_txtlist]
          TimeScale = np.asarray(Temp_floatlist)
          TimeInterval = TimeScale[1]-TimeScale[0]
        if child3.tag == 'UNorm':
            for child4 in child3:
              parameter_list=[]
              for i_sp in range(len(list(child4.attrib.keys()))-1):
                key = 'schedulingParameter_%d'%(i_sp)
                parameter_list.append(float(child4.get(key)))
              # parameter_list = [float(x) for x in list(child4.attrib.values())]
              para_array.append(parameter_list)
              Temp_txtlist = child4.text.split(' ')
              Temp_floatlist = [float(item) for item in Temp_txtlist]
              UNorm_list.append(np.asarray(Temp_floatlist))
            # para_array = np.asarray(para_array)[:,0:-1]
            para_array = np.asarray(para_array)

        if child3.tag == 'XNorm':
          for child4 in child3:
            Temp_txtlist = child4.text.split(' ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            XNorm_list.append(np.asarray(Temp_floatlist))

        if child3.tag == 'XLast':
          for child4 in child3:
            Temp_txtlist = child4.text.split(' ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            XLast_list.append(np.asarray(Temp_floatlist))

        if child3.tag == 'YNorm':
          for child4 in child3:
            Temp_txtlist = child4.text.split(' ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            YNorm_list.append(np.asarray(Temp_floatlist))

        for child4 in child3:
          for child5 in child4:
            if child5.tag == 'real':
              Temp_txtlist = child5.text.split(' ')
              Temp_floatlist = [float(item) for item in Temp_txtlist]
              if child3.tag == 'Atilde':
                A_Re_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Btilde':
                B_Re_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Ctilde':
                C_Re_list.append(np.asarray(Temp_floatlist))

            if child5.tag == 'imaginary':
              Temp_txtlist = child5.text.split(' ')
              Temp_floatlist = [float(item) for item in Temp_txtlist]
              if child3.tag == 'Atilde':
                A_Im_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Btilde':
                B_Im_list.append(np.asarray(Temp_floatlist))
              if child3.tag == 'Ctilde':
                C_Im_list.append(np.asarray(Temp_floatlist))

  n = len(XNorm_list[0]) # dimension of x
  m = len(UNorm_list[0]) # dimension of u
  p = len(YNorm_list[0]) # dimension of y

  # Reshape the A, B, C lists
  for i in range(len(para_array)):
      A_Re_list[i]=np.reshape(A_Re_list[i],(n,n)).T
      A_Im_list[i]=np.reshape(A_Im_list[i],(n,n)).T
      B_Re_list[i]=np.reshape(B_Re_list[i],(m,n)).T
      B_Im_list[i]=np.reshape(B_Im_list[i],(m,n)).T
      C_Re_list[i]=np.reshape(C_Re_list[i],(n,p)).T
      C_Im_list[i]=np.reshape(C_Im_list[i],(n,p)).T

  A_list = A_Re_list
  B_list = B_Re_list
  C_list = C_Re_list

  eig_A_array=[]
  # eigenvalue of A
  for i in range(len(para_array)):
      w,v = np.linalg.eig(A_list[i])
      eig_A_array.append(abs(max(w)))
  eig_A_array = np.asarray(eig_A_array)

  return TimeInterval, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array, actuatorList, stateList, outputList

def check_YNorm_within_Range(y_min, y_max, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array):
  UNorm_list_ = []; XNorm_list_ = []; XLast_list_ = []; YNorm_list_ =[]
  A_list_ = []; B_list_ = []; C_list_ = []; para_array_ = []; eig_A_array_ =[]

  for i in range(len(YNorm_list)):
    state = True
    for j in range(len(YNorm_list[i])):
      if YNorm_list[i][j] < y_min[j] or YNorm_list[i][j] > y_max[j]:
        state = False
    if state == True:
      UNorm_list_.append(UNorm_list[i])
      XNorm_list_.append(XNorm_list[i])
      XLast_list_.append(XLast_list[i])
      YNorm_list_.append(YNorm_list[i])
      A_list_.append(A_list[i])
      B_list_.append(B_list[i])
      C_list_.append(C_list[i])
      para_array_.append(para_array[i])
      eig_A_array_.append(eig_A_array[i])

  para_array_ = np.asarray(para_array_); eig_A_array_ = np.asarray(eig_A_array_)
  return para_array_, UNorm_list_, XNorm_list_, XLast_list_, YNorm_list_, A_list_, B_list_, C_list_, eig_A_array_

def fun_MOAS_noinf(*argv):
  # A, B, C, D, s, g, delay_steps
  A = argv[0]; B = argv[1]; C = argv[2]; D = argv[3]; s = argv[4]; g = argv[5]
  if len(argv) < 7:
    delay_steps = 0
  else:
    delay_steps = int(argv[6])

  p = len(C)  # dimension of y
  T = np.linalg.solve(np.identity(len(A))-A, B)
  """ Build the S matrix"""
  S = np.zeros((2*p, p))
  for i in range(0,p):
    S[2*i, i] = 1.0
    S[2*i+1, i] = -1.0
  Kx = np.dot(S,C)
  # print("Kx", Kx)
  Lim = np.dot(S,(np.dot(C,T) + D))
  # print("Lim", Lim)
  Kr = np.dot(S,D)
  # print("Kr", Kr)
  """ Build the infinity term of H and h """
  # H = np.concatenate((0*Kx, Lim),axis=1); h = s
  
  """ Build the t=0 term of H and h """
  i = 0
  NewBlock = np.concatenate((Kx, Kr),axis=1)
  H = NewBlock; h = s

  """ Build the add-on blocks of H and h """
  while i < (g+delay_steps) :
    i = i + 1
    Kx = np.dot(Kx, A)
    Kr = Lim - np.dot(Kx,T)
    NewBlock = np.concatenate((Kx,Kr), axis=1)
    H = np.concatenate((H,NewBlock)); h = np.concatenate((h,s))
  
  """ Remove the first rows according to the "delay_steps" """
  if not delay_steps == 0:
    for i in range(2*p*delay_steps):
      H = np.delete(H, 0, 0)
      h = np.delete(h, 0, 0)
  # print(H.shape,"\n", H)
  # print(h.shape,"\n", h)

  return H, h

def fun_MOAS_MIMO_Setpoint_Shift(*argv):
  # A, B, C, D, s, g, v0, setpoints_shift_step, delay_steps
  A = argv[0]; B = argv[1]; C = argv[2]; D = argv[3]; s = argv[4]; g = argv[5]; v0 = argv[6]; setpoints_shift_step = argv[7]
  if len(argv) < 9:
    delay_steps = 0
  else:
    delay_steps = int(argv[8])
  n,m = B.shape
  p = len(C)  # dimension of y
  # print(n,m,p) # 4 2 4
  """ Build the S matrix"""
  S = np.zeros((2*p, p))
  for i in range(0,p):
    S[2*i, i] = 1.0
    S[2*i+1, i] = -1.0
  
  """ Build R and L matrix """
  R = np.zeros((g+delay_steps+1,m,m))
  L = np.zeros((g+delay_steps+1,m,m))
  for i in range(g+delay_steps+1):
    L[i,:,:]=np.identity(m)
    if i < (m-1)*setpoints_shift_step:
      for j in range(math.floor(i/setpoints_shift_step)+1,m):
        R[i,j,j]=1
        L[i,j,j]=0
    # print("R[{}]=".format(i),"\n",R[i])
    # print("L[{}]=".format(i),"\n",L[i])
  
  """ Build the initial term of H and h """
  i = 0
  Kx = np.dot(S,C)
  observeOperatorList=[np.dot(S, D)]
  Kr = np.dot(observeOperatorList[0],L[0,:,:])
  Kv0= np.dot(observeOperatorList[0],R[0,:,:]) 
  NewBlock = np.concatenate((Kx,Kr), axis=1)
  NewLine = s - np.dot(Kv0,v0)
  H = NewBlock; h = NewLine
  
  """ Build each add-on blocks of H and h """

  while i < (g+delay_steps):
    i = i + 1
    observeOperatorList.append(np.dot(Kx,B))
    Kx = np.dot(Kx, A)

    Kr = np.zeros((2*p,m))
    Kv0 = np.zeros((2*p,m))
    for j in range(i+1):
      Kr = Kr + np.dot(observeOperatorList[j],L[(i-j),:,:])
      Kv0= Kv0+ np.dot(observeOperatorList[j],R[(i-j),:,:])       

    NewBlock = np.concatenate((Kx,Kr), axis=1)
    NewLine = s - np.dot(Kv0,v0)
    H = np.concatenate((H,NewBlock)); h = np.concatenate((h,NewLine))
      
  """ Remove the first rows according to the "delay_steps" """
  if not delay_steps == 0:
    for i in range(2*p*delay_steps):
      H = np.delete(H, 0, 0)
      h = np.delete(h, 0, 0)
  
  # print(H.shape,"\n", H)
  # print(h.shape,"\n", h)
  return H, h

def fun_RG_SISO(v_0, x, r, H, h, p):
  n = len(x) # dimension of x
  x = np.vstack(x) # x is horizontal array, must convert to vertical for matrix operation
  # because v_0 and r are both scalar, so no need to vstack
  Hx = H[:, 0:n]; Hv = H[:, n:]
  alpha = h - np.dot(Hx,x) - np.dot(Hv,v_0) # alpha is the system remaining vector
  beta = np.dot(Hv, (r-v_0)) # beta is the anticipated response vector with r

  kappa = 1
  for k in range(0,len(alpha)):
    if 0 < alpha[k] and alpha[k] < beta[k]:
      kappa = min(kappa, alpha[k]/beta[k])
    else:
      kappa = kappa
  v = np.asarray(v_0 + kappa*(r-v_0)).flatten()

  return v

def fun_2nd_gstep_calc(x, Hm, hm, A_m, B_m, g):
  n = len(x) # dimension of x
  # x = np.vstack(x) # x is horizontal array, must convert to vertical for matrix operation
  # because v_0 and r are both scalar, so no need to vstack
  Hxm = Hm[:, 0:n]; Hvm = Hm[:, n:]

  T = np.linalg.solve(np.identity(n)-A_m, B_m)
  Ag = np.identity(n)
  for k in range(g+1):
      Ag = np.dot(Ag,A_m)

  alpha = hm - np.dot(Hxm, np.dot(Ag, np.vstack(x)))
  beta = np.dot(Hxm, np.dot((np.identity(n)-Ag),T))
  v_st = []; v_bt = []
  for k in range(0,len(alpha)):
    if beta[k]>0:
      v_st.append(alpha[k]/beta[k])
    elif beta[k]<0:
      v_bt.append(alpha[k]/beta[k])

  v_max = np.asarray(min(v_st))
  v_min = np.asarray(max(v_bt))
  return v_max, v_min

def fun_CG_MIMO(r, Hv, hv):
  # print("x=", x, "\nr=", r)
  # print("x=", x, "\nr=", r, "\nH=", H, "\nh=", h)
  m = len(r)
  r = np.vstack(r).reshape(m,) # first convert r to vertical, then reshape to 1 dimensional vector
  # linear equality for electrical power dispatch problem
  Cv = np.ones(m);  dv = np.sum(r)
  # define the P matrix for quad form v.T * P * v
  P = np.identity(m)
  
  """ CVXPY solvers """
  # # define v as a cp variable of length m
  # v = cp.Variable(m)
  # # Define objective
  # objective = cp.Minimize((1/2)*cp.quad_form(v, P) - r.T @ v)
  # # Define constraints
  # constriants = [Hv @ v <= hv, Cv @ v == dv] # with explicit constraints (sum of power shouldn't be changed)
  # # constriants = [Hv @ v <= hv] # no explicit constraints, all the dimensions of v are free to be adjusted  
  # # Assemble problem
  # prob = cp.Problem(objective, constriants)   
  # # solve with solver specifications  
  # # prob.solve(solver=cp.OSQP, verbose=False) # occasionally works
  # # prob.solve(solver=cp.OSQP, verbose=False, eps_abs=1.0e-2, eps_rel=1.0e-2) # This works, but with suboptimal solution!  
  # prob.solve(solver=cp.ECOS, verbose=False) # This also works
  # # prob.solve(solver=cp.ECOS, verbose=False, abstol=1.0e-8, reltol=1.0e-8) # This also works
  # # prob.solve(solver=cp.SCS, verbose=False) # This also works
  # v = np.asarray(v.value).reshape(-1,1)
  # # print(cp.installed_solvers())

  """ CVXOPT solvers """
  # # make sure P is symmetric
  # cvx_P = (.5 * (P + P.T)).astype(np.double)
  # cvx_q = -1*r.astype(np.double) 
  # # add P and r to args
  # cvx_Args = [cvxopt.matrix(cvx_P), cvxopt.matrix(cvx_q)] 
  # # add Hv and hv to args (Hv*v<hv)
  # cvx_Args.extend([cvxopt.matrix(Hv.astype(np.double)), cvxopt.matrix(hv.astype(np.double))])
  # # add Cv and dv to args (Cv*v=dv)
  # cvx_Args.extend([cvxopt.matrix(Cv.reshape(1,-1).astype(np.double)), cvxopt.matrix(dv.astype(np.double))])
  # cvx_Sol = cvxopt.solvers.qp(*cvx_Args)
  # # if 'optimal' not in sol['status']:
  # #     return None
  # v = np.array(cvx_Sol['x']).reshape(-1,1)

  """ quadprog solvers """
  # # make sure P is symmetric
  # qp_P = (.5 * (P + P.T)).astype(np.double)
  # qp_q = r.astype(np.double) 
  # # with explicit constraints (sum of power shouldn't be changed)
  # qp_A = Cv.reshape(1,-1);  qp_b = dv.reshape(-1,)
  # qp_C = -1*np.vstack([qp_A, Hv]).T.astype(np.double);  qp_d = -1*np.hstack([qp_b, hv]).astype(np.double)
  # meq = qp_A.shape[0]
  # # # no equality constraint
  # # qp_C = -Hv.T.astype(np.double); qp_d = -hv.astype(np.double)
  # # meq = 0
  # v= quadprog.solve_qp(qp_P, qp_q, qp_C, qp_d, meq)[0].reshape(-1,1)

  """ qpsolvers """
  qps_P = P.astype(np.double)
  qps_q = -r.astype(np.double)
  qps_G = Hv.astype(np.double)
  qps_h = hv.astype(np.double)
  qps_A = Cv.reshape(1,-1).astype(np.double)
  qps_b = dv.reshape(-1,).astype(np.double)

  # try a series of solvers, in order to ensure solution
  try:
    v = qpsolvers.solve_qp(qps_P, qps_q, qps_G, qps_h, qps_A, qps_b, solver='cvxopt').reshape(-1,1) # cvxopt uses interior point algorithm
    print("QP Solution found by solver: cvxopt")
  except:
    try:
      v = qpsolvers.solve_qp(qps_P, qps_q, qps_G, qps_h, qps_A, qps_b, solver='ecos').reshape(-1,1) # ecos uses interior point algorithm
      print("QP Solution found by solver: ecos")
    except:
      try:
        v = qpsolvers.solve_qp(qps_P, qps_q, qps_G, qps_h, qps_A, qps_b, solver='daqp').reshape(-1,1) # daqp uses active set algorithm
        print("QP Solution found by solver: daqp")
      except:
        try:
          v = qpsolvers.solve_qp(qps_P, qps_q, qps_G, qps_h, qps_A, qps_b, solver='quadprog').reshape(-1,1) # quadprog uses active set algorithm
          print("QP Solution found by solver: quadprog")
        except:
          raise RuntimeError("Unable to find a solution v within the feasible region using the following QP solvers: cvxopt, ecos, daqp, quadprog.")
  
  return v
    
def computeTruncatedSingularValueDecomposition(X, truncationRank, full = False, conj = True):
  """
  Compute Singular Value Decomposition and truncate it till a rank = truncationRank
  @ In, X, numpy.ndarray, the 2D matrix on which the SVD needs to be performed
  @ In, truncationRank, int or float, optional, the truncation rank:
                                                  * -1 = no truncation
                                                  *  0 = optimal rank is computed
                                                  *  >1  user-defined truncation rank
                                                  *  >0. and < 1. computed rank is the number of the biggest sv needed to reach the energy identified by truncationRank
  @ In, full, bool, optional, compute svd returning full matrices
  @ In, conj, bool, optional, compute conjugate of right-singular vectors matrix)
  @ Out, (U, s, V), tuple of numpy.ndarray, (left-singular vectors matrix, singular values, right-singular vectors matrix)
  """
  U, s, V = np.linalg.svd(X, full_matrices=full)
  V = V.conj().T if conj else V.T

  if truncationRank == 0:
    omeg = lambda x: 0.56 * x**3 - 0.95 * x**2 + 1.82 * x + 1.43
    rank = np.sum(s > np.median(s) * omeg(np.divide(*sorted(X.shape))))
  elif truncationRank > 0 and truncationRank < 1:
    rank = np.searchsorted(np.cumsum(s / s.sum()), truncationRank) + 1
  elif truncationRank >= 1 and isinstance(truncationRank, int):
    rank = min(truncationRank, U.shape[1])
  else:
    rank = U.shape[1]
  U = U[:, :rank]
  V = V[:, :rank]
  s = np.diag(s)[:rank, :rank] if full else s[:rank]
  return U, s, V

def fun_DMDc(X1, X2, U, Y1, rankSVD):
  """
      Evaluate the the matrices (A and B tilde)
      @ In, X1, np.ndarray, n dimensional state vectors (n*L)
      @ In, X2, np.ndarray, n dimensional state vectors (n*L)
      @ In, U, np.ndarray, m-dimension control vector by L (m*L)
      @ In, Y1, np.ndarray, m-dimension output vector by L (y*L)
      @ In, rankSVD, int, rank of the SVD
      @ Out, A, np.ndarray, the A matrix
      @ Out, B, np.ndarray, the B matrix
      @ Out, C, np.ndarray, the C matrix
  """
  n = len(X2)
  # Omega Matrix, stack X1 and U
  omega = np.concatenate((X1, U), axis=0)
  # SVD
  Utsvd, stsvd, Vtsvd = computeTruncatedSingularValueDecomposition(omega, rankSVD, False, False)
  # print(stsvd)
  # print(Utsvd)

  # Find the truncation rank triggered by "Maximum Condition Number"
  rank_s = sum(map(lambda x : x>=np.max(stsvd)*1e-9, stsvd.tolist()))
  # print(rank_s)
  if rank_s < Utsvd.shape[1]:
    Ut = Utsvd[:, :rank_s]
    Vt = Vtsvd[:, :rank_s]
    St = np.diag(stsvd)[:rank_s, :rank_s]
  else:
    Ut = Utsvd
    Vt = Vtsvd
    St = np.diag(stsvd)

  # print('Ut',Ut.shape)
  # print('St',St.shape, "\n", St)
  # print('Vt',Vt.shape)

  # QR decomp. St=Q*R, Q unitary, R upper triangular
  Q, R = np.linalg.qr(St)
  # if R is singular matrix, raise an error
  if np.linalg.det(R) == 0:
    raise RuntimeError("The R matrix is singlular, Please check the singularity of [X1;U]!")
  beta = X2.dot(Vt).dot(np.linalg.inv(R)).dot(Q.T)
  A = beta.dot(Ut[0:n, :].T)
  B = beta.dot(Ut[n:, :].T)
  C = Y1.dot(scipy.linalg.pinv(X1))

  return A, B, C

def noredundant(A,b,baseLength):
  zero_threshold = 1e-9
  rowA, colA = A.shape
  
  A_cumulate = [A[0:baseLength,:]]; b_cumulate = [b[0:baseLength]]; eqnId_cumulate = np.arange(0,baseLength,1).tolist(); 
  vertices = []; ver_eqs = []
  # first step: try to find the vertices using the first baseLength rows in A and b (the first section in MOAS)
  lst = np.arange(0,baseLength,1)

  for combo in combinations(lst,colA):
    # i,j = combo

    tempA = np.asarray([A[i,:] for i in combo]).reshape(-1,colA)
    tempb = np.asarray([b[i] for i in combo]).reshape(-1,)
    try:
      x_vertex = np.linalg.solve(tempA,tempb)
        
    except np.linalg.LinAlgError as err:
      if 'Singular matrix' in str(err):
        x_vertex = None
    # print("\nsolve_test, i={}, j={} \n tempA={}, \n tempb={}, \n x_vertex={}".format(i,j,tempA,tempb,x_vertex))
    

    # calculate this A*x_vertex-b:
    if x_vertex is not None:
      rslt = np.dot(np.asarray(A_cumulate).reshape(-1,colA),x_vertex)-np.asarray(b_cumulate).reshape(-1,)
      # print(" rslt={}, max={}, min={}".format(rslt.shape, max(rslt), min(rslt)), "\n", rslt)
      # time.sleep(1)
      
      # add x_vertex and associated equations to list
      if max(rslt)<=zero_threshold :
        vertices.append(x_vertex)
        ver_eqs.append([i for i in combo])
        # time.sleep(1)
  # re-count the unique equations that 
  eqnId_cumulate = list(set(np.asarray(ver_eqs).reshape(-1,).tolist()))
  A_cumulate = []; b_cumulate = []
  for item in eqnId_cumulate:
    A_cumulate.append(A[item,:]); b_cumulate.append(b[item])

  # print(vertices, ver_eqs, eqnId_cumulate,"\n") 
  # print(a)

  # second step: adding inequalities line by line, see if the newline is redundant
  i_eqn = baseLength - 1
  while i_eqn < rowA-1:
    # initialize the vertices and associated equations to be deleted
    flagAdd = False; vertices_to_add=[]; ver_eqs_to_add=[]; vertices_to_keep=[]; ver_eqs_to_keep=[]
    # fetch the new line of inequality
    i_eqn += 1
    A_test = A[i_eqn,:]; b_test = b[i_eqn]
    # print("Test, i_eqn={}\n A_test={}, b_test={}".format(i_eqn, A_test, b_test))
    # time.sleep(1)
    # loop through all the existing vertices, see if they are outside the box
    for i_ver in range(len(vertices)):
      rslt = np.dot(np.asarray(A_test).reshape(-1,colA), vertices[i_ver])-np.asarray(b_test).reshape(-1,)
      # print(" vertex={}, ver_eqs={}, max={}".format(vertices[i_ver], ver_eqs[i_ver], min(rslt)))
      
      if max(rslt) > zero_threshold: # if an existing vertice is outside the domain of A_test*x-b_test<0
        flagAdd = True; 
        # find the new vertices between this new line and the two equations of the vertices to be deleted
        for combo in combinations(ver_eqs[i_ver],(colA-1)):
          tempA = np.concatenate((np.asarray(A_test).reshape(-1,colA), np.asarray([A[j,:] for j in combo]).reshape(-1,colA)),axis=0)
          tempb = np.concatenate((np.asarray(b_test).reshape(-1,), np.asarray([b[j] for j in combo]).reshape(-1,)),axis=0)
          try:
            x_vertex = np.linalg.solve(tempA,tempb)
          except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
              x_vertex = None
          # print("\n Solve_Trial, i_eqn={}, j={} \n tempA={}, \n tempb={}, \n x_vertex={}".format(i_eqn,combo,tempA,tempb,x_vertex))
          # time.sleep(1)
          if x_vertex is not None:
            # add x_vertex and associated equations to list
            rslt_temp = np.dot(np.asarray(A_cumulate).reshape(-1,colA),x_vertex)-np.asarray(b_cumulate).reshape(-1,)
            if max(rslt_temp) < zero_threshold:
              vertices_to_add.append(x_vertex)
              ver_eqs_to_add.append([i_eqn] + [j for j in combo])
      else: # if this vertice is still in the box, need to keep
        vertices_to_keep.append(vertices[i_ver])
        ver_eqs_to_keep.append(ver_eqs[i_ver])

    # print(i_eqn, len(vertices_to_keep), len(vertices_to_add))
    # join the two lists
    vertices = vertices_to_add + vertices_to_keep
    ver_eqs = ver_eqs_to_add + ver_eqs_to_keep
    # print("Number of vertices = {}, Number of equations = {}, ver_eqs={}".format(len(vertices), len(ver_eqs),ver_eqs))
    # time.sleep(1)

    # re-generate the eqnId_cumulate
    eqnId_cumulate = list(set(np.asarray(ver_eqs).reshape(-1,).tolist()))
    A_cumulate = []; b_cumulate = []
    for item in eqnId_cumulate:
        A_cumulate.append(A[item,:]); b_cumulate.append(b[item])
  
  # Third Step: re-shape the A_cumulate and b_cumulate
  A_cumulate = np.asarray(A_cumulate).reshape(-1,colA)
  b_cumulate = np.asarray(b_cumulate).reshape(-1,)
  print("no-redundant A shape = {}, b shape = {}".format(A_cumulate.shape, b_cumulate.shape))

  return A_cumulate,b_cumulate,vertices

def noRedund(A,b,Idx):
  # A is a two-dimensional ndarray, shape = (N,nx)
  # b is a one-dimensional ndarray, shape = (N,)
  # A and b defines the linear constraints for vector x:
  #       Ax <= b
  # NOTES: (1) Unbounded feasible regions are permitted.
  #        (2) This program requires that the feasible region have some
  #            finite extent in all dimensions. For example, the feasible
  #            regions cannot be a line segment in 2-D space, or a plane
  #            in 3-D space.
  #        (3) At least two dimensions are required.
  #        (4) See function CON2VERT which is limited to bounded feasible
  #            regions but also outputs vertices for the region.
  #        (5) Written in MATLAB by Michael Kleder, June 2005; 
  #            Implemented to python by Haoyu Wang, August 2023.

  # first, attempt to locate a feasible point "c" that satisfies Ac<=b:
  c,_residuals,_rank,_singularValues = np.linalg.lstsq(A,b,rcond=None)
  # print("A rank = ",np.linalg.matrix_rank(A))
  # print("c=", c)
  
  # if the "c" provided by least square solver is not "fully" inside Ac<b domain:
  if not all(np.dot(A,c)<b):
    c, fopt, _iter, _funcalls, warnflag = fmin(obj, c, args=(A,b), disp=False, full_output=True)
    if warnflag !=0:
      raise RuntimeError("Unable to locate a point within the interior of a feasible region.")
  
  # move polytope to contain origin
  bk = b; # preserve
  b = b - np.dot(A,c); # polytope A*x <= b now includes the origin

  # obtain dual polytope vertices
  B = np.tile(b.reshape(-1,1), A.shape[1])
  D = np.divide(A,B)
  
  # generate a convex hull
  hull = ConvexHull(D)
  
  # record which constraints generate points on the convex hull
  nr = np.unique(hull.simplices.flatten())
  
  An=A[nr]
  bn=bk[nr]
  Idxn = Idx[nr]

  return An, bn, Idxn

def obj(c,*args):
  # unload A and b from input arguments
  A=args[0]
  b=args[1]
  
  # calculate residuals
  d = np.dot(A,c)-b
  # exclude quasi-boundary points
  k=(d>=-1e-15); 
  # print(k)
  d[k]=d[k]+1
  # return 0 if each element of d is smaller than zero
  # otherwise return the max element of d
  d = max(np.append(d,0))
  # print(d)
  return d