# Copyright 2021 UChicago Argonne, LLC
# Author:
# - Haoyu Wang and Roberto Ponciroli, Argonne National Laboratory
# - Andrea Alfonsi, Idaho National Laboratory

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#   https://www.apache.org/licenses/LICENSE-2.0.txt

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
  Author:  H. Wang
  Date  :  07/30/2020
"""
from __future__ import division, print_function , unicode_literals, absolute_import

#External Modules---------------------------------------------------------------
import numpy as np
import os
import math
from scipy import signal
from scipy import io
from scipy.interpolate import interp1d
from datetime import datetime
import csv
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End-----------------------------------------------------------


class RefGov_parameterized_SIMO(ExternalModelPluginBase):
  # External Model plugin class, Reference Governor
  #################################
  #### RAVEN API methods BEGIN ####
  #################################
  def _readMoreXML(self, container, xmlNode):
    """
      Method to read the portion of the XML that belongs to this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, xmlNode, xml.etree.ElementTree.Element, XML node that needs to be read
      @ Out, None
    """
    """ Initialization of 4 entries """
    container.constants = {}

    # Extract the output Variable name: container.outputVariables = ['V1', 'V1min', 'V1max']
    outputVarsNode    = xmlNode.find("outputVariables")
    # print(outputVarsNode)
    if outputVarsNode is None: # if cannot find the "outputVariables" tag, return error
      raise IOError("RG Plugin: <outputVariables> XML block must be inputted!")
    # container.outputVariables = outputVarsNode.text.strip()
    container.outputVariables = [var.strip() for var in outputVarsNode.text.split(",")]
    # print(container.outputVariables) # ['V1', 'V1min', 'V1max']

    for child in xmlNode:
      # print(child.tag)
      # xmlNode is the Nodes within the section of <ExternalModel>.
      # child.tag are the strings containing each node name. child.tag == child.tag.strip()
      if child.tag.strip() == "variables":
        # get verbosity if it exists
        # Extract Variable names: container.variables = ['Vi', 'Pi']
        container.variables = [var.strip() for var in child.text.split(",")]
        # print(container.variables) # ['V1', 'V1min', 'V1max', 'P1']
        # if container.outputVariable not in container.variables:
        #   raise IOError("RG Plug-in: "+container.outputVariable+" variable MUST be present in the <variables> definition!")

      container.constants['Sys_State_x']=[] # place holder
      if child.tag.strip() == "constant":
        # Extract the constant names and their values: container.constants = {'TimeInterval': 3600.0}
        # varName has to be provided in the <constant> entry
        if "varName" not in child.attrib:
          raise IOError("RG Plug-in: attribute varName must be present in <coefficient> XML node!")
        # extract the system state variable, the only vector
        if child.attrib['varName'] == "Sys_State_x":
          container.constants['Sys_State_x'] = [float(var.strip()) for var in child.text.split(",")]
        else:
          container.constants[child.attrib['varName']] = float(child.text)
    # print(container.constants) # {'TimeInterval': 3600.0}
    # print(a)

    Min_counter = 0; Max_counter = 0
    for key, value in container.constants.items(): # count the inputs
      if key.startswith('Min_Target'):
        Min_counter += 1
        # print(Min_counter,key)
      elif key.startswith('Max_Target'):
        Max_counter += 1
        # print(Max_counter, key)
    # print(Min_counter, Max_counter)

    container.RG_Min_Targets = []
    container.RG_Max_Targets = []

    if Min_counter ==0 or Max_counter ==0: # check if Min/Max entry exists
      raise IOError("RG Plug-in: Missing 'Min_Target' or 'Max_Target' inputs!")
    else:
      if Min_counter != Max_counter: # check if Min and Max have the same length
        raise IOError("RG Plug-in: 'Min_Target' and 'Max_Target' are different in size!")
      else:
        for i in range(0,Min_counter):
          try:
            container.RG_Min_Targets.append(container.constants['Min_Target%d' % (i+1)])
          except:
            raise IOError("RG Plug-in: 'Min_Target%d' does not exist!" % (i+1))
          try:
            container.RG_Max_Targets.append(container.constants['Max_Target%d' % (i+1)])
          except:
            raise IOError("RG Plug-in: 'Max_Target%d' does not exist!" % (i+1))

    # print(container.RG_Min_Targets)
    # print(container.RG_Max_Targets)
    # print(a)

    # check if yMin < yMax is satisfied
    a = np.asarray(container.RG_Max_Targets)-np.asarray(container.RG_Min_Targets)
    # print(a)
    if any(n<=0 for n in a):
      # print("negative found")
      raise IOError("RG Plug-in: 'Min_Targets < Max_Targets' is not satisfied. Check the <ExternalModel> node!")

    inputvariables = set(container.variables)-set(container.outputVariables)
    container.variables = inputvariables

    # print(container.variables) # {'P1'}

  def initialize(self, container, runInfoDict, inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    # print("\n###############################################\n")
    # # print(runInfoDict['WorkingDir'])
    # print(inputFiles[1].__dict__)
    # print("\n###############################################\n")
    # initialization: ensure each var has an initial value
    # for var in container.variables:
    #   if var not in container.coefficients:
    #     container.coefficients[var] = 1.0
    #     print("ExamplePlugin: not found coefficient for variable "+var+". Default value is 1.0!")
    # container.stepSize = (container.endValue - container.startValue)/float(container.numberPoints)

  def createNewInput(self, container, inputs, samplerType, **Kwargs):
    # Extract the matrix file name
    for item in inputs:
      # print(item)
      if 'UserGenerated' in item.__class__.__name__: # look for the file input that contains the path to XML file
        # Assemble the filename
        MatrixFileName = item.__dict__['_File__path']+item.__dict__['_File__base']+'.'+item.__dict__['_File__ext']
        # print(MatrixFileName)
        f = open("MatrixFilePath.txt","w")
        f.write(MatrixFileName)
        f.close()

        # Remove this item from inputs list
        inputs.remove(item)

    if 'MatrixFileName' not in locals():
      f = open("MatrixFilePath.txt","r")
      MatrixFileName = f.read()
      f.close()
    # print(MatrixFileName)

    # Load the XML file containing the ABC matrices
    container.Tss, container.n, container.m, container.p, container.para_array, container.UNorm_list, container.XNorm_list, container.XLast_list, container.YNorm_list, container.A_list, container.B_list, container.C_list, container.eig_A_array = read_parameterized_XML(MatrixFileName)
    # Tss is the sampling period of discrete A,B,C matrices

    if len(container.RG_Min_Targets)!=container.p or len(container.RG_Max_Targets)!=container.p:
      sys.exit('ERROR:  Check the size of "Min_Target" ({}) or "Max_Target" ({}). \n\tBoth should contain {} items.\n'.format(len(container.RG_Min_Targets), len(container.RG_Max_Targets), container.p))

    """ Keep only the profiles with YNorm within the [y_min, y_max] range """
    container.para_array, container.UNorm_list, container.XNorm_list, container.XLast_list, container.YNorm_list, container.A_list, container.B_list, container.C_list, container.eig_A_array = check_YNorm_within_Range(
      container.RG_Min_Targets, container.RG_Max_Targets, container.para_array, container.UNorm_list, container.XNorm_list, container.XLast_list, container.YNorm_list, container.A_list, container.B_list, container.C_list, container.eig_A_array)
    if container.YNorm_list == []:
      sys.exit('ERROR:  No proper linearization point (YNorm) found in Matrix File. \n\tPlease provide a state space profile linearized within the [Min_Target, Max_Target] range\n')
    max_eigA_id = container.eig_A_array.argmax()
    container.A_m = container.A_list[max_eigA_id]; container.B_m = container.B_list[max_eigA_id]; container.C_m = container.C_list[max_eigA_id]; container.D_m = np.zeros((container.p,container.m)) # all zero D matrix
    # print(container.eig_A_array)
    # print(max_eigA_id)

    # print("\n###############################################\n")

    return Kwargs['SampledVars']

  def run(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      This method takes the variables in input and computes
      oneOutputOfThisPlugin(t) = var1Coefficient*exp(var1*t)+var2Coefficient*exp(var2*t) ...
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    """ Process the input from XML file """
    # extract the power setpoint from Inputs, type == <class 'float'>
    for var in container.variables:
      r_value = Inputs[var]
    # print("\n###############################################\n")
    # print("r_value=", r_value, type(r_value))

    """ MOAS steps Limit """
    g = int(container.constants['MOASsteps'])  # numbers of steps to look forward

    """ Select the correct profile with ABCD matrices """
    # Find the correct profile according to r_value
    profile_id = (np.abs(container.para_array - r_value)).argmin()
    # print(profile_id)
    # Retrive the correct A, B, C matrices
    A_d = container.A_list[profile_id]; B_d = container.B_list[profile_id]; C_d = container.C_list[profile_id]; D_d = np.zeros((container.p,container.m)) # all zero D matrix
    # Retrive the correct y_0, r_0 and X
    y_0 = container.YNorm_list[profile_id]; r_0 = float(container.UNorm_list[profile_id]);
    xLast=container.XLast_list[profile_id]; xNorm=container.XNorm_list[profile_id]
    # print(type(r_0))

    """ XLast and r_value """
    if container.constants['Sys_State_x']==[]: # if user didn't supply the final system state vector
      X_Last_RG = np.asarray(xLast - xNorm)
    else:
      X_Last_RG = np.asarray(container.constants['Sys_State_x']) - np.asarray(xNorm)
    # print("X_Last_RG=", X_Last_RG, type(X_Last_RG))
    # print(a)
    r_value_RG = float(r_value) - r_0


    """ Calculate Maximal Output Admissible Set (MOAS) """
    s = [] # type == <class 'list'>
    for i in range(0,container.p):
      s.append([abs(container.RG_Max_Targets[i] - y_0[i])])
      s.append([abs(y_0[i] - container.RG_Min_Targets[i])])
    # print(s)
    H, h = fun_MOAS_noinf(A_d, B_d, C_d, D_d, s, g) # H and h, type = <class 'numpy.ndarray'>
    # print("H:\n", H); print("h:\n", h)

    """ Call the Reference Governor to mild the r_value """
    v_RG = fun_RG_SISO(0, X_Last_RG, r_value_RG, H, h, container.p) # v_RG: type == <class 'numpy.ndarray'>

    """ 2nd adjustment """
    # MOAS for the steps "g+1" - step "2g"
    Hm, hm = fun_MOAS_noinf(container.A_m, container.B_m, container.C_m, container.D_m, s, g)
    # Calculate the max/min for v, ensuring the hm-Hxm*x(g+1) always positive for the next g steps.
    v_max, v_min = fun_2nd_gstep_calc(X_Last_RG, Hm, hm, container.A_m, container.B_m, g)

    if v_RG < v_min:
      v_RG = v_min
    elif v_RG > v_max:
      v_RG = v_max

    # Provide the Output variable Vi with value
    container.__dict__[container.outputVariables[0]] = v_RG + r_0
    container.__dict__[container.outputVariables[1]] = v_min + r_0
    container.__dict__[container.outputVariables[2]] = v_max + r_0

  ###############################
  #### RAVEN API methods END ####
  ###############################

  ##################################
  #### Sub Functions Definition ####
  ##################################

def read_parameterized_XML(MatrixFileName):
    tree = ET.parse(MatrixFileName)
    root = tree.getroot()
    para_array = []; UNorm_list = []; XNorm_list = []; XLast_list = []; YNorm_list =[]
    A_Re_list = []; B_Re_list = []; C_Re_list = []; A_Im_list = []; B_Im_list = []; C_Im_list = []
    for child1 in root:
        # print(' ',child1.tag) # DMDrom
        for child2 in child1:
            # print('  > ', child2.tag) # ROM, DMDcModel
            for child3 in child2:
                # print('  >  > ', child3.tag) # dmdTimeScale, UNorm, XNorm, XLast, Atilde, Btilde, Ctilde
                if child3.tag == 'dmdTimeScale':
                    # print(child3.text)
                    Temp_txtlist = child3.text.split(' ')
                    Temp_floatlist = [float(item) for item in Temp_txtlist]
                    TimeScale = np.asarray(Temp_floatlist)
                    TimeInterval = TimeScale[1]-TimeScale[0]
                    # print(TimeInterval) #10.0
                if child3.tag == 'UNorm':
                    for child4 in child3:
                        # print('  >  >  > ', child4.tag)
                        # print('  >  >  > ', child4.attrib)
                        para_array.append(float(child4.attrib['ActuatorParameter']))
                        Temp_txtlist = child4.text.split(' ')
                        Temp_floatlist = [float(item) for item in Temp_txtlist]
                        UNorm_list.append(np.asarray(Temp_floatlist))
                    para_array = np.asarray(para_array)
                    # print(para_array)
                    # print(UNorm_list)
                    # print(np.shape(self.UNorm))
                if child3.tag == 'XNorm':
                    for child4 in child3:
                        Temp_txtlist = child4.text.split(' ')
                        Temp_floatlist = [float(item) for item in Temp_txtlist]
                        XNorm_list.append(np.asarray(Temp_floatlist))
                    # print(XNorm_list)
                    # print(np.shape(self.XNorm))
                if child3.tag == 'XLast':
                    for child4 in child3:
                        Temp_txtlist = child4.text.split(' ')
                        Temp_floatlist = [float(item) for item in Temp_txtlist]
                        XLast_list.append(np.asarray(Temp_floatlist))
                    # print(XLast_list)
                    # print(np.shape(self.XLast))
                if child3.tag == 'YNorm':
                    for child4 in child3:
                        Temp_txtlist = child4.text.split(' ')
                        Temp_floatlist = [float(item) for item in Temp_txtlist]
                        YNorm_list.append(np.asarray(Temp_floatlist))
                    # print(YNorm_list)
                    # print(YNorm_list[0])
                    # print(np.shape(YNorm_list))
                    # print(np.shape(self.YNorm))
                for child4 in child3:
                    for child5 in child4:
                        # print('  >  >  > ', child5.tag) # real, imaginary, matrixShape, formatNote
                        if child5.tag == 'real':
                            Temp_txtlist = child5.text.split(' ')
                            Temp_floatlist = [float(item) for item in Temp_txtlist]
                            # print(Temp_txtlist)
                            # print(Temp_floatlist)
                            if child3.tag == 'Atilde':
                                A_Re_list.append(np.asarray(Temp_floatlist))
                            if child3.tag == 'Btilde':
                                B_Re_list.append(np.asarray(Temp_floatlist))
                            if child3.tag == 'Ctilde':
                                C_Re_list.append(np.asarray(Temp_floatlist))

                        if child5.tag == 'imaginary':
                            Temp_txtlist = child5.text.split(' ')
                            Temp_floatlist = [float(item) for item in Temp_txtlist]
                            # print(Temp_txtlist)
                            # print(Temp_floatlist)
                            if child3.tag == 'Atilde':
                                A_Im_list.append(np.asarray(Temp_floatlist))
                            if child3.tag == 'Btilde':
                                B_Im_list.append(np.asarray(Temp_floatlist))
                            if child3.tag == 'Ctilde':
                                C_Im_list.append(np.asarray(Temp_floatlist))

    # print(A_Re_list)
    # print(C_Im_list)
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

    # print(A_Re_list[19])
    # print(B_Re_list[19])
    # print(C_Re_list[19])

    A_list = A_Re_list
    B_list = B_Re_list
    C_list = C_Re_list

    eig_A_array=[]
    # eigenvalue of A
    for i in range(len(para_array)):
        w,v = np.linalg.eig(A_list[i])
        eig_A_array.append(max(w))
    eig_A_array = np.asarray(eig_A_array)
    # print(eig_A_array)

    return TimeInterval, n, m, p, para_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array

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

def fun_MOAS_noinf(A, B, C, D, s, g):
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
    """ Build the core of H and h """
    # H = np.concatenate((0*Kx, Lim),axis=1); h = s
    # NewBlock = np.concatenate((Kx, Kr),axis=1)
    # H = np.concatenate((H,NewBlock)); h = np.concatenate((h,s))
    H = np.concatenate((Kx, Kr),axis=1); h = s

    """ Build the add-on blocks of H and h """
    i = 0
    while i < g :
        i = i + 1
        Kx = np.dot(Kx, A)
        Kr = Lim - np.dot(Kx,T)

        NewBlock = np.concatenate((Kx,Kr), axis=1)
        H = np.concatenate((H,NewBlock)); h = np.concatenate((h,s))
        """ To Insert the ConstRedunCheck """

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
    # print(np.shape(alpha))
    # print(np.shape(beta))
    v_st = []; v_bt = []
    for k in range(0,len(alpha)):
        if beta[k]>0:
            v_st.append(alpha[k]/beta[k])
        elif beta[k]<0:
            v_bt.append(alpha[k]/beta[k])
    # print('v_smaller_than,\n',v_st)
    v_max = np.asarray(min(v_st))
    v_min = np.asarray(max(v_bt))
    return v_max, v_min

