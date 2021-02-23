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
# import zip
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
# from MOAS import fun_MOAS, fun_RG_SISO
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End-----------------------------------------------------------


class RefGov_SIMO(ExternalModelPluginBase):
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

      if child.tag.strip() == "constant":
        # Extract the constant names and their values: container.constants = {'TimeInterval': 3600.0}
        if "varName" not in child.attrib:
          raise IOError("RG Plug-in: attribute varName must be present in <coefficient> XML node!")
        container.constants[child.attrib['varName']] = float(child.text)
    # print(container.constants) # {'TimeInterval': 3600.0}

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

    # print("\n###############################################\n")

    # Load the XML file containing the ABC matrices
    tree = ET.parse(MatrixFileName)
    root = tree.getroot()
    for child1 in root:
      # print(' ',child1.tag) # DMDrom
      for child2 in child1:
        # print('  > ', child2.tag) # ROM, DMDcModel

        for child3 in child2:
          # print('  >  > ', child3.tag) # UNorm, XNorm, XLast, Atilde, Btilde, Ctilde
          if child3.tag == 'UNorm':
            # print(child3.text)
            Temp_txtlist = child3.text.split('; ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            container.UNorm = np.asarray(Temp_floatlist)
            # print(np.shape(container.UNorm))
          if child3.tag == 'XNorm':
            # print(child3.text)
            Temp_txtlist = child3.text.split('; ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            container.XNorm = np.asarray(Temp_floatlist)
            # print(np.shape(container.XNorm))
          if child3.tag == 'XLast':
            # print(child3.text)
            Temp_txtlist = child3.text.split('; ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            container.XLast = np.asarray(Temp_floatlist)
            # print(np.shape(container.XLast))
          if child3.tag == 'YNorm':
            # print(child3.text)
            Temp_txtlist = child3.text.split('; ')
            Temp_floatlist = [float(item) for item in Temp_txtlist]
            container.YNorm = np.asarray(Temp_floatlist)
            # print(np.shape(container.YNorm))

          for child4 in child3:
            # print('  >  >  > ', child4.tag) # real, imaginary, matrixShape, formatNote
            if child4.tag == 'real':
              Temp_txtlist = child4.text.split('; ')
              Temp_txtlist = [item.split(' ') for item in Temp_txtlist]
              Temp_floatlist = [[float(y) for y in x ] for x in Temp_txtlist]
              # print(Temp_txtlist)
              # print(Temp_floatlist)
              if child3.tag == 'Atilde':
                A_Re = np.asarray(Temp_floatlist)
              if child3.tag == 'Btilde':
                B_Re = np.asarray(Temp_floatlist)
              if child3.tag == 'Ctilde':
                C_Re = np.asarray(Temp_floatlist)

            if child4.tag == 'imaginary':
              Temp_txtlist = child4.text.split('; ')
              Temp_txtlist = [item.split(' ') for item in Temp_txtlist]
              Temp_floatlist = [[float(y) for y in x ] for x in Temp_txtlist]
              # print(Temp_txtlist)
              # print(Temp_floatlist)
              if child3.tag == 'Atilde':
                A_Im = np.asarray(Temp_floatlist)
              if child3.tag == 'Btilde':
                B_Im = np.asarray(Temp_floatlist)
              if child3.tag == 'Ctilde':
                C_Im = np.asarray(Temp_floatlist)

    container.Atilde = A_Re
    container.Btilde = B_Re
    container.Ctilde = C_Re

    if len(container.Ctilde) != len(container.RG_Max_Targets) or \
            len(container.Ctilde) != len(container.RG_Min_Targets):
      raise IOError("RG Plug-in: Dimension error in Min_Targets or Max_Targets. Each should have {} entries!".format(len(container.Ctilde)))


    # Keep following item in "container": A, B, C, UNorm, XNorm, XLast, YNorm
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
    # print(container.constants)
    # print("*** *** *** *** *** ***")
    # print(Inputs)
    # print("UNorm: ", container.UNorm)
    # print("XNorm: ", container.XNorm)
    # print("XLast: ", container.XLast)
    # print("YNorm: ", container.YNorm)
    # print("Atilde:\n", container.Atilde)
    # print("Btilde:\n", container.Btilde)
    # print("Ctilde:\n", container.Ctilde)
    # print("*** *** *** *** *** ***")

    """ Reference input / output, and output bound """
    r_0 = container.UNorm # [35.0E6] # reference input value, type == <class 'float'>
    y_0 = container.YNorm # [35.0E6, 1340.0]  # nominal value of output, 35MW, 1340 degC
    # set up the output value bounds
    y_min = container.RG_Min_Targets # [20.0E6, 1270.]  # minimum values
    y_max = container.RG_Max_Targets # [50.0E6, 1410.]  # maximum value
    """ MOAS steps Limit """
    g = int(container.constants['MOASsteps'])  # numbers of steps to look forward


    """ Process the input from XML file """
    # extract the power setpoint from Inputs, type == <class 'float'>
    for var in container.variables:
      Ri = Inputs[var]
    # print("Time_Int=",Time_Int, type(Time_Int), "; Pi=", Pi, type(Pi))

    """ A, B, C, D matrices """
    A_d = container.Atilde; B_d = container.Btilde; C_d = container.Ctilde
    n = len(A_d); m = len(B_d[0]); p = len(C_d)  # n: dim of x; m: dim of u; p: dim of y. Type = <class 'int'>
    D_d = np.zeros((p,m))
    # print("D_d:\n", D_d)

    """ XLast and r_value """
    X_Last_RG = container.XLast - container.XNorm
    # X_Last_RG = np.zeros(3)
    # print(X_Last_RG)
    r_value_RG = Ri - r_0[0]
    # print("r_0 = ",r_0[0])


    """ Calculate Maximal Output Admissible Set (MOAS) """
    s = [] # type == <class 'list'>
    for i in range(0, p):
      s.append([y_max[i] - y_0[i]])
      s.append([y_0[i] - y_min[i]])
    # print(s)
    H, h = fun_MOAS(A_d, B_d, C_d, D_d, s, g) # H and h, type = <class 'numpy.ndarray'>
    # print("H:\n", H); print("h:\n", h)

    """ Call the Reference Governor to mild the r_value """
    v_RG, v_min, v_max = fun_RG_SISO_vBound(0, X_Last_RG, r_value_RG, H, h, p)  # v_RG: type == <class 'numpy.ndarray'>
    # print("v_RG = ", v_RG)
    # print("v_min= ", v_min)
    # print("v_max= ", v_max)

    # Provide the Output variable Vi with value
    container.__dict__[container.outputVariables[0]] = v_RG + r_0[0]
    container.__dict__[container.outputVariables[1]] = v_min + r_0[0]
    container.__dict__[container.outputVariables[2]] = v_max + r_0[0]

  ###############################
  #### RAVEN API methods END ####
  ###############################

  ##################################
  #### Sub Functions Definition ####
  ##################################


def fun_KalmanFilter(A_d, B_d, C_d, D_d, sigma_u, sigma_y, x_KF, v_RG, P_KF, y_sim):
  n = len(A_d);  p = len(C_d)  # n: dim of x; m: dim of v; p: dim of y. Type = <class 'int'>
  # Calculate covariance matrices
  Q = np.identity(n) * sigma_u
  R = np.identity(p) * sigma_y
  # 1. Project the State xp ahead
  xp = A_d.dot(x_KF) + B_d.dot(v_RG)
  # print("x_pro =",xp)
  # 2. Project the State Error Covariance matrix ahead
  Pp = (A_d.dot(P_KF)).dot(A_d.T) + Q
  # print("Pp=", Pp)
  # 3. Compute the Kalman gain
  K = (Pp.dot(C_d.T)).dot(np.linalg.inv((C_d.dot(Pp)).dot(C_d.T) + R))
  # 4. Update State estimate and State Error Covariance matrix
  x_KF = xp + K.dot(y_sim - D_d.dot(v_RG) - C_d.dot(xp))
  P_KF = (np.identity(n) - K.dot(C_d)).dot(Pp)

  return x_KF, P_KF

def fun_MOAS(A, B, C, D, s, g):
  p = len(C)  # dimension of y
  T = np.linalg.solve(np.identity(len(A)) - A, B)
  """ Build the S matrix"""
  S = np.zeros((2 * p, p))
  for i in range(0, p):
    S[2 * i, i] = 1.0
    S[2 * i + 1, i] = -1.0
  Kx = np.dot(S, C)
  # print("Kx", Kx)
  Lim = np.dot(S, (np.dot(C, T) + D))
  # print("Lim", Lim)
  Kr = np.dot(S, D)
  # print("Kr", Kr)
  """ Build the core of H and h """
  H = np.concatenate((0 * Kx, Lim), axis=1);
  h = s
  NewBlock = np.concatenate((Kx, Kr), axis=1)
  H = np.concatenate((H, NewBlock));
  h = np.concatenate((h, s))

  """ Build the add-on blocks of H and h """
  i = 0
  while i < g:
    i = i + 1
    Kx = np.dot(Kx, A)
    Kr = Lim - np.dot(Kx, T)

    NewBlock = np.concatenate((Kx, Kr), axis=1)
    H = np.concatenate((H, NewBlock));
    h = np.concatenate((h, s))
    """ To Insert the ConstRedunCheck """

  return H, h

def fun_RG_SISO(v_0, x, r, H, h, p):
  n = len(x)  # dimension of x
  x = np.vstack(x)  # x is horizontal array, must convert to vertical for matrix operation
  # because v_0 and r are both scalar, so no need to vstack
  Hx = H[:, 0:n];
  Hv = H[:, n:]
  alpha = h - np.dot(Hx, x) - np.dot(Hv, v_0)  # alpha is the system remaining vector
  beta = np.dot(Hv, (r - v_0))  # beta is the anticipated response vector with r

  kappa = 1
  for k in range(0, len(alpha)):
    if alpha[k] < beta[k]:
      if alpha[k] > 0:
        kappa = min(kappa, alpha[k] / beta[k])
      else:
        kappa = kappa
    else:
      kappa = kappa
  v = np.asarray(v_0 + kappa*(r-v_0)).flatten()

  return v

def fun_RG_SISO_vBound(v_0, x, r, H, h, p):
  n = len(x)  # dimension of x
  x = np.vstack(x)  # x is horizontal array, must convert to vertical for matrix operation
  # because v_0 and r are both scalar, so no need to vstack
  Hx = H[:, 0:n];  Hv = H[:, n:]
  alpha = h - np.dot(Hx, x) - np.dot(Hv, v_0)  # alpha is the system remaining vector
  beta = np.dot(Hv, (r - v_0))  # beta is the anticipated response vector with r
  # print("alpha = \n",alpha)
  # print("Hv = \n", Hv)
  # print("beta = \n", beta)

  """ Calculate the vBounds """
  v_st = [] # smaller than
  v_bt = [] # bigger than
  # for the first 2p rows (final steady state corresponding to constant v), keep the max/min.
  for k in range(0, 2 * p):
    if Hv[k] > 0:
      v_st.append(alpha[k] / Hv[k] + v_0)
    elif Hv[k] < 0:
      v_bt.append(alpha[k] / Hv[k] + v_0)
  # for the following rows, adjust the max/min when necessary.
  for k in range(2 * p, len(alpha)):
    if Hv[k] > 0 and alpha[k] > 0 and alpha[k] < beta[k]:
      v_st.append(alpha[k] / Hv[k] + v_0)
    elif Hv[k] < 0 and alpha[k] > 0 and alpha[k] < beta[k]:
      v_bt.append(alpha[k] / Hv[k] + v_0)
  v_max = float(min(v_st))
  v_min = float(max(v_bt))
  # print("r =",r,"type=",type(r))
  # print("v_min=",v_min,"type=",type(v_min))
  # print("v_max=",v_max,"type=",type(v_max))

  if r > v_max:
    v = v_max
  elif r < v_min:
    v = v_min
  else:
    v = r

  # v = np.asarray(v).flatten()
  # print("v=",v,"type=",type(v))

  return v, v_min, v_max
