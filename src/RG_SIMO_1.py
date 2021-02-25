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
  Date  :  04/27/2020
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
# from MOAS import fun_MOAS, fun_RG_SISO
#External Modules End-----------------------------------------------------------

#Internal Modules---------------------------------------------------------------
from PluginsBaseClasses.ExternalModelPluginBase import ExternalModelPluginBase
#Internal Modules End-----------------------------------------------------------


class RG_SIMO_1(ExternalModelPluginBase):
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
    # container.coefficients = {}
    # container.startValue   = None
    # container.endValue     = None
    # container.numberPoints = 10

    # Extract the output Variable name: container.outputVariable = 'Vi'
    outputVarNode    = xmlNode.find("outputVariable")
    # print(outputVarNode)
    if outputVarNode is None:
      raise IOError("RG Plugin: <outputVariable> XML block must be inputted!")
    # container.outputVariable = outputVarNode.text.strip()
    container.outputVariable = [var.strip() for var in outputVarNode.text.split(",")]
    # print(container.outputVariable)
    # Extract the Monotonic Variable name: container.monotonicVariableName = 'monotonicVariable'
    # monotonicVarNode = xmlNode.find("monotonicVariable")
    # if monotonicVarNode is None:
    #   raise IOError("ExamplePlugin: <monotonicVariable> XML block must be inputted!")
    # container.monotonicVariableName = monotonicVarNode.text.strip()

    for child in xmlNode:
      # xmlNode is the Nodes within the section of <ExternalModel>.
      # child.tag are the strings containing each node name. child.tag == child.tag.strip()
      if child.tag.strip() == "variables":
        # get verbosity if it exists
        # Extract Variable names: container.variables = ['Vi', 'Pi']
        container.variables = [var.strip() for var in child.text.split(",")]
        # print(container.variables) # OK
        # if container.outputVariable not in container.variables:
        #   raise IOError("RG Plug-in: "+container.outputVariable+" variable MUST be present in the <variables> definition!")

      if child.tag.strip() == "constant":
        # Extract the constant names and their values: container.constants = {'TimeInterval': 3600.0}
        if "varName" not in child.attrib:
          raise IOError("RG Plug-in: attribute varName must be present in <coefficient> XML node!")
        container.constants[child.attrib['varName']] = float(child.text)

    #   if child.tag.strip() == "coefficient":
    #     # Extract the coefficients names and their values: container.coefficients = {'a': 1.1, 'b': -1.1, 'c': -1.1}
    #     if "varName" not in child.attrib:
    #       raise IOError("ExamplePlugin: attribute varName must be present in <coefficient> XML node!")
    #     container.coefficients[child.attrib['varName']] = float(child.text)
    #   if child.tag.strip() == "startMonotonicVariableValue":
    #     # Extract the monotonic Variable start value: container.startValue = 0.0
    #     container.startValue = float(child.text)
    #   if child.tag.strip() == "endMonotonicVariableValue":
    #     # Extract the monotonic Variable end value: container.endValue = 100.0
    #     container.endValue = float(child.text)
    #   if child.tag.strip() == "numberCalculationPoints":
    #     # Extract the total calculation points: container.numberPoints = 100
    #     container.numberPoints = int(child.text)
    # if container.startValue is None:
    #   raise IOError("ExamplePlugin: <startMonotonicVariableValue> XML has not been inputted!")
    # if container.endValue is None:
    #   raise IOError("ExamplePlugin: <endMonotonicVariableValue> XML has not been inputted!")
    # pop: kick one element from the list
    # container.variables.pop(container.variables.index("monotonicVariable"))
    # print(container.constants)
    inputvariables = set(container.variables)-set(container.outputVariable)
    container.variables = inputvariables

  # container.variables.pop(container.variables.index("TimeInt"))
    # print(container.variables)

  def initialize(self, container,runInfoDict,inputFiles):
    """
      Method to initialize this plugin
      @ In, container, object, self-like object where all the variables can be stored
      @ In, runInfoDict, dict, dictionary containing all the RunInfo parameters (XML node <RunInfo>)
      @ In, inputFiles, list, list of input files (if any)
      @ Out, None
    """
    # initialization: ensure each var has an initial value
    # for var in container.variables:
    #   if var not in container.coefficients:
    #     container.coefficients[var] = 1.0
    #     print("ExamplePlugin: not found coefficient for variable "+var+". Default value is 1.0!")
    # container.stepSize = (container.endValue - container.startValue)/float(container.numberPoints)

  def run(self, container, Inputs):
    """
      This is a simple example of the run method in a plugin.
      This method takes the variables in input and computes
      oneOutputOfThisPlugin(t) = var1Coefficient*exp(var1*t)+var2Coefficient*exp(var2*t) ...
      @ In, container, object, self-like object where all the variables can be stored
      @ In, Inputs, dict, dictionary of inputs from RAVEN

    """
    # print(container.constants)
    """ Reference input / output, and output bound """
    r_0 = 35.0E6 # reference input value, type == <class 'float'>
    y_0 = [35.0E6, 1340.0]  # nominal value of output, 35MW, 1340 degC
    # set up the output value bounds
    y_min = [20.0E6, 1270.]  # minimum values
    y_max = [50.0E6, 1410.]  # maximum value
    """ MOAS steps Limit """
    g = 10  # numbers of steps to look forward


    """ Process the input from XML file """
    # extract the Time_Int from container.constants, type == <class 'float'>
    Time_Int = container.constants["TimeInterval"]
    # extract the power setpoint from Inputs, type == <class 'float'>
    for var in container.variables:
      Pi = float(Inputs[var][0])
    # print("Time_Int=",Time_Int, type(Time_Int), "; Pi=", Pi, type(Pi))

    """ External r signal "r_ext" and associated time stamp "t_ext" """
    # t_ext = np.arange(0, int(Time_Int)+1, 1)
    t_ext = np.arange(0, 2, 1) # This is the minimum length, t_ext == [0 1]
    r_ext = np.ones(len(t_ext))*(Pi - r_0)  # external r (scalar at each time step)
    # print(t_ext)

    """ Load the A,B,C,D matrices from MAT file, from the folder defined in <RunInfo> - <WorkingDir> """
    # print(os.getcwd())

    # MAT_filename = "Matrix_c.mat"
    MAT_filename = "Matrix_SIMO.mat"
    MAT_contents = io.loadmat(MAT_filename); # print(MAT_contents.keys())
    # Read A,B,C,D matrices, type == <class 'numpy.ndarray'>
    A_c = MAT_contents['A_c']; B_c = MAT_contents['B_c']; C_c = MAT_contents['C_c']; D_c = MAT_contents['D_c']
    # print('A eigenvalues', np.linalg.eig(A_c)[0])  # returns the eigenvalues and eigenvectors
    n = len(A_c); m = len(B_c[0]); p = len(C_c)  # n: dim of x; m: dim of v; p: dim of y. Type = <class 'int'>
    # print("n=", n, ", m=", m, ", p=", p)

    """ Build a linear system based on A, B, C, D matrices and dT """
    Ts_Trig = 0.1  # sampling time (seconds) for discrete linear system, type == <class 'float'>
    # Discrete system, returning ABCD in disc. type == <class 'numpy.ndarray'>
    A_d, B_d, C_d, D_d, _ = signal.cont2discrete([A_c, B_c, C_c, D_c], Ts_Trig)
    LinSys_11_d = signal.StateSpace(A_d, B_d, C_d, D_d, dt=Ts_Trig)

    """ Calculate Maximal Output Admissible Set (MOAS) """
    s = [] # type == <class 'list'>
    for i in range(0, p):
      # s.append([abs(y_max[i] - y_0[i])])
      # s.append([abs(y_0[i] - y_min[i])])
      s.append([y_max[i] - y_0[i]])
      s.append([y_0[i] - y_min[i]])
    # print(s)
    H, h = fun_MOAS(A_d, B_d, C_d, D_d, s, g) # H and h, type = <class 'numpy.ndarray'>
    # print("H", H); print("h", h)

    """ Initialize the time, system x, and simulation history """
    t = Ts_Trig  # time stamp

    # Try to load the x from past file, # x_sys type == <class 'numpy.ndarray'>
    try:
      x_sys=[]
      with open('Test_GasTurbine_X_1.txt', 'r') as x_file:
        for line in x_file:
          currentvalue = line[:-1]
          x_sys.append(currentvalue)
        x_file.close()
      x_sys = np.asarray(x_sys, dtype=np.float)
    except:
      # if the loading fails (no file), use the zero value
      x_sys = np.zeros(len(A_d))
    # print("Start x_sys=",x_sys, type(x_sys))

    t_hist = []  # the history of time stamps
    r_hist = []  # the history of r signal (scalar)
    v_hist = []  # the history of v signal (scalar)
    y_hist = []  # the history of y signal (vector)
    v_min_hist = []
    v_max_hist = []

    """ Initialize the variables for Kalman Filter usage """
    y_sim = np.zeros(p)  # Initial value of y_sim, for KALMAN FILTER USAGE
    x_KF = np.zeros(n)  # Initial value of x vector, for KALMAN FILTER USAGE
    v_RG = np.zeros(m)  # Initial value of v vector, for KALMAN FILTER USAGE
    P_KF = np.zeros((n, n))  # Initial value of State Error Covariance Matrix
    sigma_u = 1000.0  # Process covariance
    sigma_y = 2.0  # Measured vairable covariance

    """ Start the step-by-step simulation """
    while t <= t_ext[-1]:
      """ Kalman Filter SIMO to estimate the current state x_KF, for RG to use """
      x_KF, P_KF = fun_KalmanFilter(A_d, B_d, C_d, D_d, sigma_u, sigma_y, x_KF, v_RG, P_KF, y_sim)

      t_sim = [0.0, Ts_Trig]  # t and v have to be more than 2 elements to run dlsim
      # t_sim: type = <class 'list'>
      r_value = r_ext[np.where(t_ext >= t)][0]  # find the index of r_ext when triggered
      # r_ext[np.where(t_ext >= t)] type == <class 'numpy.ndarray'>
      # r_ext[np.where(t_ext >= t)][0] type == <class 'float64'>

      """ Call the Reference Governor to mild the r_value """
      # v_RG = fun_RG_SISO(0, x_KF, r_value, H, h, p) # v_RG: type == <class 'numpy.ndarray'>
      v_RG, v_min, v_max = fun_RG_SISO_vBound(0, x_KF, r_value, H, h, p) # v_RG: type == <class 'numpy.ndarray'>
      # v_RG = r_value
      # print(v_RG)
      v_sim = [v_RG, v_RG]

      """ step-by-step simulation, to be replaced by Dymola model: Input = v_sim, output = y_sim"""
      _, y_sim, x_sim = signal.dlsim(LinSys_11_d, v_sim, t_sim, x_sys)  # Simulate a discrete-time linear system.
      # The 3 outputs: t_sys(_), y_sim, x_sim are all <class 'numpy.ndarray'> with 2 components each.
      # Use the 2nd component of x_sim (index=1) to update system x
      x_sys = x_sim[1];  # print("x_sys =", x_sys)
      # discard the first component of y_sim
      y_sim = y_sim[1];   # print(y_sim[1])
      # Add noise to y_sim, for Kalman Filter calculation
      y_sim[0] += np.random.normal(0.0, 0.05E6); y_sim[1] += np.random.normal(0.0, 0.5)

      """ Save time, r, v, and y """
      t_hist.append(t); r_hist.append(r_value);
      v_hist.append(float(v_RG)) # convert v_RG to type == <class 'float'>
      y_hist.append(y_sim)
      t = t + Ts_Trig  # t increment, for next trigger.
      v_min_hist.append(float(v_min)); v_max_hist.append(float(v_max))

    # Transpose y_hist. Orignally y_hist is type == <class 'list'>,
    # has to be converted to <class 'numpy.ndarray'> before transposing
    y_hist = (np.array(y_hist).T)
    # After transposing, y_hist type == <class 'numpy.ndarray'>

    # print(y_hist)
    # print("Final x_sys=",x_sys, type(x_sys))
    with open('Test_GasTurbine_X_1.txt', 'w') as x_file:
      for x in x_sys:
        x_file.write('%.16f\n' % x)
      x_file.close()

    # convert the incremental form values to "real-world" value
    for i in range(len(t_hist)):
      r_hist[i] += r_0
      v_hist[i] += r_0
      y_hist[0, i] += y_0[0]
      y_hist[1, i] += y_0[1]
      v_min_hist[i] += r_0
      v_max_hist[i] += r_0

    # Provide the Output variable Vi with value
    container.__dict__[container.outputVariable[0]] = float(v_hist[-1])
    container.__dict__[container.outputVariable[1]] = float(v_min_hist[-1])
    container.__dict__[container.outputVariable[2]] = float(v_max_hist[-1])

    # "__dict__" stands for dictionary, which is to find the "Vi" item

    # y_hist = y_hist.tolist()
    # print(type(r_hist),type(v_hist),type(y_hist))

    """ Store running result and plot out for debugging purpose """
    # Generate the string of current time
    str_now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    # Save r,v,y1,y2 results to csv
    with open("Test_GasTurbine_CSV_1.csv","a+", newline="") as csvfile:
      writer = csv.writer(csvfile, delimiter=',')
      writer.writerows(zip(r_hist, v_hist, v_min_hist, v_max_hist, y_hist[0], y_hist[1]))
      csvfile.close()

    # Read the csv and plot the signals by now
    t_hist=[];r_hist=[]; v_hist=[]; y_hist=[]; v_min_hist=[]; v_max_hist=[]
    with open("Test_GasTurbine_CSV_1.csv","r", newline="") as csvfile:
      content = csv.reader(csvfile, delimiter=',')
      counter = 0
      for row in content:
        t_hist.append(counter); counter += Ts_Trig
        r_hist.append(float(row[0]))
        v_hist.append(float(row[1]))
        v_min_hist.append(float(row[2]))
        v_max_hist.append(float(row[3]))
        y_hist.append([float(row[4]),float(row[5])])
      csvfile.close()

    y_hist = np.array(y_hist).T

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
  # print("alpha = ",alpha)
  # print("Hv = ", Hv)

  """ Calculate the v_RG """
  # kappa = 1
  # for k in range(0, len(alpha)):
  #   if alpha[k] < beta[k]:
  #     if alpha[k] > 0:
  #       kappa = min(kappa, alpha[k] / beta[k])
  #     else:
  #       kappa = kappa
  #   else:
  #     kappa = kappa
  # v = np.asarray(v_0 + kappa * (r - v_0)).flatten()

  # """ Calculate the vBounds """
  # v_st = []
  # v_bt = []
  # # for the first 2p rows (final steady state corresponding to constant v), keep the max/min.
  # for k in range(0, 2 * p):
  #   if Hv[k] > 0:
  #     v_st.append(alpha[k] / Hv[k] + v_0)
  #   elif Hv[k] < 0:
  #     v_bt.append(alpha[k] / Hv[k] + v_0)
  # # for the following rows, adjust the max/min when necessary.
  # for k in range(2 * p, len(alpha)):
  #   if Hv[k] > 0 and alpha[k] > 0 and alpha[k] < beta[k]:
  #     v_st.append(alpha[k] / Hv[k] + v_0)
  #   elif Hv[k] < 0 and alpha[k] > 0 and alpha[k] < beta[k]:
  #     v_bt.append(alpha[k] / Hv[k] + v_0)
  # v_max = np.asarray(min(v_st))
  # v_min = np.asarray(max(v_bt))
  # # print("v_min=",v_min,"v=",v,"v_max=",v_max)
  # # print("v_min <= v:", v_min <= v, "; v <= v_max:", v <= v_max)
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

  # print("v_min=",v_min,"v=",v,"v_max=",v_max)
  # print("v_min <= v:", v_min <= v, "; v <= v_max:", v <= v_max)

  if r > v_max:
    v = v_max
  elif r < v_min:
    v = v_min
  else:
    v = r

  v = np.asarray(v).flatten()
  # print("v=",v,"type=",type(v))

  return v, v_min, v_max
