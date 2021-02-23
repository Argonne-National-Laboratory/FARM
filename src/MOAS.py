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

import numpy as np
import xml.etree.ElementTree as ET

def fun_GasTurb_ABCD_u(u,A_Hist,B_Hist,C_Hist,D_Hist,u0_Hist,y0_Hist):
    # need to convert u0_hist to a one-row array
    result = np.where(u0_Hist.reshape(1, -1)[0] <= u)
    # "result" is a tuple. need to extract the array within and find the index.
    idx = result[0][-1]
    # print(idx)
    A_d = A_Hist[:, :, idx];    B_d = B_Hist[:, :, idx]
    C_d = C_Hist[:, :, idx];    D_d = D_Hist[:, :, idx]
    u0 = u0_Hist[:, idx];    y0 = y0_Hist[:, idx]
    return A_d, B_d, C_d, D_d, u0, y0

def fun_DMDc(X1,X2,U,Y1,A_0,B_0,C_0,k,nr):
    # Input dimensions:
    # X1, X2: both [n*L] matrices, n-dimension state vectors by L entries
    # U: [m*L] matrix, m-dimension control vector by L entries
    # A_0[n*n], B_0[n*m]: initial guess of A and B, <class 'numpy.ndarray'>
    # k: int
    # nr: int
    n=len(X2) # Dimension of State Vector
    Omega=np.concatenate((X1, U), axis=0) # Omega Matrix, stack X1 and U
    U,S,V = np.linalg.svd(Omega,full_matrices=True) # Singular Value Decomp. U*S*V'=Omega
    S=np.diag(S); V=V.T
    # print("U=",U); print("S=",S); print("V=",V)
    p=len(S) # p is the number of non-zero element in S
    Ut=U[:,0:p];St=S[0:p,0:p];Vt=V[:,0:p] # truncation for the first p elements
    # print("Ut=", Ut); print("St=", St); print("Vt=", Vt)
    U1=Ut[0:n,:];U2=Ut[n:,:] # Cut Ut into U1 (for x) and U2 (for v)
    # print(U1)
    # print(U2)
    Q,R=np.linalg.qr(St) # QR decomp. St=Q*R, Q unitary, R upper triangular
    r=1/np.linalg.cond(R) # inverse of condition number of invention: smallest eigenvalue/biggest eigenvalue
    # print("St=",St);    print("Q=",Q);    print("R=",R);    print("r=",r)

    if np.linalg.det(R)==0: # if R is singular matrix, return the reference value
        A_id = A_0; B_id = B_0; C_id = C_0
    else:
        beta = X2.dot(Vt).dot(np.linalg.inv(R)).dot(Q.T)
        A_id=beta.dot(U1.T)
        B_id=beta.dot(U2.T)
        C_id=Y1.dot(np.linalg.pinv(X1))

    # if k<=0:
    #     # beta=X2*Vt*inv(R)*(Q');
    #     beta=(X2.dot(Vt)).dot(np.linalg.inv(St))
    #     A_id=beta.dot(U1.T)
    #     B_id=beta.dot(U2.T)
    # elif k>0 and k<nr+1:
    #     A_id=A_0; B_id=B_0
    # elif k>=nr+1 and r<1e-7:
    #     A_id=A_0; B_id=B_0
    # else:
    #     beta=X2.dot(Vt).dot(np.linalg.inv(R)).dot(Q.T)
    #     A_id=beta.dot(U1.T)
    #     B_id=beta.dot(U2.T)
    # outputs:
    # A_id: [n*n]. Estimated A matrix <class 'numpy.ndarray'>
    # B_id: [n*m]. Estimated B matrix <class 'numpy.ndarray'>
    return A_id, B_id, C_id

def fun_KalmanFilter(A_d,B_d,C_d,D_d, sigma_u, sigma_y, x_KF, v_RG, P_KF, y_sim):
    # Input dimensions:
    # A[n*n]; B[n*m]; C[p*n]; D[p*m]. All <class 'numpy.ndarray'>
    # sigma_u (for Q), sigma_y (for R): floating number
    sigma_u = float(sigma_u); sigma_y = float(sigma_y)
    # x_KF[n*1]. Column vector, <class 'numpy.ndarray'>
    x_KF = x_KF.reshape(-1,1)
    # v_RG[m*1]. Column vector, <class 'numpy.ndarray'>
    v_RG = v_RG.reshape(-1,1)
    # P_KF[n*n]. Square matrix, <class 'numpy.ndarray'>
    # y_sim[p*1]. Column vector, <class 'numpy.ndarray'>
    y_sim = y_sim.reshape(-1,1)

    n = len(A_d); m = len(B_d[0]); p = len(C_d)  # n: dim of x; m: dim of v; p: dim of y. Type = <class 'int'>
    # Calculate covariance matrices
    Q = np.identity(n) * sigma_u
    R = np.identity(p) * sigma_y
    # 1. Project the State xp ahead
    xp = A_d.dot(x_KF) + B_d.dot(v_RG)
    # print("x_pro =",xp)
    # 2. Project the State Error Covariance matrix ahead
    Pp = A_d.dot(P_KF).dot(A_d.T) + Q
    # print("Pp=", Pp)
    # 3. Compute the Kalman gain
    K = Pp.dot(C_d.T).dot(np.linalg.inv(C_d.dot(Pp).dot(C_d.T) + R))
    # 4. Update State estimate and State Error Covariance matrix
    x_KF = (xp + K.dot(y_sim - D_d.dot(v_RG) - C_d.dot(xp))).reshape(-1,1)
    P_KF = (np.identity(n) - K.dot(C_d)).dot(Pp)
    # final output:
    # x_KF: [n*1]. Column vector, <class 'numpy.ndarray'>
    # P_KF: [n*n]. Square matrix, <class 'numpy.ndarray'>
    return x_KF, P_KF

def fun_MOAS(A, B, C, D, s, g):
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
    H = np.concatenate((0*Kx, Lim),axis=1); h = s
    NewBlock = np.concatenate((Kx, Kr),axis=1)
    H = np.concatenate((H,NewBlock)); h = np.concatenate((h,s))

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
    v = np.asarray(v_max).flatten()
  elif r < v_min:
    v = np.asarray(v_min).flatten()
  else:
    v = np.asarray(r).flatten()

  # v = np.asarray(v).flatten()
  # print("v=",v,"type=",type(v))

  return v, v_min, v_max

def read_parameterized_XML(MatrixFileName):
    tree = ET.parse(MatrixFileName)
    root = tree.getroot()
    power_array = []; UNorm_list = []; XNorm_list = []; XLast_list = []; YNorm_list =[]
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
                        power_array.append(float(child4.attrib['power']))
                        Temp_txtlist = child4.text.split(' ')
                        Temp_floatlist = [float(item) for item in Temp_txtlist]
                        UNorm_list.append(np.asarray(Temp_floatlist))
                    power_array = np.asarray(power_array)
                    # print(power_array)
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
    for i in range(len(power_array)):
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
    for i in range(len(power_array)):
        w,v = np.linalg.eig(A_list[i])
        eig_A_array.append(max(w))
    eig_A_array = np.asarray(eig_A_array)
    # print(eig_A_array)
    
    return TimeInterval, n, m, p, power_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array

def check_YNorm_within_Range(y_min, y_max, power_array, UNorm_list, XNorm_list, XLast_list, YNorm_list, A_list, B_list, C_list, eig_A_array):
    UNorm_list_ = []; XNorm_list_ = []; XLast_list_ = []; YNorm_list_ =[]
    A_list_ = []; B_list_ = []; C_list_ = []; power_array_ = []; eig_A_array_ =[]

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
            power_array_.append(power_array[i])
            eig_A_array_.append(eig_A_array[i])

    power_array_ = np.asarray(power_array_); eig_A_array_ = np.asarray(eig_A_array_)
    return power_array_, UNorm_list_, XNorm_list_, XLast_list_, YNorm_list_, A_list_, B_list_, C_list_, eig_A_array_


        
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

    