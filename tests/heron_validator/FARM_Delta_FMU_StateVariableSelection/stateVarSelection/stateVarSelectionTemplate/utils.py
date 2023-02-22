import numpy as np
import scipy
from scipy.interpolate import interp1d


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

def DMDc(X1, X2, U, Y1, rankSVD):
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
    uTrucSVD, sTrucSVD, vTrucSVD = computeTruncatedSingularValueDecomposition(omega, rankSVD, False, False)
    # Find the truncation rank triggered by "s>=SminValue"
    rankTruc = sum(map(lambda x : x>=np.max(sTrucSVD)*1e-9, sTrucSVD.tolist()))
    if rankTruc < uTrucSVD.shape[1]:
        uTruc = uTrucSVD[:, :rankTruc]
        vTruc = vTrucSVD[:, :rankTruc]
        sTruc = np.diag(sTrucSVD)[:rankTruc, :rankTruc]
    else:
        uTruc = uTrucSVD
        vTruc = vTrucSVD
        sTruc = np.diag(sTrucSVD)

    # QR decomp. St=Q*R, Q unitary, R upper triangular
    qsTruc, rsTruc = np.linalg.qr(sTruc)
    # if R is singular matrix, raise an error
    if np.linalg.det(rsTruc) == 0:
        raise RuntimeError("The R matrix is singlular, Please check the singularity of [X1;U]!")
    beta = X2.dot(vTruc).dot(np.linalg.inv(rsTruc)).dot(qsTruc.T)
    A = beta.dot(uTruc[0:n, :].T)
    B = beta.dot(uTruc[n:, :].T)
    C = Y1.dot(scipy.linalg.pinv(X1))

    return A, B, C

def fcnAggrDMDcDataPrep(X_try, X0data, Udata, Ydata):
    # prepare data for DMDc calculation (subtracting the initial/equlibrium value)
    U1 = np.subtract(Udata[0:-1,:], Udata[0,:]).T    
    Y1 = np.subtract(Ydata[0:-1,:], Ydata[0,:]).T    
    if X0data.size == 0:
        X1_0 = np.array([]); X2_0 = np.array([])
    else:
        X1_0 = np.subtract(X0data[0:-1,:], X0data[0,:]).T 
        X2_0 = np.subtract(X0data[1:,  :], X0data[0,:]).T 

    if X_try.size == 0:
        X_try = np.array([])
    else:
        X_try = np.subtract(X_try, X_try[0,:]).T
    
    return X_try, X1_0, X2_0, U1, Y1


def CostFunc(V_Ref,V_Hat, Wt):
    # Calculate the average and standard deviation for each dimension
    V_avg = np.average(V_Ref,axis=1) # average value of each row
    V_std = np.std(V_Ref,axis=1,ddof=1) # standard deviation of each row
    Nr,Ns = V_Ref.shape # number of rows and columns
    # print(V_avg, V_std)
    
    # Default Weight
    if Wt.size == 0:
        Wt = np.ones((Nr,))

    # Initialize normalized array
    V_Ref_n = np.zeros(V_Ref.shape)
    V_Hat_n = np.zeros(V_Ref.shape)

    # Normalize each array
    for i in range(Nr):
        for j in range(Ns):
            V_Ref_n[i,j] = (V_Ref[i,j]-V_avg[i])/V_std[i]
            V_Hat_n[i,j] = (V_Hat[i,j]-V_avg[i])/V_std[i]
    
    # Calculate Cost
    Cost = 0.
    for i in range(Nr):
        for j in range(Ns):
            Cost += Wt[i]*np.square(V_Ref_n[i,j] - V_Hat_n[i,j])
            # print(i,j,Cost)
    
    return Cost




    