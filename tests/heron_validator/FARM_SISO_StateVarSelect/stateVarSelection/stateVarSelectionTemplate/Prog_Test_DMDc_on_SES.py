import os
import numpy as np
from scipy.interpolate import interp1d
import csv
import matplotlib.pyplot as plt
import math
import time
from tqdm import tqdm
from utils import fcnAggrDMDcDataPrep, DMDc, CostFunc

time_start_general = time.time()
# Specify the file folder and file list
fileFolder = os.path.join(os. getcwd(),'simulatedDataSES')
indexFileName = 'outputData.csv'

# Generate file list
# open the file in read mode
indexFile = open(os.path.join(fileFolder, indexFileName), 'r')
# creating dictreader object
file = csv.DictReader(indexFile)
# creating empty lists
file_list = []
# iterating over each row and append values to empty list
for col in file:
    file_list.append(col['filename'])
# print(file_list)

Num_file = len(file_list)

# Specify the U, Y and X variable names in the CSV
U_Names = ["SES_Demand_MW"]
Y_Names = [
    "Electric_Power", "Firing_Temperature"
    ]
plot_profile = 2

for profile in range(1,3):

    # Windows
    if profile == 1:
        # 1: 1800s - 18000s
        test_variables = ['SES.ED.sensorBus.W_gen', 'SES.generator.P_flow'];
    if profile == 2:
        # 2: 1800s - 5400s
        test_variables = ['SES.CS.feedback_W_gen.u2', 'SES.ED.sensorBus.W_gen', 'SES.sensorBus.W_gen'];
    
    X_Names = test_variables

    # Specify the time step (seconds), for interpolation purpose
    T_Step = 10

    cf_xcand=0.
    # Loop over all the files
    for i_file in range(Num_file):
        time_start_file = time.time()
        # Specify the mat file to read
        file_to_read = os.path.join(fileFolder, file_list[i_file]) # type)file_to_read == <class 'str'>
        """File processing: open, extract csv header, extract csv data"""
        with open(file_to_read, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            line_count = 0
            csvdata = []
            for row in reader:
                if line_count == 0: # for the first row, get the header
                    csvheader = row
                    line_count += 1

                else:
                    csvdata.append(row) # from the 2nd row, get data
                    line_count += 1
            
            # convert csvdata from list to ndarray
            csvdata = np.asarray(csvdata) # csvdata.shape = (50000, 8)
            # close file
            csvfile.close()

        """check if all the U_Names, Y_Names, X_Names are in the csvheader"""
        Header_correct = True
        for item in U_Names:
            if not (item in csvheader):
                print('ERROR: {} from "U_Names" does not exist in csv header.\n'.format(item))
                Header_correct = False
        for item in Y_Names:
            if not (item in csvheader):
                print('ERROR: {} from "Y_Names" does not exist in csv header.\n'.format(item))
                Header_correct = False
        for item in X_Names:
            if not (item in csvheader):
                print('ERROR: {} from "X_Names" does not exist in csv header.\n'.format(item))
                Header_correct = False
        if not Header_correct:
            raise RuntimeError('Check U_Names, Y_Names and X_Names. \n Hint: csvheader={}\n'.format(csvheader))

        """Extract the Time, X, Y, U data from csvdata"""
        Num_Entry, Num_Var = csvdata.shape # Num_Entry = 50000, Num_Var = 8
        csvTime = np.zeros([Num_Entry,1])
        csvUdata = np.zeros([Num_Entry,len(U_Names)]) # csvUdata.shape = (50000,1)
        csvYdata = np.zeros([Num_Entry,len(Y_Names)]) # csvYdata.shape = (50000,2)
        csvXdata = np.zeros([Num_Entry,len(X_Names)]) # csvXdata.shape = (50000,4)
        # Extract Time
        csvTime[:,0] = csvdata[:,csvheader.index('Time')]
        # print(csvTime.shape)
        # Extract Udata
        for item in U_Names:
            idx_csvheader = csvheader.index(item)
            idx_UYXName = U_Names.index(item)
            csvUdata[:,idx_UYXName] = csvdata[:,idx_csvheader]
        # Extract Ydata
        for item in Y_Names:
            idx_csvheader = csvheader.index(item)
            idx_UYXName = Y_Names.index(item)
            csvYdata[:,idx_UYXName] = csvdata[:,idx_csvheader]
        # Extract Udata
        for item in X_Names:
            idx_csvheader = csvheader.index(item)
            idx_UYXName = X_Names.index(item)
            csvXdata[:,idx_UYXName] = csvdata[:,idx_csvheader]
        
        """Interpolate every T_Step seconds"""
        # Time = np.arange(csvTime[0,0],csvTime[-1,0]+T_Step,T_Step)
        # # remove the last Time element if it's out of csvTime range.
        # if Time[-1] > csvTime[-1,0]:
        #     Time = np.delete(Time, -1)
        # # print(len(Time))
        # Udata = np.zeros([len(Time),len(U_Names)]) # Udata.shape = (5000,1)
        # Ydata = np.zeros([len(Time),len(Y_Names)]) # Ydata.shape = (5000,2)
        # Xdata = np.zeros([len(Time),len(X_Names)]) # Xdata.shape = (5000,4)
        # # Interpolate Udata
        # for i in range(len(U_Names)):
        #     f_interp = interp1d(csvTime.flatten(),csvUdata[:,i].flatten(),kind='nearest') # train the interpolation function
        #     # print(f_interp(Time), f_interp(Time).shape)
        #     Udata[:,i] = f_interp(Time) # get the interpolated values associated with the "Time" axis   
        # # Interpolate Ydata
        # for i in range(len(Y_Names)):
        #     f_interp = interp1d(csvTime.flatten(),csvYdata[:,i].flatten(),kind='nearest') # train the interpolation function
        #     Ydata[:,i] = f_interp(Time) # get the interpolated values associated with the "Time" axis
        # # Interpolate Xdata
        # for i in range(len(X_Names)):
        #     f_interp = interp1d(csvTime.flatten(),csvXdata[:,i].flatten(),kind='nearest') # train the interpolation function
        #     Xdata[:,i] = f_interp(Time) # get the interpolated values associated with the "Time" axis

        uNorm = csvUdata[0,:]; xNorm = csvXdata[0,:]; yNorm = csvYdata[0,:]
        # print(uNorm,uNorm.shape); print(xNorm); print(yNorm)
        Time = csvTime; Udata = csvUdata; Xdata = csvXdata; Ydata = csvYdata

        """By now, all the data are stored in Time, Udata, Ydata, Xdata"""

        """prep data for DMDc"""
        X_try, X1_0, X2_0, U1, Y1 = fcnAggrDMDcDataPrep(np.array([]), Xdata, Udata, Ydata)
        # print(U1.shape)
        # print(Y1.shape)
        # print(X1_0.shape)
        
        """Run DMDc"""
        # A, B, C = DMDc(X1, X2, U, Y1, rankSVD (0 for auto, -1 for no, +N for N rank), MaxCondNum):
        A_idf, B_idf, C_idf = DMDc(X1_0, X2_0, U1, Y1, -1)
        D_idf = np.zeros([len(Y_Names),len(U_Names)])
        # print(A_idf)
        # print(B_idf)
        # print(C_idf)
        # print(D_idf)

        """ Calculate the self-progress """
        X1_self = np.zeros(X1_0.shape)
        Y1_self = np.zeros(Y1.shape)
        n, Nt1 = X1_0.shape # Nt1 == 49
        p, _ = Y1.shape

        
        for k in range(0, Nt1):
            if k < Nt1-1:
                X1_self[:,k+1] = A_idf.dot(X1_self[:,k]) + B_idf.dot(U1[:,k])
            Y1_self[:,k] = C_idf.dot(X1_self[:,k])

        """ Calculate the cost function """
        # cost_file=DMDcBruteForceTrial([],X1_0,X2_0,U1,Y1);

        V_Ref = np.concatenate((X1_0,Y1),axis=0)
        V_Hat = np.concatenate((X1_self,Y1_self),axis=0)
        Wt = np.concatenate(((1/n)*np.ones(n,), (1/p)*np.ones(p,)), axis=0)
        # print(n,p,Wt)
        cost_cand_file = CostFunc(V_Ref,V_Hat,Wt)
        # accumulate the cost function over files
        cf_xcand = cf_xcand + cost_cand_file

        # calculate time difference
        time_file = time.time() - time_start_file
        # print("File #%3d loaded, cost function = %.4f, deltaT= %.2f" %(i_file,cost_cand_file,time_file))
    cf_xcand = cf_xcand/Nt1
    print("Profile %2d, %2d x Vars, %3d Files loaded, total cost function = %.4e, deltaT= %.2f" %(profile, n, Num_file,cf_xcand,time.time() - time_start_general))

    """ Plot section """
    if profile == plot_profile:
        # number of rows & columns in the subplot
        nPlotCols = 5
        nPlotRows = math.ceil(len(X_Names)/nPlotCols)+2
        
        fig1, axs = plt.subplots(nPlotRows,nPlotCols)
        fig1.set_size_inches(16, 10)
        subplot_offset = 0
        # 1. actuation signal U
        for i in range(len(U_Names)):
            idx_row = math.floor(subplot_offset/nPlotCols)
            idx_col = subplot_offset % nPlotCols
            axs[idx_row,idx_col].title.set_text('U[{}]:{}'.format(i,U_Names[i]))
            axs[idx_row,idx_col].step(Time[0:-1], U1[i,:]+uNorm[i], linestyle='-', color='#D95319', linewidth=3)  
            axs[idx_row,idx_col].legend(['Training Data'], loc='best')      
            subplot_offset += 1
        subplot_offset = nPlotCols
        # 2. Output signal Y
        for i in range(len(Y_Names)):
            idx_row = math.floor(subplot_offset/nPlotCols)
            idx_col = subplot_offset % nPlotCols
            axs[idx_row,idx_col].title.set_text('Y[{}]:{}'.format(i,Y_Names[i]))
            axs[idx_row,idx_col].step(Time[0:-1], Y1[i,:]+yNorm[i], linestyle='--', color='#4DBEEE', linewidth=3)
            axs[idx_row,idx_col].step(Time[0:-1], Y1_self[i,:]+yNorm[i], linestyle='-', color='b', linewidth=1)
            axs[idx_row,idx_col].legend(['Training Data','DMDc Model'], loc='best')      
            subplot_offset += 1
        subplot_offset = nPlotCols*2
        # 3. Output signal X
        for i in range(len(X_Names)):
            idx_row = math.floor(subplot_offset/nPlotCols)
            idx_col = subplot_offset % nPlotCols
            axs[idx_row,idx_col].title.set_text('X[{}]:{}'.format(i,X_Names[i]))
            axs[idx_row,idx_col].step(Time[0:-1], X1_0[i,:]+xNorm[i], linestyle='--', color='#77AC30', linewidth=3)
            axs[idx_row,idx_col].step(Time[0:-1], X1_self[i,:]+xNorm[i], linestyle='-', color='c', linewidth=1)
            axs[idx_row,idx_col].legend(['Training Data','DMDc Model'], loc='best')      
            subplot_offset += 1

        fig1.suptitle('Self-Evolution Plot #%.2d, use all zero X[1] to propagate X[k>1] and Y[k>1], CostFunc=%.4e\n %s'%(i_file, cf_xcand,file_list[i_file]))

plt.show()



    