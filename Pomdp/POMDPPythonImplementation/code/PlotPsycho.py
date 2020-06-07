import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import norm
import random
import statistics as st

def compare_matrix(matrix, val):
    rown = len(matrix)
    coln = len(matrix[0])
    ref_matrix = []
    for i in range(0, rown):
        dummy = []
        for j in range(0, coln):
            if matrix[i][j] != val:
                dummy.append(0)
            else:
                dummy.append(1)
        ref_matrix.append(dummy)
    return ref_matrix

def mean(matrix, dim):
    mean_list = []
    if(dim == 0) :
        for i in range(0, len(matrix)):
            mean_list.append(st.mean(matrix[i]))
    else :
        for j in range(0, len(matrix[0])) :
            mean_list.append(st.mean(matrix[:][j]))
    return mean_list



def PlotPsycho(input,output):
    output['action']=[(i+1)/2 for i in output['action'] ]
    perStim = [float("NaN") for i in range(len(list(set(input['stimTrials']))))]
    for iBlock in np.transpose(np.unique((input['extraRewardTrials']))) :
        # inverse => input need to be transposed
        c=1
        for iStim in np.transpose(np.unique(input['stimTrials'])): # inverse => input need to be transposed
            x= compare_matrix(input['extraRewardTrials'],iBlock)
            perStim[c]=mean(output['action'](input['stimTrials']==iStim and x),1)
            c=c+1
    if (len(set(iBlock)) == len(iBlock)) and iBlock[0] == 'left' :
        color = 'b'
    elif (len(set(iBlock)) == len(iBlock)) and iBlock[0] == 'right' :
        color = 'r'
    elif (len(set(iBlock)) == len(iBlock)) and iBlock[0] == 'none' :
        color = 'k' 

    # inverse => input need to be transposed
    plt.plot(np.transpose(list(set(input['stimTrials']))),perStim,Color=color,marker='o',markersize=10,linestyle='dashed',linewidth=1.2 )
    plt.ylim(0,1)
    plt.legend(['Reward after L action','No extra reward','Reward after R action'], loc='upper right')
    plt.ylabel('Fraction rightward choice')
    plt.xlabel('Stimulus')
    plt.title('Psychometric function of POMDP Model')
    plt.show()