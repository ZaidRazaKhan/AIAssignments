import numpy as np
from matplotlib import pyplot as plt

def smooth(array,window):
    ans = np.zeros(array.shape)
    for i in range(len(array)):
        s=0
        halfwin = int(window/2)
        for j in range(max(0,i-halfwin),min(len(array),i+halfwin+1)):
           s+=1
           ans[i]+=array[j]
        ans[i]/=s   
    return ans    
        
def PlotTrials(input,output):
    output['action'] = np.asarray(output['action'])
    output['action'] = (1+output['action'])/2
    plotLen = 2000
    plotLenVec = np.concatenate((np.ones(plotLen),np.zeros(len(input['stimTrials']))),axis=0)
    action  = output['action'][:plotLen]
    
    smoothAction = smooth(action,7)
    
    blockLeft = np.zeros(plotLen)
    for i in range(plotLen):
        if input['extraRewardTrials'][i][0]=='left':
            blockLeft[i]=-0.03
        else:
            blockLeft[i]=float("NaN")

    blockRight = np.zeros(plotLen)
    for i in range(plotLen):
        if input['extraRewardTrials'][i][0]=='right':
            blockRight[i]=1.03
        else:
            blockRight[i]=float("NaN")
            
    blockNone = np.zeros(plotLen)
    for i in range(plotLen):
        if input['extraRewardTrials'][i][0]=='none':
            blockNone[i]=0.5
        else:
            blockNone[i]=float("NaN")        
            
    plt.xlabel('Trial number')
    plt.ylabel('Fraction rightward choice')
    #print(smoothAction)
    plt.plot(smoothAction,color='gray',label='Actions')
    plt.plot(blockLeft,marker='o',color='green',label='Left reward')
    plt.plot(blockNone,marker='o',color='black',label='No extra reward')
    plt.plot(blockRight,marker='o',color='red',label='Right reward')
    plt.legend()
    plt.show()