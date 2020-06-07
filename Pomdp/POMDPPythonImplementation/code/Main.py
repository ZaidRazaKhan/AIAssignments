import numpy as np
from Runpomdp import RunPOMDP
from PlotPsycho import PlotPsycho
from PlotTrials import PlotTrials

###########################################################################################3
learningRate = 0.35
extraRewardVal = 4.0
beliefNoiseSTD = 0.18

params = [learningRate, extraRewardVal, beliefNoiseSTD]

trialN = 4000
blockN = 20
extraReward = ['right', 'left', 'none']
stimulus = [-0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5]

input = {}
input["stimTrials"] = [0]*trialN
print(input["stimTrials"][2])
for i in range(trialN):
    input["stimTrials"][i] = stimulus[int(np.random.uniform(len(stimulus)))]
    
midBlockLen = trialN/blockN
minBlockLen = 0.75*midBlockLen
maxBlockLen = 1.25*midBlockLen
rangeBlockLen = [i for i in range(int(minBlockLen),int(maxBlockLen))]
blockLen = np.random.choice(rangeBlockLen,size=blockN-1,replace=True)

print(np.cumsum(blockLen))
#somebody check this part
if sum(blockLen)<trialN:
	blockLen=np.append(blockLen,trialN-sum(blockLen))
	print('if')
else:
    blockLen[9] = trialN - sum(blockLen[0:7])

print('--6')
blockLenCumul = np.cumsum(blockLen)
input["extraRewardTrials"] = [[] for i in range(trialN)]#cell(trialN,1) #somebody translate this

blockID = [[] for i in range(blockN)]; #somebody translate this
blockID[1] = np.random.choice(extraReward, size=1, replace=True)

def Diff(li1, li2): 
    return (list(set(li1) - set(li2))) 
#check these loops
for i in range(1, blockN):
	blockID[i] = np.random.choice(Diff(extraReward,blockID[i-1]), size=1, replace=True) #check that curly bracket
print(blockLenCumul)
for i in range(blockLenCumul[1]):
	input["extraRewardTrials"][i] = blockID[1]
for i in range(1, blockN):
	for j in range(blockLenCumul[i-1],blockLenCumul[i]):
		input["extraRewardTrials"][j] = blockID[i]

output = RunPOMDP(input,params)
print(input)
print(output)
# PlotPsycho(input,output)

PlotTrials(input,output)

