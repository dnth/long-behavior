import sys
import numpy as np
from pybrain.datasets import SequenceClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain import LinearLayer, FullConnection, LSTMLayer, BiasUnit, MDLSTMLayer, IdentityConnection, TanhLayer, SoftmaxLayer
from pybrain.utilities import percentError
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from naoqi import ALProxy
import Image
import time
import theanets
import vision_definitions
from numpy.random.mtrand import randint
from numpy import argmax
from random import randint
from scipy.interpolate import interp1d

BallLiftJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallLift/JointData.txt').astype(np.float32)
BallRollJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallRoll/JointData.txt').astype(np.float32)
BellRingLJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BellRingL/JointData.txt').astype(np.float32)
BellRingRJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BellRingR/JointData.txt').astype(np.float32)
BallRollPlateJoint = np.loadtxt('../../20fpsFullBehaviorSampling/BallRollPlate/JointData.txt').astype(np.float32)
RopewayJoint = np.loadtxt('../../20fpsFullBehaviorSampling/Ropeway/JointData.txt').astype(np.float32)

LBallLift = np.vstack((BellRingLJoint[0:100], BallLiftJoint[0:100]))
RBallLift = np.vstack((BellRingRJoint[0:100], BallLiftJoint[0:100]))
LBallRoll = np.vstack((BellRingLJoint[0:100], BallRollJoint[0:100]))
RBallRoll = np.vstack((BellRingRJoint[0:100], BallRollJoint[0:100]))
LBallRollPlate = np.vstack((BellRingLJoint[0:100], BallRollPlateJoint[0:100]))
RBallRollPlate = np.vstack((BellRingRJoint[0:100], BallRollPlateJoint[0:100]))
LRopeway = np.vstack((BellRingLJoint[0:100], RopewayJoint[0:100]))
RRopeway = np.vstack((BellRingRJoint[0:100], RopewayJoint[0:100]))

for i in range(100,10000,100):
    LBallLift = np.vstack((LBallLift, BellRingLJoint[i:i+100], BallLiftJoint[i:i+100]))
for i in range(100,10000,100):
    RBallLift = np.vstack((RBallLift, BellRingRJoint[i:i+100], BallLiftJoint[i:i+100]))
    
for i in range(100,10000,100):
    LBallRoll = np.vstack((LBallRoll, BellRingLJoint[i:i+100], BallRollJoint[i:i+100]))
for i in range(100,10000,100):
    RBallRoll = np.vstack((RBallRoll, BellRingRJoint[i:i+100], BallRollJoint[i:i+100]))

for i in range(100,10000,100):
    LBallRollPlate = np.vstack((LBallRollPlate, BellRingLJoint[i:i+100], BallRollPlateJoint[i:i+100]))
for i in range(100,10000,100):
    RBallRollPlate = np.vstack((RBallRollPlate, BellRingRJoint[i:i+100], BallRollPlateJoint[i:i+100]))
    
for i in range(100,10000,100):
    LRopeway = np.vstack((LRopeway, BellRingLJoint[i:i+100], RopewayJoint[i:i+100]))
for i in range(100,10000,100):
    RRopeway = np.vstack((RRopeway, BellRingRJoint[i:i+100], RopewayJoint[i:i+100]))

tdnnclassifier = NetworkReader.readFrom('25sigmoid/TrainUntilConv.xml')
print 'Loaded 25 sigmoid TDNN Trained Network!'

twentylstmaccdata = []
twentylstmstddata = []
twentylstmstderror = []

# predictedBLLabels = []
# predictedBRLabels = []
# predictedBRLLabels = []
# predictedBRRLabels = []
# predictedBRPLabels = []
# predictedRWLabels = []


predictedLBLLabels = []
predictedRBLLabels = []
predictedLBRLabels = []
predictedRBRLabels = []
predictedLBRPLabels = []
predictedRBRPLabels = []
predictedLRWLabels = []
predictedRRWLabels = []

print "1st Iteration, noiseless test data"
sequenceStartIndex = range(16000,19800,200)
offset = 200
accuracyOverall = []
for testnumber in range(30):    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(LBallLift[start:start+10].flatten())
    predictedLBLLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(RBallLift[start:start+10].flatten())
    predictedRBLLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(LBallRoll[start:start+10].flatten())
    predictedLBRLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(RBallRoll[start:start+10].flatten())
    predictedRBRLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(LBallRollPlate[start:start+10].flatten())
    predictedLBRPLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(RBallRollPlate[start:start+10].flatten())
    predictedRBRPLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(LRopeway[start:start+10].flatten())
    predictedLRWLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    x = tdnnclassifier.activate(RRopeway[start:start+10].flatten())
    predictedRRWLabels.append(argmax(x))
    
testnumAcc = []
behaviorAccuracyfortestnumber = []
for testnumber in range(30):
    LBLAcc = 100-percentError(predictedLBLLabels[testnumber], [0])
    RBLAcc = 100-percentError(predictedRBLLabels[testnumber], [1])
    LBRAcc = 100-percentError(predictedLBRLabels[testnumber], [2])
    RBRAcc = 100-percentError(predictedRBRLabels[testnumber], [3])
    LBRPAcc = 100-percentError(predictedLBRPLabels[testnumber], [4])
    RBRPAcc = 100-percentError(predictedRBRPLabels[testnumber], [5])
    LRWAcc = 100-percentError(predictedLRWLabels[testnumber], [6])
    RRWAcc = 100-percentError(predictedRRWLabels[testnumber], [7])
      
    behaviorAccuracyfortestnumber.append((LBLAcc + RBLAcc + LBRAcc + RBRAcc + LBRPAcc + RBRPAcc + LRWAcc + RRWAcc) / 8)
      
print behaviorAccuracyfortestnumber
  
print "Mean Accuracy for 30 trials:", np.mean(np.array(behaviorAccuracyfortestnumber))
print "Std Deviation for 30 trials:", np.std(np.array(behaviorAccuracyfortestnumber))
 
twentylstmaccdata.append(np.mean(np.array(behaviorAccuracyfortestnumber)))
twentylstmstddata.append(np.std(np.array(behaviorAccuracyfortestnumber)))
 
print "Length of data (iteration number):",len(twentylstmaccdata)
 
# ######## with noise ######
std_deviation = 0
mean = 0
while (std_deviation<=2.0):
    std_deviation += 0.1
    print "Gaussian Noise std deviation:",std_deviation
    predictedLBLLabels = []
    predictedRBLLabels = []
    predictedLBRLabels = []
    predictedRBRLabels = []
    predictedLBRPLabels = []
    predictedRBRPLabels = []
    predictedLRWLabels = []
    predictedRRWLabels = []
       
       
    offset = 200
    accuracyOverall = []
    for testnumber in range(30): # test for 30 times
        BallLiftJoint = BallLiftJoint + np.random.normal(mean,std_deviation,(10000,10))
        BallRollJoint = BallRollJoint + np.random.normal(mean,std_deviation,(10000,10))
        BellRingLJoint = BellRingLJoint + np.random.normal(mean,std_deviation,(10000,10))
        BellRingRJoint = BellRingRJoint + np.random.normal(mean,std_deviation,(10000,10))
        BallRollPlateJoint = BallRollPlateJoint + np.random.normal(mean,std_deviation,(10000,10))
        RopewayJoint = RopewayJoint + np.random.normal(mean,std_deviation,(10000,10))
        
        
        # reconstruct the hybrind beahvior after inject noise into original behaviors
        LBallLift = np.vstack((BellRingLJoint[0:100], BallLiftJoint[0:100]))
        RBallLift = np.vstack((BellRingRJoint[0:100], BallLiftJoint[0:100]))
        LBallRoll = np.vstack((BellRingLJoint[0:100], BallRollJoint[0:100]))
        RBallRoll = np.vstack((BellRingRJoint[0:100], BallRollJoint[0:100]))
        LBallRollPlate = np.vstack((BellRingLJoint[0:100], BallRollPlateJoint[0:100]))
        RBallRollPlate = np.vstack((BellRingRJoint[0:100], BallRollPlateJoint[0:100]))
        LRopeway = np.vstack((BellRingLJoint[0:100], RopewayJoint[0:100]))
        RRopeway = np.vstack((BellRingRJoint[0:100], RopewayJoint[0:100]))
        
        for i in range(100,10000,100):
            LBallLift = np.vstack((LBallLift, BellRingLJoint[i:i+100], BallLiftJoint[i:i+100]))
        for i in range(100,10000,100):
            RBallLift = np.vstack((RBallLift, BellRingRJoint[i:i+100], BallLiftJoint[i:i+100]))
            
        for i in range(100,10000,100):
            LBallRoll = np.vstack((LBallRoll, BellRingLJoint[i:i+100], BallRollJoint[i:i+100]))
        for i in range(100,10000,100):
            RBallRoll = np.vstack((RBallRoll, BellRingRJoint[i:i+100], BallRollJoint[i:i+100]))
        
        for i in range(100,10000,100):
            LBallRollPlate = np.vstack((LBallRollPlate, BellRingLJoint[i:i+100], BallRollPlateJoint[i:i+100]))
        for i in range(100,10000,100):
            RBallRollPlate = np.vstack((RBallRollPlate, BellRingRJoint[i:i+100], BallRollPlateJoint[i:i+100]))
            
        for i in range(100,10000,100):
            LRopeway = np.vstack((LRopeway, BellRingLJoint[i:i+100], RopewayJoint[i:i+100]))
        for i in range(100,10000,100):
            RRopeway = np.vstack((RRopeway, BellRingRJoint[i:i+100], RopewayJoint[i:i+100]))
          
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(LBallLift[start:start+10].flatten())
        predictedLBLLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(RBallLift[start:start+10].flatten())
        predictedRBLLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(LBallRoll[start:start+10].flatten())
        predictedLBRLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(RBallRoll[start:start+10].flatten())
        predictedRBRLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(LBallRollPlate[start:start+10].flatten())
        predictedLBRPLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(RBallRollPlate[start:start+10].flatten())
        predictedRBRPLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(LRopeway[start:start+10].flatten())
        predictedLRWLabels.append(argmax(x))
        
        start = np.random.choice(sequenceStartIndex)
        x = tdnnclassifier.activate(RRopeway[start:start+10].flatten())
        predictedRRWLabels.append(argmax(x))
          
          
    testnumAcc = []
    behaviorAccuracyfortestnumber = []
    for testnumber in range(30):
        LBLAcc = 100-percentError(predictedLBLLabels[testnumber], [0])
        RBLAcc = 100-percentError(predictedRBLLabels[testnumber], [1])
        LBRAcc = 100-percentError(predictedLBRLabels[testnumber], [2])
        RBRAcc = 100-percentError(predictedRBRLabels[testnumber], [3])
        LBRPAcc = 100-percentError(predictedLBRPLabels[testnumber], [4])
        RBRPAcc = 100-percentError(predictedRBRPLabels[testnumber], [5])
        LRWAcc = 100-percentError(predictedLRWLabels[testnumber], [6])
        RRWAcc = 100-percentError(predictedRRWLabels[testnumber], [7])
          
        behaviorAccuracyfortestnumber.append((LBLAcc + RBLAcc + LBRAcc + RBRAcc + LBRPAcc + RBRPAcc + LRWAcc + RRWAcc) / 8)
          
#     print behaviorAccuracyfortestnumber
  
    print "Mean Accuracy for 30 trials:", np.mean(np.array(behaviorAccuracyfortestnumber))
    print "Std Deviation for 30 trials:", np.std(np.array(behaviorAccuracyfortestnumber))
      
    twentylstmaccdata.append(np.mean(np.array(behaviorAccuracyfortestnumber)))
    twentylstmstddata.append(np.std(np.array(behaviorAccuracyfortestnumber)))
      
    print "Length of data (iteration number)",len(twentylstmaccdata)
      
  
print twentylstmaccdata
print twentylstmstddata
  
for i in range(21):
    twentylstmstderror.append(twentylstmstddata[i]/np.sqrt(30))
      
print twentylstmstderror

np.savetxt("AccuracyData.txt",twentylstmaccdata )
np.savetxt("SigmaData.txt",twentylstmstddata )
np.savetxt("ErrorBarData.txt",twentylstmstderror )

      
  
plt.errorbar(y=twentylstmaccdata, x=np.arange(0.0,2.1,0.1), yerr=twentylstmstderror, linewidth=2)
plt.xlim([0.0,2.1])
plt.xlabel(r"$\sigma$")
plt.ylabel("Classification Accuracy (%)")
plt.grid()
plt.legend()
plt.show()