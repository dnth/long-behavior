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
    
    

LSTMClassificationNet = NetworkReader.readFrom('20LSTMCell(1)/TrainUntilConv.xml')
print 'Loaded 20 LSTM Trained Network!'

twentylstmaccdata = []
twentylstmstddata = []
twentylstmstderror = []

predictedLBLLabels = []
predictedRBLLabels = []
predictedLBRLabels = []
predictedRBRLabels = []
predictedLBRPLabels = []
predictedRBRPLabels = []
predictedLRWLabels = []
predictedRRWLabels = []

sequenceStartIndex = range(16000,19800,200)


offset = 200
accuracyOverall = []
for testnumber in range(30):    
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(LBallLift[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(BallLiftJoint[i])
    predictedLBLLabels.append(argmax(x))
     
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(RBallLift[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(BallLiftJoint[i])
    predictedRBLLabels.append(argmax(x))
     
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(LBallRoll[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(BallRollJoint[i])
    predictedLBRLabels.append(argmax(x))
     
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(RBallRoll[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(BallRollJoint[i])
    predictedRBRLabels.append(argmax(x))
     
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(LBallRollPlate[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
    predictedLBRPLabels.append(argmax(x))
     
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(RBallRollPlate[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
    predictedRBRPLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(LRopeway[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(RopewayJoint[i])
    predictedLRWLabels.append(argmax(x))
    
    start = np.random.choice(sequenceStartIndex)
    end = start + offset
    timestep=range(start,end)
    LSTMClassificationNet.reset()
    for i in timestep:
        x = LSTMClassificationNet.activate(RRopeway[i])
    for i in range(400):
        x = LSTMClassificationNet.activate(RopewayJoint[i])
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

print len(twentylstmaccdata)