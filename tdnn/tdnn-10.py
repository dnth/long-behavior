import sys
import numpy as np
from pybrain.datasets import SequenceClassificationDataSet, ClassificationDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain import LinearLayer, FullConnection, LSTMLayer, BiasUnit, MDLSTMLayer, IdentityConnection, TanhLayer, SoftmaxLayer, SigmoidLayer
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
from scipy.interpolate import interp1d
# robotIP="192.168.0.108"
# tts=ALProxy("ALTextToSpeech", robotIP, 9559)
# motion = ALProxy("ALMotion", robotIP, 9559)
# memory = ALProxy("ALMemory", robotIP, 9559)
# posture = ALProxy("ALRobotPosture", robotIP, 9559)
# camProxy = ALProxy("ALVideoDevice", robotIP, 9559)
# resolution = 0    # kQQVGA
# colorSpace = 11   # RGB


# tts.say("Hello")
# posture.goToPosture("Crouch", 1.0)
# motion.rest()
#############
# Functions #
#############
   
def plotLearningCurve():
    fig=plt.figure(0, figsize=(10,8) )
    fig.clf()
    plt.ioff()
    plt.subplot(211)
    plt.plot(trn_error, label='Training Set Error', linestyle="--", linewidth=2)
    plt.plot(tst_error, label='Validation Set Error', linewidth=2)
    plt.title('Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
       
    plt.subplot(212)
    plt.plot(trn_class_accu, label='Training Set Accuracy', linestyle="--", linewidth=2)
    plt.plot(tst_class_accu, label='Validation Set Accuracy', linewidth=2)
    plt.ylim([0,103])
    plt.ylabel('Percent')
    plt.xlabel('Epoch')
    plt.title('Classification Accuracy')
    plt.legend(loc=4)
       
#     plt.draw()
    plt.grid(True)
    plt.tight_layout(pad=2.1)
    plt.savefig(figPath)
           
################
# Load Dataset #
################

# Original Joint data
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


trndata = ClassificationDataSet(100,1, nb_classes=8)
tstdata = ClassificationDataSet(100,1, nb_classes=8)

# BL = BallLiftJoint.flatten()
# BR = BallRollJoint.flatten()
# BRL = BellRingLJoint.flatten()
# BRR = BellRingRJoint.flatten()
# BRP = BallRollPlateJoint.flatten()
# RW = RopewayJoint.flatten()

LBL = LBallLift.flatten()
RBL = RBallLift.flatten()
LBR = LBallRoll.flatten()
RBR = RBallRoll.flatten()
LBRP = LBallRollPlate.flatten()
RBRP = RBallRollPlate.flatten()
LRW = LRopeway.flatten()
RRW = RRopeway.flatten()

for i in range(0,120000,2000):
    trndata.addSample(LBL[i:i+100], [0])
    trndata.addSample(RBL[i:i+100], [1])
    trndata.addSample(LBR[i:i+100], [2])
    trndata.addSample(RBR[i:i+100], [3])
    trndata.addSample(LBRP[i:i+100], [4])
    trndata.addSample(RBRP[i:i+100], [5])
    trndata.addSample(LRW[i:i+100], [6])
    trndata.addSample(RRW[i:i+100], [7])
    
for i in range(120000,160000,2000):
    tstdata.addSample(LBL[i:i+100], [0])
    tstdata.addSample(RBL[i:i+100], [1])
    tstdata.addSample(LBR[i:i+100], [2])
    tstdata.addSample(RBR[i:i+100], [3])
    tstdata.addSample(LBRP[i:i+100], [4])
    tstdata.addSample(RBRP[i:i+100], [5])
    tstdata.addSample(LRW[i:i+100], [6])
    tstdata.addSample(RRW[i:i+100], [7])

# for i in range(0,60000, 100):
#     trndata.addSample(BL[i:i+100], [0])
#     trndata.addSample(BR[i:i+100], [1])
#     trndata.addSample(BRL[i:i+100], [2])
#     trndata.addSample(BRR[i:i+100], [3])
#     trndata.addSample(BRP[i:i+100], [4])
#     trndata.addSample(RW[i:i+100], [5])
#     
# for i in range(60000,80000, 100):
#     tstdata.addSample(BL[i:i+100], [0])
#     tstdata.addSample(BR[i:i+100], [1])
#     tstdata.addSample(BRL[i:i+100], [2])
#     tstdata.addSample(BRR[i:i+100], [3])
#     tstdata.addSample(BRP[i:i+100], [4])
#     tstdata.addSample(RW[i:i+100], [5])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()    
print 'Loaded Dataset!'
 
 
print 'Building Network'
TDNNClassificationNet = buildNetwork(trndata.indim,25,trndata.outdim, hiddenclass=SigmoidLayer, outclass=SoftmaxLayer) 
 
print "Number of weights:",TDNNClassificationNet.paramdim
 
trainer = RPropMinusTrainer(TDNNClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
 
print TDNNClassificationNet.paramdim
 
tstErrorCount=0
oldtstError=0
trn_error=[]
tst_error=[]
trn_class_accu=[]
tst_class_accu=[]
         
trnErrorPath='25sigmoid/trn_error'
tstErrorPath='25sigmoid/tst_error'
trnClassErrorPath='25sigmoid/trn_ClassAccu'
tstClassErrorPath='25sigmoid/tst_ClassAccu'
networkPath='25sigmoid/TrainUntilConv.xml'
figPath='25sigmoid/ErrorGraph'
  
#####################
#####################
print "Training Data Length: ", len(trndata)
print "Validation Data Length: ", len(tstdata)
        
                         
# print 'Start Training'
# time_start = time.time()
# while (tstErrorCount<100):
#     print "********** Classification with 25sigmoid with RP- **********"   
#     trnError=trainer.train()
#     tstError = trainer.testOnData(dataset=tstdata)
#     trnAccu = 100-percentError(trainer.testOnClassData(), trndata['class'])
#     tstAccu = 100-percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
#     trn_class_accu.append(trnAccu)
#     tst_class_accu.append(tstAccu)
#     trn_error.append(trnError)
#     tst_error.append(tstError)
#                                                                                                                                                    
#     np.savetxt(trnErrorPath, trn_error)
#     np.savetxt(tstErrorPath, tst_error)
#     np.savetxt(trnClassErrorPath, trn_class_accu)
#     np.savetxt(tstClassErrorPath, tst_class_accu)
#                                                                                                                                                  
#     if(oldtstError==0):
#         oldtstError = tstError
#                                                                                                                                                      
#     if(oldtstError<tstError):
#         tstErrorCount = tstErrorCount+1
#         print 'No Improvement, count=%d' % tstErrorCount
#         print '    Old Validation Error:', oldtstError 
#         print 'Current Validation Error:', tstError
#                                                                                                                                                      
#     if(oldtstError>tstError):
#         print 'Improvement made!'
#         print '    Old Validation Error:', oldtstError 
#         print 'Current Validation Error:', tstError
#         tstErrorCount=0
#         oldtstError = tstError
#         NetworkWriter.writeToFile(TDNNClassificationNet, networkPath)
#         plotLearningCurve()
#             
#            
# trainingTime = time.time()-time_start
# trainingTime=np.reshape(trainingTime, (1))
# np.savetxt("25sigmoid/Trainingtime.txt", trainingTime)


####################
# Manual OFFLINE Test
####################        
# TDNNClassificationNet = NetworkReader.readFrom('25sigmoid/TrainUntilConv.xml')
# trainer = RPropMinusTrainer(TDNNClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
# print 'Loaded Trained Network!'
# print TDNNClassificationNet.paramdim
#  
# testingdata = ClassificationDataSet(100,1, nb_classes=8)
#  
#      
# for i in range(120000,160000,2000):
#     testingdata.addSample(LBL[i:i+100], [0])
#     testingdata.addSample(RBL[i:i+100], [1])
#     testingdata.addSample(LBR[i:i+100], [2])
#     testingdata.addSample(RBR[i:i+100], [3])
#     testingdata.addSample(LBRP[i:i+100], [4])
#     testingdata.addSample(RBRP[i:i+100], [5])
#     testingdata.addSample(LRW[i:i+100], [6])
#     testingdata.addSample(RRW[i:i+100], [7])
#      
# testingdata._convertToOneOfMany()
#  
# testingAccu = 100-percentError(trainer.testOnClassData(dataset=testingdata), testingdata['class'])
# print testingAccu


#  
# testdata = RopewayJoint.flatten()[90000:90100]
# x = TDNNClassificationNet.activate(testdata)
# print argmax(x)