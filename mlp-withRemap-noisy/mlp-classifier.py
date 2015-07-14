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

jointRemap = interp1d([-2.2,2.2],[-1,1])
BallLiftJoint = jointRemap(BallLiftJoint)
BallRollJoint = jointRemap(BallRollJoint)
BellRingLJoint = jointRemap(BellRingLJoint)
BellRingRJoint = jointRemap(BellRingRJoint)
BallRollPlateJoint = jointRemap(BallRollPlateJoint)
RopewayJoint = jointRemap(RopewayJoint)

LBallLift = np.vstack((BellRingLJoint[0:200], BallLiftJoint[0:200]))
RBallLift = np.vstack((BellRingRJoint[0:200], BallLiftJoint[0:200]))
LBallRoll = np.vstack((BellRingLJoint[0:200], BallRollJoint[0:200]))
RBallRoll = np.vstack((BellRingRJoint[0:200], BallRollJoint[0:200]))
LBallRollPlate = np.vstack((BellRingLJoint[0:200], BallRollPlateJoint[0:200]))
RBallRollPlate = np.vstack((BellRingRJoint[0:200], BallRollPlateJoint[0:200]))
LRopeway = np.vstack((BellRingLJoint[0:200], RopewayJoint[0:200]))
RRopeway = np.vstack((BellRingRJoint[0:200], RopewayJoint[0:200]))

for i in range(200,10000,200):
    LBallLift = np.vstack((LBallLift, BellRingLJoint[i:i+200], BallLiftJoint[i:i+200]))
for i in range(200,10000,200):
    RBallLift = np.vstack((RBallLift, BellRingRJoint[i:i+200], BallLiftJoint[i:i+200]))
    
for i in range(200,10000,200):
    LBallRoll = np.vstack((LBallRoll, BellRingLJoint[i:i+200], BallRollJoint[i:i+200]))
for i in range(200,10000,200):
    RBallRoll = np.vstack((RBallRoll, BellRingRJoint[i:i+200], BallRollJoint[i:i+200]))

for i in range(200,10000,200):
    LBallRollPlate = np.vstack((LBallRollPlate, BellRingLJoint[i:i+200], BallRollPlateJoint[i:i+200]))
for i in range(200,10000,200):
    RBallRollPlate = np.vstack((RBallRollPlate, BellRingRJoint[i:i+200], BallRollPlateJoint[i:i+200]))
    
for i in range(200,10000,200):
    LRopeway = np.vstack((LRopeway, BellRingLJoint[i:i+200], RopewayJoint[i:i+200]))
for i in range(200,10000,200):
    RRopeway = np.vstack((RRopeway, BellRingRJoint[i:i+200], RopewayJoint[i:i+200]))

print LBallLift.shape
print RBallLift.shape
print LBallRoll.shape
print RBallRoll.shape
print LBallRollPlate.shape
print RBallRollPlate.shape
print LRopeway.shape
print RRopeway.shape
    
trndata = ClassificationDataSet(10,1, nb_classes=8)
tstdata = ClassificationDataSet(10,1, nb_classes=8)

for i in range(12000):
    trndata.appendLinked(LBallLift[i,:], [0])
for i in range(12000):
    trndata.appendLinked(RBallLift[i,:], [1])
for i in range(12000):
    trndata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000):
    trndata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000):
    trndata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000):
    trndata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000):
    trndata.appendLinked(LRopeway[i,:], [6])
for i in range(12000):
    trndata.appendLinked(RRopeway[i,:], [7])
    
for i in range(12000,16000):
    tstdata.appendLinked(LBallLift[i,:], [0])
for i in range(12000,16000):
    tstdata.appendLinked(RBallLift[i,:], [1])
for i in range(12000,16000):
    tstdata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000,16000):
    tstdata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000,16000):
    tstdata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000,16000):
    tstdata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000,16000):
    tstdata.appendLinked(LRopeway[i,:], [6])
for i in range(12000,16000):
    tstdata.appendLinked(RRopeway[i,:], [7])
    
# superimpose noise on data as additional dataset    
mean = 0
std_deviation = 1
BallLiftJoint = BallLiftJoint + np.random.normal(mean,std_deviation,(10000,10))
BallRollJoint = BallRollJoint + np.random.normal(mean,std_deviation,(10000,10))
BellRingLJoint = BellRingLJoint + np.random.normal(mean,std_deviation,(10000,10))
BellRingRJoint = BellRingRJoint + np.random.normal(mean,std_deviation,(10000,10))
BallRollPlateJoint = BallRollPlateJoint + np.random.normal(mean,std_deviation,(10000,10))
RopewayJoint = RopewayJoint + np.random.normal(mean,std_deviation,(10000,10))

LBallLift = np.vstack((BellRingLJoint[0:200], BallLiftJoint[0:200]))
RBallLift = np.vstack((BellRingRJoint[0:200], BallLiftJoint[0:200]))
LBallRoll = np.vstack((BellRingLJoint[0:200], BallRollJoint[0:200]))
RBallRoll = np.vstack((BellRingRJoint[0:200], BallRollJoint[0:200]))
LBallRollPlate = np.vstack((BellRingLJoint[0:200], BallRollPlateJoint[0:200]))
RBallRollPlate = np.vstack((BellRingRJoint[0:200], BallRollPlateJoint[0:200]))
LRopeway = np.vstack((BellRingLJoint[0:200], RopewayJoint[0:200]))
RRopeway = np.vstack((BellRingRJoint[0:200], RopewayJoint[0:200]))

for i in range(200,10000,200):
    LBallLift = np.vstack((LBallLift, BellRingLJoint[i:i+200], BallLiftJoint[i:i+200]))
for i in range(200,10000,200):
    RBallLift = np.vstack((RBallLift, BellRingRJoint[i:i+200], BallLiftJoint[i:i+200]))
    
for i in range(200,10000,200):
    LBallRoll = np.vstack((LBallRoll, BellRingLJoint[i:i+200], BallRollJoint[i:i+200]))
for i in range(200,10000,200):
    RBallRoll = np.vstack((RBallRoll, BellRingRJoint[i:i+200], BallRollJoint[i:i+200]))

for i in range(200,10000,200):
    LBallRollPlate = np.vstack((LBallRollPlate, BellRingLJoint[i:i+200], BallRollPlateJoint[i:i+200]))
for i in range(200,10000,200):
    RBallRollPlate = np.vstack((RBallRollPlate, BellRingRJoint[i:i+200], BallRollPlateJoint[i:i+200]))
    
for i in range(200,10000,200):
    LRopeway = np.vstack((LRopeway, BellRingLJoint[i:i+200], RopewayJoint[i:i+200]))
for i in range(200,10000,200):
    RRopeway = np.vstack((RRopeway, BellRingRJoint[i:i+200], RopewayJoint[i:i+200]))

print LBallLift.shape
print RBallLift.shape
print LBallRoll.shape
print RBallRoll.shape
print LBallRollPlate.shape
print RBallRollPlate.shape
print LRopeway.shape
print RRopeway.shape

for i in range(12000):
    trndata.appendLinked(LBallLift[i,:], [0])
for i in range(12000):
    trndata.appendLinked(RBallLift[i,:], [1])
for i in range(12000):
    trndata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000):
    trndata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000):
    trndata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000):
    trndata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000):
    trndata.appendLinked(LRopeway[i,:], [6])
for i in range(12000):
    trndata.appendLinked(RRopeway[i,:], [7])
    
for i in range(12000,16000):
    tstdata.appendLinked(LBallLift[i,:], [0])
for i in range(12000,16000):
    tstdata.appendLinked(RBallLift[i,:], [1])
for i in range(12000,16000):
    tstdata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000,16000):
    tstdata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000,16000):
    tstdata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000,16000):
    tstdata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000,16000):
    tstdata.appendLinked(LRopeway[i,:], [6])
for i in range(12000,16000):
    tstdata.appendLinked(RRopeway[i,:], [7])
    

    

    
    
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()    
print 'Loaded Dataset!'

#####################
#####################
     
print 'Building Network'
MLPClassificationNet = buildNetwork(trndata.indim,139,trndata.outdim, hiddenclass=SigmoidLayer, 
                                     outclass=SoftmaxLayer, bias=True) 

print "Number of weights:",MLPClassificationNet.paramdim

trainer = RPropMinusTrainer(MLPClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
    
tstErrorCount=0
oldtstError=0
trn_error=[]
tst_error=[]
trn_class_accu=[]
tst_class_accu=[]
        
trnErrorPath='139sigmoid/trn_error'
tstErrorPath='139sigmoid/tst_error'
trnClassErrorPath='139sigmoid/trn_ClassAccu'
tstClassErrorPath='139sigmoid/tst_ClassAccu'
networkPath='139sigmoid/TrainUntilConv.xml'
figPath='139sigmoid/ErrorGraph'
 
#####################
#####################
print "Training Data Length: ", len(trndata)
print "Validation Data Length: ", len(tstdata)
 
                  
print 'Start Training'
time_start = time.time()
while (tstErrorCount<100):
    print "********** Classification with 139sigmoid with RP- **********"   
    trnError=trainer.train()
    tstError = trainer.testOnData(dataset=tstdata)
    trnAccu = 100-percentError(trainer.testOnClassData(), trndata['class'])
    tstAccu = 100-percentError(trainer.testOnClassData(dataset=tstdata), tstdata['class'])
    trn_class_accu.append(trnAccu)
    tst_class_accu.append(tstAccu)
    trn_error.append(trnError)
    tst_error.append(tstError)
                                                                                                                                            
    np.savetxt(trnErrorPath, trn_error)
    np.savetxt(tstErrorPath, tst_error)
    np.savetxt(trnClassErrorPath, trn_class_accu)
    np.savetxt(tstClassErrorPath, tst_class_accu)
                                                                                                                                          
    if(oldtstError==0):
        oldtstError = tstError
                                                                                                                                              
    if(oldtstError<tstError):
        tstErrorCount = tstErrorCount+1
        print 'No Improvement, count=%d' % tstErrorCount
        print '    Old Validation Error:', oldtstError 
        print 'Current Validation Error:', tstError
                                                                                                                                              
    if(oldtstError>tstError):
        print 'Improvement made!'
        print '    Old Validation Error:', oldtstError 
        print 'Current Validation Error:', tstError
        tstErrorCount=0
        oldtstError = tstError
        NetworkWriter.writeToFile(MLPClassificationNet, networkPath)
        plotLearningCurve()
     
    
trainingTime = time.time()-time_start
trainingTime=np.reshape(trainingTime, (1))
np.savetxt("139sigmoid/Trainingtime.txt", trainingTime)


####################
# Manual OFFLINE Test
####################        
MLPClassificationNet = NetworkReader.readFrom('139sigmoid//TrainUntilConv.xml')
print 'Loaded Trained Network!'
from random import randint

print MLPClassificationNet.paramdim


x = MLPClassificationNet.activate(RBallLift[10])
print argmax(x)