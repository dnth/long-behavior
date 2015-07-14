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

print LBallLift.shape
print RBallLift.shape
print LBallRoll.shape
print RBallRoll.shape
print LBallRollPlate.shape
print RBallRollPlate.shape
print LRopeway.shape
print RRopeway.shape

    
trndata = SequenceClassificationDataSet(10,1, nb_classes=8)
tstdata = SequenceClassificationDataSet(10,1, nb_classes=8)

for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LBallLift[i,:], [0])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RBallLift[i,:], [1])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LRopeway[i,:], [6])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RRopeway[i,:], [7])
    
    
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LBallLift[i,:], [0])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RBallLift[i,:], [1])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LRopeway[i,:], [6])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
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

print LBallLift.shape
print RBallLift.shape
print LBallRoll.shape
print RBallRoll.shape
print LBallRollPlate.shape
print RBallRollPlate.shape
print LRopeway.shape
print RRopeway.shape



for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LBallLift[i,:], [0])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RBallLift[i,:], [1])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(LRopeway[i,:], [6])
for i in range(12000):
    if i%200==0:
        trndata.newSequence()
    trndata.appendLinked(RRopeway[i,:], [7])
    
    
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LBallLift[i,:], [0])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RBallLift[i,:], [1])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LBallRoll[i,:], [2])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RBallRoll[i,:], [3])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LBallRollPlate[i,:], [4])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RBallRollPlate[i,:], [5])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(LRopeway[i,:], [6])
for i in range(12000,16000):
    if i%200==0:
        tstdata.newSequence()
    tstdata.appendLinked(RRopeway[i,:], [7])

trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()    
print 'Loaded Dataset!'

#####################
#####################
     
print 'Building Network'
LSTMClassificationNet = buildNetwork(trndata.indim,20,trndata.outdim, hiddenclass=LSTMLayer, 
                                     outclass=SoftmaxLayer, bias=True, recurrent=True, outputbias=False) 
print "Total Number of weights:",LSTMClassificationNet.paramdim
trainer = RPropMinusTrainer(LSTMClassificationNet, dataset=trndata, verbose=True, weightdecay=0.01)
    
tstErrorCount=0
oldtstError=0
trn_error=[]
tst_error=[]
trn_class_accu=[]
tst_class_accu=[]
        
trnErrorPath='20LSTMCell/trn_error'
tstErrorPath='20LSTMCell/tst_error'
trnClassErrorPath='20LSTMCell/trn_ClassAccu'
tstClassErrorPath='20LSTMCell/tst_ClassAccu'
networkPath='20LSTMCell/TrainUntilConv.xml'
figPath='20LSTMCell/ErrorGraph'
 
#####################
#####################
print "Training Data Length: ", len(trndata)
print "Num of Training Seq: ", trndata.getNumSequences()
print "Validation Data Length: ", len(tstdata)
print "Num of Validation Seq: ", tstdata.getNumSequences()
                    
print 'Start Training'
time_start = time.time()
while (tstErrorCount<100):
    print "********** Classification with 20LSTMCell with RP- **********"   
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
        NetworkWriter.writeToFile(LSTMClassificationNet, networkPath)
        plotLearningCurve()
      
     
trainingTime = time.time()-time_start
trainingTime=np.reshape(trainingTime, (1))
np.savetxt("20LSTMCell(1)/Trainingtime.txt", trainingTime)

#####################
# Online Live Test# 6 Behaviors POLISHED
#####################     
# LSTMClassificationNet = NetworkReader.readFrom(networkPath)
# e.load("../../../autoencoder/OverallBehavior-deep-AE.gz")
# print 'Loaded Trained Network!'
# def setup_backend(backend='TkAgg'):
#     import sys
#     del sys.modules['matplotlib.backends']
#     del sys.modules['matplotlib.pyplot']
#     import matplotlib as mpl
#     mpl.use(backend)  # do this before importing pyplot
#     import matplotlib.pyplot as plt
#     return plt
#                  
#                                          
# def animate():
#     number_of_behaviors=6  
#     videoClient = camProxy.subscribe("python_client7", resolution, colorSpace, 5)
#     naoImage = camProxy.getImageRemote(videoClient)
#     camProxy.setParam(vision_definitions.kCameraBrightnessID, 40)
#     camProxy.setParam(vision_definitions.kCameraAutoWhiteBalanceID, 0)
#     camProxy.setParam(vision_definitions.kCameraAutoExpositionID, 0)
# #     camProxy.unsubscribe(videoClient)
#                
#     # Get the image size and pixel array.
#     imageWidth = naoImage[0]
#     imageHeight = naoImage[1]
#     array = naoImage[6]
#                                
#     im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
#     imr=im.resize((20,20),Image.ANTIALIAS)
#     imgray = imr.convert('L')
#                                
#     imgrayArray = np.array(imgray)
#     fig3=plt.figure(3) # input image
#     fig3.canvas.set_window_title('Figure 3: Input Image') 
#     inputimage = plt.imshow(imgrayArray,interpolation='Nearest',animated=True,label="blah", cmap=plt.get_cmap('gray'))
#                       
#     imgrayArray = imgrayArray.flatten()
#     imgrayArray = np.reshape(imgrayArray, (1,400))
#                 
#                              
#                 
# ##########################################
#     compressedImageFeature = e.network.feed_forward(imgrayArray)[3]
#     x = LSTMClassificationNet.activate(compressedImageFeature)
#     reconstructedImage = e.network.predict(imgrayArray)
#     reconstructedImage = np.reshape(reconstructedImage, (20,20))
#           
#     fig4=plt.figure(4) 
#     fig4.canvas.set_window_title('Figure 4: Reconstructed Image') 
#     reconstructedimage = plt.imshow(reconstructedImage,interpolation='Nearest',animated=True,label="blah", cmap=plt.get_cmap('gray'))
#                                          
#     fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
#     rects = plt.bar(range(number_of_behaviors), x, align='center')
#     fig.canvas.set_window_title('Figure 1: Classification Histogram') 
#     plt.ylim([0,1])
#     plt.xticks(np.arange(number_of_behaviors) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
#     plt.ylabel('Probability')
#     plt.xlabel('Behavior')
#                                        
#     outputBallLiftList = [0] * 500 
#     outputBallRollList = [0] * 500
#     outputBellRingLList = [0] * 500
#     outputBellRingRList = [0] * 500 
#     outputBallRollPlateList = [0] * 500
#     outputRopewayList = [0] * 500
#          
#     fig2=plt.figure(2, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
#     fig2.canvas.set_window_title('Figure 2: Classification Time Step') 
#     plt.xlabel('Time Step')
#     plt.ylabel('Probability')
#     ax1=plt.axes()
#     ax1.axis([0,500,-1,1.5])    
#     ax1.yaxis.tick_right()
#     ax1.yaxis.set_label_position("right")
#     ax1.grid(True)
#     lineOut1, = plt.plot(outputBallLiftList, label='Ball Lift', linewidth=3)
#     lineOut2, = plt.plot(outputBallRollList, label='Ball Roll', linewidth=3)
#     lineOut3, = plt.plot(outputBellRingLList, label='Bell Ring L', linewidth=3)
#     lineOut4, = plt.plot(outputBellRingRList, label='Bell Ring R', linewidth=3)
#     lineOut5, = plt.plot(outputBallRollPlateList, label='Ball Roll Plate', linewidth=3)
#     lineOut6, = plt.plot(outputRopewayList, label='Ropeway', linewidth=3)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.15, 1.), fancybox=True, shadow=True, ncol=1)
#       
#     try:
#         
#         iteration=0
#         while True:
#             time_start= time.time()
#             # Get the image size and pixel array.
#             im = Image.fromstring("RGB", (imageWidth, imageHeight), camProxy.getImageRemote(videoClient)[6])
#             imr=im.resize((20,20),Image.ANTIALIAS)
#             imgrayArray = np.array(imr.convert('L'))
#                          
#             inputimage.set_data(imgrayArray)
#             fig3.canvas.draw()
#                 
#             imgrayArray = np.reshape((imgrayArray.flatten()/255.0).astype(np.float32), (1,400)) 
#                              
#             # Activate all nets
#             reconstructedImage = e.network.predict(imgrayArray)
#             behaviorPrediction = LSTMClassificationNet.activate(e.network.feed_forward(imgrayArray)[3])
#                          
# #             Draw the figs
#             reconstructedimage.set_data(np.reshape(reconstructedImage, (20,20)))
#             fig4.canvas.draw()
#             
#             for rect, h in zip(rects, behaviorPrediction):
#                 rect.set_height(h)
#             fig.canvas.draw()
#             
#             fig2=plt.figure(2)
#             BallLiftNeuron, BallRollNeuron, BellRingLNeuron, BellRingRNeuron, BallRollPlateNeuron, RopewayNeuron = behaviorPrediction
#             outputBallLiftList.append(BallLiftNeuron)
#             outputBallRollList.append(BallRollNeuron)
#             outputBellRingLList.append(BellRingLNeuron)
#             outputBellRingRList.append(BellRingRNeuron)
#             outputBallRollPlateList.append(BallRollPlateNeuron)
#             outputRopewayList.append(RopewayNeuron)
#             del outputBallLiftList[0]
#             del outputBallRollList[0]
#             del outputBellRingLList[0]
#             del outputBellRingRList[0]
#             del outputBallRollPlateList[0]
#             del outputRopewayList[0]
#             lineOut1.set_ydata(outputBallLiftList)  
#             lineOut2.set_ydata(outputBallRollList)
#             lineOut3.set_ydata(outputBellRingLList)
#             lineOut4.set_ydata(outputBellRingRList)
#             lineOut5.set_ydata(outputBallRollPlateList)
#             lineOut6.set_ydata(outputRopewayList)
#             
#             iteration+=1
#                          
#             print iteration, "FPS:",1/(time.time()-time_start)
#               
#     except KeyboardInterrupt:
#         print "Unsubcribed"
#         camProxy.unsubscribe(videoClient)
#         sys.exit("Ended")
#                                          
# 
# plt = setup_backend()
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# win = fig.canvas.manager.window
# win.after(10, animate)
# plt.ion()
# plt.show()

####################
# Manual OFFLINE Test
####################        
# LSTMClassificationNet = NetworkReader.readFrom('20LSTMCell(1)/TrainUntilConv.xml')
# print 'Loaded Trained Network!'
# from random import randint
#  
#   
# for i in range(200):
#     x = LSTMClassificationNet.activate(LBallLift[i])
# for i in range(300):
#     x = LSTMClassificationNet.activate(BallLiftJoint[i])
# print argmax(x)



# 
# std_deviation = 1
# mean = 0
#  
# BallLiftJoint = BallLiftJoint + np.random.normal(mean,std_deviation,(10000,10))
# BallRollJoint = BallRollJoint + np.random.normal(mean,std_deviation,(10000,10))
# BellRingLJoint = BellRingLJoint + np.random.normal(mean,std_deviation,(10000,10))
# BellRingRJoint = BellRingRJoint + np.random.normal(mean,std_deviation,(10000,10))
# BallRollPlateJoint = BallRollPlateJoint + np.random.normal(mean,std_deviation,(10000,10))
# RopewayJoint = RopewayJoint + np.random.normal(mean,std_deviation,(10000,10))
# 
# predictedBLLabels = []
# predictedBRLabels = []
# predictedBRLLabels = []
# predictedBRRLabels = []
# predictedBRPLabels = []
# predictedRWLabels = []
#  
#  
# offset = 200
# for j in range(10):
#     start = randint(7999,9799)
#     end = start + offset
#     timestep=range(start,end)
#     LSTMClassificationNet.reset()
#     for i in timestep:
#         x = LSTMClassificationNet.activate(BallLiftJoint[i])
#     predictedBLLabels.append(argmax(x))
# 
# for j in range(10):
#     start = randint(7999,9799)
#     end = start + offset
#     timestep=range(start,end)
#     LSTMClassificationNet.reset()
#     for i in timestep:
#         x = LSTMClassificationNet.activate(BallRollJoint[i])
#     predictedBRLabels.append(argmax(x))
#     
# for j in range(10):
#     start = randint(7999,9799)
#     end = start + offset
#     timestep=range(start,end)
#     LSTMClassificationNet.reset()
#     for i in timestep:
#         x = LSTMClassificationNet.activate(BellRingLJoint[i])
#     predictedBRLLabels.append(argmax(x))
#     
# for j in range(10):
#     start = randint(7999,9799)
#     end = start + offset
#     timestep=range(start,end)
#     LSTMClassificationNet.reset()
#     for i in timestep:
#         x = LSTMClassificationNet.activate(BellRingRJoint[i])
#     predictedBRRLabels.append(argmax(x))
#     
# for j in range(10):
#     start = randint(7999,9799)
#     end = start + offset
#     timestep=range(start,end)
#     LSTMClassificationNet.reset()
#     for i in timestep:
#         x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
#     predictedBRPLabels.append(argmax(x))
#     
# for j in range(10):
#     start = randint(7999,9799)
#     end = start + offset
#     timestep=range(start,end)
#     LSTMClassificationNet.reset()
#     for i in timestep:
#         x = LSTMClassificationNet.activate(RopewayJoint[i])
#     predictedRWLabels.append(argmax(x))
#     
# print predictedBLLabels
# print predictedBRLabels
# print predictedBRLLabels
# print predictedBRRLabels
# print predictedBRPLabels
# print predictedRWLabels
# 
# BLAcc = 100-percentError(predictedBLLabels, [0,0,0,0,0,0,0,0,0,0])
# BRAcc = 100-percentError(predictedBRLabels, [1,1,1,1,1,1,1,1,1,1])
# BRLAcc = 100-percentError(predictedBRLLabels, [2,2,2,2,2,2,2,2,2,2])
# BRRAcc = 100-percentError(predictedBRRLabels, [3,3,3,3,3,3,3,3,3,3])
# BRPAcc = 100-percentError(predictedBRPLabels, [4,4,4,4,4,4,4,4,4,4])
# RWAcc = 100-percentError(predictedRWLabels, [5,5,5,5,5,5,5,5,5,5])
# 
# overallAcc = (BLAcc + BRAcc + BRLAcc + BRRAcc + BRPAcc + RWAcc) /6
# print "Ball Lift Accuracy:", BLAcc,"%"
# print "Ball Roll Accuracy:", BRAcc,"%"
# print "Bell Ring Left Accuracy:", BRLAcc,"%"
# print "Bell Ring Right Accuracy:", BRRAcc,"%"
# print "Ball Roll Plate Accuracy:", BRPAcc,"%"
# print "Ropeway Accuracy:", RWAcc,"%"
# print "Overall Accuracy:",overallAcc, "%"
    
    



    
# for i in timestep:
#     x = LSTMClassificationNet.activate(BallLiftJoint[i])
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# rects = plt.bar(range(6), x, align='center')
# for rect, h in zip(rects, x):
#     rect.set_height(h)
# fig.canvas.draw()
# plt.tight_layout(pad=2.5)
# plt.ylim([0,1])
# plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
# plt.ylabel('Posterior Probability')
# plt.xlabel('Behavior')
# plt.show()
#    
# # LSTMClassificationNet.reset()   
# for i in timestep:
#     x = LSTMClassificationNet.activate(BallRollJoint[i])
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# rects = plt.bar(range(6), x, align='center')
# for rect, h in zip(rects, x):
#     rect.set_height(h)
# fig.canvas.draw()
# plt.tight_layout(pad=2.5)
# plt.ylim([0,1])
# plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
# plt.ylabel('Posterior Probability')
# plt.xlabel('Behavior')
# plt.show()
# #   
# for i in timestep:
#     x = LSTMClassificationNet.activate(BellRingLJoint[i])
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# rects = plt.bar(range(6), x, align='center')
# for rect, h in zip(rects, x):
#     rect.set_height(h)
# fig.canvas.draw()
# plt.tight_layout(pad=2.5)
# plt.ylim([0,1])
# plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
# plt.ylabel('Posterior Probability')
# plt.xlabel('Behavior')
# plt.show()
#   
# for i in timestep:
#     x = LSTMClassificationNet.activate(BellRingRJoint[i])
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# rects = plt.bar(range(6), x, align='center')
# for rect, h in zip(rects, x):
#     rect.set_height(h)
# fig.canvas.draw()
# plt.tight_layout(pad=2.5)
# plt.ylim([0,1])
# plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
# plt.ylabel('Posterior Probability')
# plt.xlabel('Behavior')
# plt.show()
#   
# for i in timestep:
#     x = LSTMClassificationNet.activate(BallRollPlateJoint[i])
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# rects = plt.bar(range(6), x, align='center')
# for rect, h in zip(rects, x):
#     rect.set_height(h)
# fig.canvas.draw()
# plt.tight_layout(pad=2.5)
# plt.ylim([0,1])
# plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
# plt.ylabel('Posterior Probability')
# plt.xlabel('Behavior')
# plt.show()
#   
# for i in timestep:
#     x = LSTMClassificationNet.activate(RopewayJoint[i])
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# rects = plt.bar(range(6), x, align='center')
# for rect, h in zip(rects, x):
#     rect.set_height(h)
# fig.canvas.draw()
# plt.tight_layout(pad=2.5)
# plt.ylim([0,1])
# plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
# plt.ylabel('Posterior Probability')
# plt.xlabel('Behavior')
# plt.show()
 



# #####################
# # Online Live Test# 6 Behaviors
# #####################     
# LSTMClassificationNet = NetworkReader.readFrom(networkPath)
# e.load("../../../autoencoder/OverallBehavior-deep-AE.gz")
# print 'Loaded Trained Network!'
# def setup_backend(backend='TkAgg'):
#     import sys
#     del sys.modules['matplotlib.backends']
#     del sys.modules['matplotlib.pyplot']
#     import matplotlib as mpl
#     mpl.use(backend)  # do this before importing pyplot
#     import matplotlib.pyplot as plt
#     return plt
#                  
#                                          
# def animate():
#     number_of_behaviors=6  
#     videoClient = camProxy.subscribe("python_client", resolution, colorSpace, 5)
#     naoImage = camProxy.getImageRemote(videoClient)
#     camProxy.setParam(vision_definitions.kCameraBrightnessID, 40)
#     camProxy.setParam(vision_definitions.kCameraAutoWhiteBalanceID, 0)
#     camProxy.setParam(vision_definitions.kCameraAutoExpositionID, 0)
# #     camProxy.unsubscribe(videoClient)
#                
#     # Get the image size and pixel array.
#     imageWidth = naoImage[0]
#     imageHeight = naoImage[1]
#     array = naoImage[6]
#                                
#     im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
#     imr=im.resize((20,20),Image.ANTIALIAS)
#     imgray = imr.convert('L')
#                                
#     imgrayArray = np.array(imgray)
#     fig3=plt.figure(3) # input image
#     fig3.canvas.set_window_title('Figure 3: Input Image') 
#     inputimage = plt.imshow(imgrayArray,interpolation='Nearest',animated=True,label="blah", cmap=plt.get_cmap('gray'))
#                       
#     imgrayArray = imgrayArray.flatten()
#     imgrayArray = np.reshape(imgrayArray, (1,400))
#                 
#                              
#                 
# ##########################################
#     compressedImageFeature = e.network.feed_forward(imgrayArray)[3]
#     x = LSTMClassificationNet.activate(compressedImageFeature)
#     reconstructedImage = e.network.predict(imgrayArray)
#     reconstructedImage = np.reshape(reconstructedImage, (20,20))
#           
#     fig4=plt.figure(4) 
#     fig4.canvas.set_window_title('Figure 4: Reconstructed Image') 
#     reconstructedimage = plt.imshow(reconstructedImage,interpolation='Nearest',animated=True,label="blah", cmap=plt.get_cmap('gray'))
#                                          
#     fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
#     rects = plt.bar(range(number_of_behaviors), x, align='center')
#     fig.canvas.set_window_title('Figure 1: Classification Histogram') 
#     plt.ylim([0,1])
#     plt.xticks(np.arange(number_of_behaviors) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
#     plt.ylabel('Probability')
#     plt.xlabel('Behavior')
#                                        
#     outputBallLiftList = [0] * 500 
#     outputBallRollList = [0] * 500
#     outputBellRingLList = [0] * 500
#     outputBellRingRList = [0] * 500 
#     outputBallRollPlateList = [0] * 500
#     outputRopewayList = [0] * 500
#          
#     fig2=plt.figure(2, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
#     fig2.canvas.set_window_title('Figure 2: Classification Time Step') 
#     plt.xlabel('Time Step')
#     plt.ylabel('Probability')
#     ax1=plt.axes()
#     ax1.axis([0,500,-1,1.5])    
#     ax1.yaxis.tick_right()
#     ax1.yaxis.set_label_position("right")
#     ax1.grid(True)
#     lineOut1, = plt.plot(outputBallLiftList, label='Ball Lift', linewidth=3)
#     lineOut2, = plt.plot(outputBallRollList, label='Ball Roll', linewidth=3)
#     lineOut3, = plt.plot(outputBellRingLList, label='Bell Ring L', linewidth=3)
#     lineOut4, = plt.plot(outputBellRingRList, label='Bell Ring R', linewidth=3)
#     lineOut5, = plt.plot(outputBallRollPlateList, label='Ball Roll Plate', linewidth=3)
#     lineOut6, = plt.plot(outputRopewayList, label='Ropeway', linewidth=3)
#     plt.legend(loc='upper center', bbox_to_anchor=(0.15, 1.), fancybox=True, shadow=True, ncol=1)
#       
#     try:
#         while True:
#             time_start= time.time()
#             # Load Image Subscription
#             naoImage = camProxy.getImageRemote(videoClient)
#             # Get the image size and pixel array.
#             im = Image.fromstring("RGB", (imageWidth, imageHeight), naoImage[6])
#             imr=im.resize((20,20),Image.ANTIALIAS)
#             imgrayArray = np.array(imr.convert('L'))
#                          
#             fig3=plt.figure(3) # input image
#             inputimage.set_data(imgrayArray)
#                   
# #             imgrayArray = imgrayArray.flatten() 
# #             imgrayArray = (imgrayArray/255.0).astype(np.float32)
# #             imgrayArray = np.reshape(imgrayArray, (1,400)) 
#             imgrayArray = np.reshape((imgrayArray.flatten()/255.0).astype(np.float32), (1,400)) 
#                              
#             # Activate all nets
#             compressedImageFeature = e.network.feed_forward(imgrayArray)[3]
#             reconstructedImage = e.network.predict(imgrayArray)
#             x = LSTMClassificationNet.activate(compressedImageFeature)
#                          
#              # Draw the figs
#             fig4=plt.figure(4)
#             reconstructedImage = np.reshape(reconstructedImage, (20,20))
#             reconstructedimage.set_data(reconstructedImage)
#                               
#             fig=plt.figure(1)
#             for rect, h in zip(rects, x):
#                 rect.set_height(h)
#             fig.canvas.draw()
#                               
#             fig2=plt.figure(2)
#             BallLiftNeuron, BallRollNeuron, BellRingLNeuron, BellRingRNeuron, BallRollPlateNeuron, RopewayNeuron = x
#             outputBallLiftList.append(BallLiftNeuron)
#             outputBallRollList.append(BallRollNeuron)
#             outputBellRingLList.append(BellRingLNeuron)
#             outputBellRingRList.append(BellRingRNeuron)
#             outputBallRollPlateList.append(BallRollPlateNeuron)
#             outputRopewayList.append(RopewayNeuron)
#             del outputBallLiftList[0]
#             del outputBallRollList[0]
#             del outputBellRingLList[0]
#             del outputBellRingRList[0]
#             del outputBallRollPlateList[0]
#             del outputRopewayList[0]
#             lineOut1.set_ydata(outputBallLiftList)  
#             lineOut2.set_ydata(outputBallRollList)
#             lineOut3.set_ydata(outputBellRingLList)
#             lineOut4.set_ydata(outputBellRingRList)
#             lineOut5.set_ydata(outputBallRollPlateList)
#             lineOut6.set_ydata(outputRopewayList)
#                          
#             print "FPS:",1/(time.time()-time_start)
#               
#     except KeyboardInterrupt:
#         print "Unsubcribed"
#         camProxy.unsubscribe(videoClient)
#         sys.exit("Ended")
#                                          
#       
# plt = setup_backend()
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# win = fig.canvas.manager.window
# win.after(10, animate)
# plt.ion()
# plt.show()



################################################
# LSTMClassificationNet = NetworkReader.readFrom(networkPath)
# e.load("../../../autoencoder/OverallBehavior-deep-AE.gz")
# plt.figure(1)
# recons = e.network.predict(BallLift[9:10,:])
# recons = np.reshape(recons, (20,20))
# plt.imshow(recons, cmap=plt.get_cmap('gray'))
# plt.figure(2)
# recons = e.network.predict(BellRingL[90:91,:])
# recons = np.reshape(recons, (20,20))
# plt.imshow(recons, cmap=plt.get_cmap('gray'))
# plt.figure(3)
# recons = e.network.predict(BellRingR[900:901,:])
# recons = np.reshape(recons, (20,20))
# plt.imshow(recons, cmap=plt.get_cmap('gray'))
# plt.figure(4)
# recons = e.network.predict(BellRingL[9999:10000,:])
# recons = np.reshape(recons, (20,20))
# plt.imshow(recons, cmap=plt.get_cmap('gray'))
# plt.show()



 
 



#####################
# Online Live Test#
#####################     
# def setup_backend(backend='TkAgg'):
#     import sys
#     del sys.modules['matplotlib.backends']
#     del sys.modules['matplotlib.pyplot']
#     import matplotlib as mpl
#     mpl.use(backend)  # do this before importing pyplot
#     import matplotlib.pyplot as plt
#     return plt
#        
#                                
# def animate():
#     number_of_behaviors=6  
#     videoClient = camProxy.subscribe("python_client1", resolution, colorSpace, 5)
#     naoImage = camProxy.getImageRemote(videoClient)
#     camProxy.setParam(vision_definitions.kCameraBrightnessID, 40)
#     camProxy.setParam(vision_definitions.kCameraAutoWhiteBalanceID, 0)
#     camProxy.setParam(vision_definitions.kCameraAutoExpositionID, 0)
#     camProxy.unsubscribe(videoClient)
#      
#     # Get the image size and pixel array.
#     imageWidth = naoImage[0]
#     imageHeight = naoImage[1]
#     array = naoImage[6]
#                      
#     im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
#     imr=im.resize((20,20),Image.ANTIALIAS)
#     imgray = imr.convert('L')
#                      
#     imgrayArray = np.array(imgray)
#             
#     imgrayArray = imgrayArray.flatten()
#     imgrayArray = np.reshape(imgrayArray, (1,400))
#       
#                    
#       
# ##########################################
#     compressedImageFeature = e.network.feed_forward(imgrayArray)[3]
#     x = LSTMClassificationNet.activate(compressedImageFeature)
#                                
#     rects = plt.bar(range(number_of_behaviors), x, align='center')
#                              
#     fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
#     plt.ylim([0,1])
#     plt.xticks(np.arange(6) , ('Ball Lift', 'Ball Roll', 'Bell Ring L', 'Bell Ring R', 'Ball Roll Plate', 'Ropeway'))
#     plt.ylabel('Probability')
#     plt.xlabel('Behavior')
#                              
#     outputBallLiftList = [0] * 500 
#     outputBallRollList = [0] * 500 
#     outputBellRingLList = [0] * 500
#     outputBellRingRList = [0] * 500 
#     outputBallRollPlateList = [0] * 500
#     outputRopewayList = [0] * 500
#                              
#     fig2=plt.figure(2, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
#        
#     plt.xlabel('Time Step')
#     plt.ylabel('Probability')
#     ax1=plt.axes()
#     ax1.axis([0,500,-1,1.5])    
#     ax1.yaxis.tick_right()
#     ax1.yaxis.set_label_position("right")
#     ax1.grid(True)
#     lineOut1, = plt.plot(outputBallLiftList, label='Ball Lift', linewidth=3)
#     lineOut2, = plt.plot(outputBallRollList, label='Ball Roll', linewidth=3)
#     lineOut3, = plt.plot(outputBellRingLList, label='Bell Ring L', linewidth=3)
#     lineOut4, = plt.plot(outputBellRingRList, label='Bell Ring R', linewidth=3)
#     lineOut5, = plt.plot(outputBallRollPlateList, label='Ball Roll Plate', linewidth=3)
#     lineOut6, = plt.plot(outputRopewayList, label='Ropeway', linewidth=3)     
#     plt.legend(loc='upper center', bbox_to_anchor=(0.15, 1.), fancybox=True, shadow=True, ncol=1)
#        
#     while True:
#         time_start= time.time()
#         # Load Image Subscription
#         videoClient = camProxy.subscribe("python_client1", resolution, colorSpace, 5)
#         naoImage = camProxy.getImageRemote(videoClient)
#         camProxy.unsubscribe(videoClient)
#         # Get the image size and pixel array.
#         imageWidth = naoImage[0]
#         imageHeight = naoImage[1]
#         array = naoImage[6]
#                         
#         im = Image.fromstring("RGB", (imageWidth, imageHeight), array)
#         imr=im.resize((20,20),Image.ANTIALIAS)
#         imgray = imr.convert('L')
#         imgrayArray = np.array(imgray)
#            
#         fig3=plt.figure(3) # input image
#         plt.clf()
#         plt.imshow(imgrayArray, cmap=plt.cm.gray)    
#                
#         imgrayArray = imgrayArray.flatten()  
#         imgrayArray = np.reshape(imgrayArray, (1,400)) 
#                
#         # Activate all nets
#         compressedImageFeature = e.network.feed_forward(imgrayArray)[3]
#         reconstructedImage = e.network.predict(imgrayArray)
#         x = LSTMClassificationNet.activate(compressedImageFeature)
#            
#         # Draw the figs
#         fig4=plt.figure(4)
#         plt.clf()
#         reconstructedImage = np.reshape(reconstructedImage, (20,20))
#         plt.imshow(reconstructedImage, cmap=plt.cm.gray)
#                 
#         fig=plt.figure(1)
#         for rect, h in zip(rects, x):
#             rect.set_height(h)
#         fig.canvas.draw()
#                 
#         fig2=plt.figure(2)
#         BallLiftNeuron, BallRollNeuron, BellRingLNeuron, BellRingRNeuron, BallRollPlateNeuron, RopewayNeuron = x
#         outputBallLiftList.append(BallLiftNeuron)
#         outputBallRollList.append(BallRollNeuron)
#         outputBellRingLList.append(BellRingLNeuron)
#         outputBellRingRList.append(BellRingRNeuron)
#         outputBallRollPlateList.append(BallRollPlateNeuron)
#         outputRopewayList.append(RopewayNeuron)
#         del outputBallLiftList[0]
#         del outputBallRollList[0]
#         del outputBellRingLList[0]
#         del outputBellRingRList[0]
#         del outputBallRollPlateList[0]
#         del outputRopewayList[0]
#         lineOut1.set_ydata(outputBallLiftList)  
#         lineOut2.set_ydata(outputBallRollList)  
#         lineOut3.set_ydata(outputBellRingLList)
#         lineOut4.set_ydata(outputBellRingRList)
#         lineOut5.set_ydata(outputBallRollPlateList)
#         lineOut6.set_ydata(outputRopewayList)
#         plt.draw()
#            
#         time_end= time.time()
#         print time_end-time_start
#                                
# plt.ioff()    
# plt = setup_backend()
# fig=plt.figure(1, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
# plt.ion()
# win = fig.canvas.manager.window
# win.after(10, animate)
# plt.show()