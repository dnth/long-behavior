import numpy as np
import theanets
import climate
import matplotlib.pyplot as plt
logging = climate.get_logger('lstm-joint')
climate.enable_default_logging()

def plotLearningCurve():
    fig=plt.figure(0, figsize=(10,8) )
    fig.clf()
    plt.plot(trn_loss, label='Training Set Error', linestyle="--", linewidth=2)
    plt.plot(valid_loss, label='Validation Set Error', linewidth=2)
    plt.title('Cross-Entropy Error')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.savefig("error")
    
# Original Joint data
BallLiftJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BallLift/JointData.txt').astype(np.float32)
BallRollJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BallRoll/JointData.txt').astype(np.float32)
BellRingLJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BellRingL/JointData.txt').astype(np.float32)
BellRingRJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BellRingR/JointData.txt').astype(np.float32)
BallRollPlateJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/BallRollPlate/JointData.txt').astype(np.float32)
RopewayJoint = np.loadtxt('../../../20fpsFullBehaviorSampling/Ropeway/JointData.txt').astype(np.float32)

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

# print LBallLift.shape
# print RBallLift.shape
# print LBallRoll.shape
# print RBallRoll.shape
# print LBallRollPlate.shape
# print RBallRollPlate.shape
# print LRopeway.shape
# print RRopeway.shape


# plt.plot(LBallLift[:200,:])
# plt.show()

LBallLift = LBallLift.reshape((200,100,10))
RBallLift = RBallLift.reshape((200,100,10))
LBallRoll = LBallRoll.reshape((200,100,10))
RBallRoll = RBallRoll.reshape((200,100,10))
LBallRollPlate = LBallRollPlate.reshape((200,100,10))
RBallRollPlate = RBallRollPlate.reshape((200,100,10))
LRopeway = LRopeway.reshape((200,100,10))
RRopeway = RRopeway.reshape((200,100,10))

# plt.plot(LBallLift[:,98:99,:])
# plt.show()

train_series_LBL = LBallLift[:,:60,:]
train_series_RBL = RBallLift[:,:60,:]
train_series_LBR = LBallRoll[:,:60,:]
train_series_RBR = RBallRoll[:,:60,:]
train_series_LBRP = LBallRollPlate[:,:60,:]
train_series_RBRP = RBallRollPlate[:,:60,:]
train_series_LRW = LRopeway[:,:60,:]
train_series_RRW = RRopeway[:,:60,:]

valid_series_LBL = LBallLift[:,60:,:]
valid_series_RBL = RBallLift[:,60:,:]
valid_series_LBR = LBallRoll[:,60:,:]
valid_series_RBR = RBallRoll[:,60:,:]
valid_series_LBRP = LBallRollPlate[:,60:,:]
valid_series_RBRP = RBallRollPlate[:,60:,:]
valid_series_LRW = LRopeway[:,60:,:]
valid_series_RRW = RRopeway[:,60:,:]


train_class_LBL = np.zeros((200,60)).astype(np.int8)
train_class_LBL.fill(0)
train_class_RBL = np.zeros((200,60)).astype(np.int8)
train_class_RBL.fill(1)
train_class_LBR = np.zeros((200,60)).astype(np.int8)
train_class_LBR.fill(2)
train_class_RBR = np.zeros((200,60)).astype(np.int8)
train_class_RBR.fill(3)
train_class_LBRP = np.zeros((200,60)).astype(np.int8)
train_class_LBRP.fill(4)
train_class_RBRP = np.zeros((200,60)).astype(np.int8)
train_class_RBRP.fill(5)
train_class_LRW = np.zeros((200,60)).astype(np.int8)
train_class_LRW.fill(6)
train_class_RRW = np.zeros((200,60)).astype(np.int8)
train_class_RRW.fill(7)

valid_class_LBL = np.zeros((200,40)).astype(np.int8)
valid_class_LBL.fill(0)
valid_class_RBL = np.zeros((200,40)).astype(np.int8)
valid_class_RBL.fill(1)
valid_class_LBR = np.zeros((200,40)).astype(np.int8)
valid_class_LBR.fill(2)
valid_class_RBR = np.zeros((200,40)).astype(np.int8)
valid_class_RBR.fill(3)
valid_class_LBRP = np.zeros((200,40)).astype(np.int8)
valid_class_LBRP.fill(4)
valid_class_RBRP = np.zeros((200,40)).astype(np.int8)
valid_class_RBRP.fill(5)
valid_class_LRW = np.zeros((200,40)).astype(np.int8)
valid_class_LRW.fill(6)
valid_class_RRW = np.zeros((200,40)).astype(np.int8)
valid_class_RRW.fill(7)

train_series = np.hstack((train_series_LBL, train_series_RBL, train_series_LBR, train_series_RBR, train_series_LBRP, train_series_RBRP, train_series_LRW, train_series_RRW))
valid_series = np.hstack((valid_series_LBL, valid_series_RBL, valid_series_LBR, valid_series_RBR, valid_series_LBRP, valid_series_RBRP, valid_series_LRW, valid_series_RRW))

train_class = np.hstack((train_class_LBL, train_class_RBL, train_class_LBR, train_class_RBR, train_class_LBRP, train_class_RBRP, train_class_LRW, train_class_RRW))
valid_class = np.hstack((valid_class_LBL, valid_class_RBL, valid_class_LBR, valid_class_RBR, valid_class_LBRP, valid_class_RBRP, valid_class_LRW, valid_class_RRW))



print train_series.shape
print train_class.shape

print valid_series.shape
print valid_class.shape

def layer(n):
    return dict(form='bidirectional', worker='lstm', size=n)

def layer(n):
    return dict(worker='lstm', size=n)
  
e = theanets.Experiment(
    theanets.recurrent.Classifier,
    layers=(10, layer(20), 8),
    recurrent_error_start=0,
    batch_size=20
)
 
  
# trn_loss=[]
# valid_loss=[]
# i=0
# for train, valid in e.itertrain([train_series, train_class], [valid_series, valid_class], optimize='rmsprop', validate_every=5, patience=50,
#         max_gradient_norm=10,):
#     print('training loss:', train['loss'])
#     print('most recent validation loss:', valid['loss'])
#     trn_loss.append(train['loss'])
#     valid_loss.append(valid['loss'])
#                                        
# np.savetxt('trn_loss.txt', trn_loss)
# np.savetxt('valid_loss.txt', valid_loss)
# plotLearningCurve()
# e.save("testbidi-joint-classifier.gz") 

e.load("testbidi-joint-classifier.gz")
  
  
testdata = RRopeway.reshape((20000,10))[:200]
# plt.plot(testdata)
# plt.show()
testdata = testdata.reshape((200,1,10))
  
print e.network.classify(testdata)
  
# print e.network.classify(LBallLift[:200,85:86,:])


  
  
  
# print e.network.predict(LBallLift[:200,85:86,:])
# outputclass = e.network.classify(LBallLift[:200,85:86,:])
# outputclass = e.network.classify(RBallLift[100:200,85:86,:])
# print outputclass


# classzerocount = 0
# classonecount = 0
# for numoutput in outputclass:
#     if outputclass[numoutput] ==0:
#         classzerocount+=1
#     elif outputclass[numoutput] ==1:
#         classonecount+=1
# final = np.array([classzerocount, classonecount])
# print np.argmax(final)
