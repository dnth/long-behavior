import numpy as np
import theanets
import climate
import matplotlib.pyplot as plt
logging = climate.get_logger('lstm-joint')
climate.enable_default_logging()
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
    
    
fig= plt.figure(1)
plt.subplot(121)   
plt.plot(LBallLift[:200,:]) # plot the first 200 timesteps

LBallLift = LBallLift.reshape((100,200,10))

plt.subplot(122)
plt.plot(LBallLift[1,:200,:]) # plot the first 200 timestep
plt.show()