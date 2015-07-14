import numpy as np
import matplotlib.pyplot as plt
 
BallLiftJoint = np.loadtxt('../20fpsFullBehaviorSampling/BallLift/JointData.txt').astype(np.float32)
BallRollJoint = np.loadtxt('../20fpsFullBehaviorSampling/BallRoll/JointData.txt').astype(np.float32)
BellRingLJoint = np.loadtxt('../20fpsFullBehaviorSampling/BellRingL/JointData.txt').astype(np.float32)
BellRingRJoint = np.loadtxt('../20fpsFullBehaviorSampling/BellRingR/JointData.txt').astype(np.float32)
BallRollPlateJoint = np.loadtxt('../20fpsFullBehaviorSampling/BallRollPlate/JointData.txt').astype(np.float32)
RopewayJoint = np.loadtxt('../20fpsFullBehaviorSampling/Ropeway/JointData.txt').astype(np.float32)
 
LBallLift = np.vstack((BellRingLJoint[0:100], BallLiftJoint[0:500]))
RBallLift = np.vstack((BellRingRJoint[0:100], BallLiftJoint[0:500]))
LBallRoll = np.vstack((BellRingLJoint[0:100], BallRollJoint[0:500]))
RBallRoll = np.vstack((BellRingRJoint[0:100], BallRollJoint[0:500]))
LBallRollPlate = np.vstack((BellRingLJoint[0:100], BallRollPlateJoint[0:500]))
RBallRollPlate = np.vstack((BellRingRJoint[0:100], BallRollPlateJoint[0:500]))
LRopeway = np.vstack((BellRingLJoint[0:100], RopewayJoint[0:500]))
RRopeway = np.vstack((BellRingRJoint[0:100], RopewayJoint[0:500]))
 
# for i in range(6000,10000,300):
#     LBallLift = np.vstack((LBallLift, BellRingLJoint[i:i+100], BallLiftJoint[i:i+400]))
# for i in range(6000,10000,300):
#     RBallLift = np.vstack((RBallLift, BellRingRJoint[i:i+100], BallLiftJoint[i:i+200]))
#       
# for i in range(6000,10000,300):
#     LBallRoll = np.vstack((LBallRoll, BellRingLJoint[i:i+100], BallRollJoint[i:i+200]))
# for i in range(6000,10000,300):
#     RBallRoll = np.vstack((RBallRoll, BellRingRJoint[i:i+100], BallRollJoint[i:i+200]))
#   
# for i in range(6000,10000,300):
#     LBallRollPlate = np.vstack((LBallRollPlate, BellRingLJoint[i:i+100], BallRollPlateJoint[i:i+200]))
# for i in range(6000,10000,300):
#     RBallRollPlate = np.vstack((RBallRollPlate, BellRingRJoint[i:i+100], BallRollPlateJoint[i:i+200]))
#       
# for i in range(6000,10000,300):
#     LRopeway = np.vstack((LRopeway, BellRingLJoint[i:i+100], RopewayJoint[i:i+200]))
# for i in range(6000,10000,300):
#     RRopeway = np.vstack((RRopeway, BellRingRJoint[i:i+100], RopewayJoint[i:i+200]))
     

plt.plot(LRopeway[:500], linewidth=2)
plt.grid(True)
plt.show()


# LBallLift = np.vstack((BellRingLJoint[0:100], BallLiftJoint[0:100]))
# RBallLift = np.vstack((BellRingRJoint[0:100], BallLiftJoint[0:100]))
# LBallRoll = np.vstack((BellRingLJoint[0:100], BallRollJoint[0:100]))
# RBallRoll = np.vstack((BellRingRJoint[0:100], BallRollJoint[0:100]))
# LBallRollPlate = np.vstack((BellRingLJoint[0:100], BallRollPlateJoint[0:100]))
# RBallRollPlate = np.vstack((BellRingRJoint[0:100], BallRollPlateJoint[0:100]))
# LRopeway = np.vstack((BellRingLJoint[0:100], RopewayJoint[0:100]))
# RRopeway = np.vstack((BellRingRJoint[0:100], RopewayJoint[0:100]))
# 
# for i in range(200,10000,200):
#     LBallLift = np.vstack((LBallLift, BellRingLJoint[i:i+100], BallLiftJoint[i:i+100]))
# for i in range(200,10000,200):
#     RBallLift = np.vstack((RBallLift, BellRingRJoint[i:i+100], BallLiftJoint[i:i+100]))
#     
# for i in range(200,10000,200):
#     LBallRoll = np.vstack((LBallRoll, BellRingLJoint[i:i+100], BallRollJoint[i:i+100]))
# for i in range(200,10000,200):
#     RBallRoll = np.vstack((RBallRoll, BellRingRJoint[i:i+100], BallRollJoint[i:i+100]))
# 
# for i in range(200,10000,200):
#     LBallRollPlate = np.vstack((LBallRollPlate, BellRingLJoint[i:i+100], BallRollPlateJoint[i:i+100]))
# for i in range(200,10000,200):
#     RBallRollPlate = np.vstack((RBallRollPlate, BellRingRJoint[i:i+100], BallRollPlateJoint[i:i+100]))
#     
# for i in range(200,10000,200):
#     LRopeway = np.vstack((LRopeway, BellRingLJoint[i:i+100], RopewayJoint[i:i+100]))
# for i in range(200,10000,200):
#     RRopeway = np.vstack((RRopeway, BellRingRJoint[i:i+100], RopewayJoint[i:i+100]))
    
    
