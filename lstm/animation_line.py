from matplotlib import animation
from naoqi import ALProxy
import matplotlib.pyplot as plt
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import numpy as np
import time

robotIP="192.168.0.101"
tts=ALProxy("ALTextToSpeech", robotIP, 9559)
motion = ALProxy("ALMotion", robotIP, 9559)
memory = ALProxy("ALMemory", robotIP, 9559)
posture = ALProxy("ALRobotPosture", robotIP, 9559)
camProxy = ALProxy("ALVideoDevice", robotIP, 9559)
resolution = 0    # kQQVGA
colorSpace = 11   # RGB

#####################################
# For long behavior with 8 classes - BLIT#
#####################################
# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(1,figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
ax = plt.axes(xlim=(0,800), ylim=(-0.5, 1.5))

fig1 = plt.figure(2, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')
plt.xticks(np.arange(8) , ('LBL', 'RBL', 'LBR', 'RBR', 'LBRP', 'RBRP','LRW','RRW'))


lineLBL, = ax.plot([], [], lw=2, label='LBL')
lineRBL, = ax.plot([], [], lw=2, label='RBL')
lineLBR, = ax.plot([], [], lw=2, label='LBR')
lineRBR, = ax.plot([], [], lw=2, label='RBR')
lineLBRP, = ax.plot([], [], lw=2, label='LBRP')
lineRBRP, = ax.plot([], [], lw=2, label='RBRP')
lineLRW, = ax.plot([], [], lw=2, label='LRW')
lineRRW, = ax.plot([], [], lw=2, label='RRW')

ax.grid(True)
ax.legend(loc='upper center', bbox_to_anchor=(0.11, 1.), fancybox=True, shadow=True, ncol=1)

LBL_list = [0] * 800
RBL_list = [0] * 800
LBR_list = [0] * 800
RBR_list = [0] * 800
LBRP_list = [0] * 800
RBRP_list = [0] * 800
LRW_list = [0] * 800
RRW_list = [0] * 800

LSTMClassificationNet = NetworkReader.readFrom("20LSTMCell(1)/TrainUntilConv.xml")
print 'Loaded Trained Network!'

def init():
    lineLBL.set_data([], [])
    return lineLBL,lineRBL,lineLBR,lineRBR,lineLBRP,lineRBRP,lineLRW,lineRRW,

# animation function.  This is called sequentially
def animate(i): 
    time_start = time.time()
    
    RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
    RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
    RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
    RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
    RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")
              
    LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
    LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
    LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
    LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
    LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")
    
           
    LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])
    

#     rects = plt.bar(range(8), LSTMNet_output, align='center')
#     for rect, h in zip(rects, LSTMNet_output):
#         rect.set_height(h)
    
#     fig1.canvas.draw()
    
    LBL_outputNode,RBL_outputNode,LBR_outputNode,RBR_outputNode,LBRP_outputNode,RBRP_outputNode,LRW_outputNode,RRW_outputNode = LSTMNet_output
    
    LBL_list.append(LBL_outputNode)
    RBL_list.append(RBL_outputNode)
    LBR_list.append(LBR_outputNode)
    RBR_list.append(RBR_outputNode)
    LBRP_list.append(LBRP_outputNode)
    RBRP_list.append(RBRP_outputNode)
    LRW_list.append(LRW_outputNode)
    RRW_list.append(RRW_outputNode)
    
    del LBL_list[0]
    del RBL_list[0]
    del LBR_list[0]
    del RBR_list[0]
    del LBRP_list[0]
    del RBRP_list[0]
    del LRW_list[0]
    del RRW_list[0]
    
    lineLBL.set_data(np.arange(0,800,1),LBL_list)
    lineRBL.set_data(np.arange(0,800,1),RBL_list)
    lineLBR.set_data(np.arange(0,800,1),LBR_list)
    lineRBR.set_data(np.arange(0,800,1),RBR_list)
    lineLBRP.set_data(np.arange(0,800,1),LBRP_list)
    lineRBRP.set_data(np.arange(0,800,1),RBRP_list)
    lineLRW.set_data(np.arange(0,800,1),LRW_list)
    lineRRW.set_data(np.arange(0,800,1),RRW_list)
    
    print "FPS:",1/(time.time()-time_start)
    return lineLBL,lineRBL,lineLBR,lineRBR,lineLBRP,lineRBRP,lineLRW,lineRRW,

def initBar():
    RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
    RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
    RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
    RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
    RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")
              
    LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
    LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
    LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
    LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
    LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")
    
           
    LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])
    

    rects = plt.bar(range(8), LSTMNet_output, align='center')
    for rect, h in zip(rects, LSTMNet_output):
        rect.set_height(h)
#     return rects,

def animateBar():
    RShoulderPitchTestData = memory.getData("Device/SubDeviceList/RShoulderPitch/Position/Sensor/Value")
    RShoulderRollTestData = memory.getData("Device/SubDeviceList/RShoulderRoll/Position/Sensor/Value")
    RElbowRollTestData = memory.getData("Device/SubDeviceList/RElbowRoll/Position/Sensor/Value")
    RElbowYawTestData = memory.getData("Device/SubDeviceList/RElbowYaw/Position/Sensor/Value")
    RWristYawTestData = memory.getData("Device/SubDeviceList/RWristYaw/Position/Sensor/Value")
              
    LShoulderPitchTestData = memory.getData("Device/SubDeviceList/LShoulderPitch/Position/Sensor/Value")
    LShoulderRollTestData = memory.getData("Device/SubDeviceList/LShoulderRoll/Position/Sensor/Value")
    LElbowRollTestData = memory.getData("Device/SubDeviceList/LElbowRoll/Position/Sensor/Value")
    LElbowYawTestData = memory.getData("Device/SubDeviceList/LElbowYaw/Position/Sensor/Value")
    LWristYawTestData = memory.getData("Device/SubDeviceList/LWristYaw/Position/Sensor/Value")
    
           
    LSTMNet_output = LSTMClassificationNet.activate([RShoulderPitchTestData, RShoulderRollTestData, RElbowRollTestData, RElbowYawTestData, RWristYawTestData, 
                                               LShoulderPitchTestData, LShoulderRollTestData, LElbowRollTestData, LElbowYawTestData, LWristYawTestData])
    

    rects = plt.bar(range(8), LSTMNet_output, align='center')
    for rect, h in zip(rects, LSTMNet_output):
        rect.set_height(h)
#     return rects,

anim = animation.FuncAnimation(fig, animate, init_func=init,interval=1, blit=True)
# anim2 = animation.FuncAnimation(fig1, animateBar, init_func=initBar,interval=1, blit=True)
plt.show()