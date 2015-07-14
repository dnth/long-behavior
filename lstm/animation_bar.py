from matplotlib import animation
from naoqi import ALProxy
import matplotlib.pyplot as plt
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import numpy as np
import time

robotIP="192.168.0.108"
tts=ALProxy("ALTextToSpeech", robotIP, 9559)
motion = ALProxy("ALMotion", robotIP, 9559)
memory = ALProxy("ALMemory", robotIP, 9559)
posture = ALProxy("ALRobotPosture", robotIP, 9559)
camProxy = ALProxy("ALVideoDevice", robotIP, 9559)
resolution = 0    # kQQVGA
colorSpace = 11   # RGB

###########################################
# For long behavior with 8 classes - BLIT #
###########################################
# First set up the figure, the axis, and the plot element we want to animate
fig1 = plt.figure(2, figsize=(10, 5), dpi=90, facecolor='w', edgecolor='k')

LSTMClassificationNet = NetworkReader.readFrom("20LSTMCell(1)/TrainUntilConv.xml")
print 'Loaded Trained Network!'
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

# plot once and set initial height to zero
rects = plt.bar(np.arange(8), LSTMNet_output, align='center')
plt.xticks(np.arange(8) , ('LBL', 'RBL', 'LBR', 'RBR', 'LBRP', 'RBRP','LRW','RRW'))
plt.xlim([-0.5,8])
plt.grid(True)
for rect, h in zip(rects, LSTMNet_output):
        rect.set_height(0)
        
def init():
    global rects 
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
    
    return rects

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
    
    rects = plt.bar(range(8), LSTMNet_output, align='center')
    for rect, h in zip(rects, LSTMNet_output):
        rect.set_height(h)
    
    print "FPS:",1/(time.time()-time_start)
    return rects
    
anim = animation.FuncAnimation(fig1, animate, init_func=init,interval=1, blit=True)
plt.show()