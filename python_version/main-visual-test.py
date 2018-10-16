## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2018
# description: some test codes for Nao golf visual part.
## ---------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import time
import os
import sys
#sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")

from visualTask import *
from naoqi import ALProxy
import vision_definitions as vd

#IP = "192.168.1.100"
IP = "169.254.67.213"

visualBasis = VisualBasis(IP,cameraId=0, resolution=vd.kVGA)
ballDetect = BallDetect(IP, resolution=vd.kVGA)
stickDetect = StickDetect(IP, cameraId=0, resolution=vd.kVGA)

##test code
"""
visualBasis.updateFrame()
visualBasis.showFrame()
visualBasis.printFrameData()
"""



while 1:
	time1 = time.time()
	ballDetect.updateBallData(client="xx")
	time2 = time.time()
	print("update data time = ", time2-time1)
	ballDetect.showBallPosition()
	cv2.waitKey(10)


"""
while 1:
	stickDetect.updateStickData(client="xxx")
	stickDetect.showStickPosition(showTime = 10)
"""


"""
print "start collecting..."
for i in range(10):
	imgName = "stick_" + str(i+127) + ".jpg"
	imgDir = os.path.join("stick_images", imgName)
	visualBasis.updateFrame()
	visualBasis.showFrame(timeMs=1000)
	visualBasis.saveFrame(imgDir)
	print "saved in ", imgDir
	time.sleep(5)
"""

"""
visualBasis._tts.say("hello world")
"""

"""
visualBasis._motionProxy.wakeUp()
"""

"""
dataList = visualBasis._memoryProxy.getDataList("camera")
print dataList
"""

"""
visualBasis._motionProxy.setStiffnesses("Body", 1.0)
visualBasis._motionProxy.moveInit()
"""

#motionProxy = ALProxy("ALMotion", IP, 9559)
#postureProxy = ALProxy("ALRobotPosture", IP, 9559)

#motionProxy.wakeUp()
#postureProxy.goToPosture("StandInit", 0.5)


#motionProxy.wakeUp()
#motionProxy.goToPosture("StandInit", 0.5)
#motionProxy.moveToward(0.1, 0.1, 0, [["Frequency", 1.0]])
#motionProxy.moveTo(0.3, 0.2, 0)
"""
"""
