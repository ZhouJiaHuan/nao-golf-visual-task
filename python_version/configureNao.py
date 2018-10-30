"""
basic class for all Nao tasks.
@author: Meringue
@date: 2018/1/15
"""

import sys
#sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
from naoqi import ALProxy

class ConfigureNao(object):
	"""
	a basic class for all nao tasks, including motion, bisualization etc.
	"""
	def __init__(self, IP, PORT=9559):
		self.IP = IP
		self.PORT = PORT
		try:
			self.cameraProxy = ALProxy("ALVideoDevice", self.IP, self.PORT)
			self.motionProxy = ALProxy("ALMotion", self.IP, self.PORT)
			self.postureProxy = ALProxy("ALRobotPosture", self.IP, self.PORT)
			self.tts = ALProxy("ALTextToSpeech",self.IP, self.PORT)
			self.memoryProxy = ALProxy("ALMemory", self.IP, self.PORT)
			self.landMarkProxy = ALProxy("ALLandMarkDetection", self.IP, self.PORT)
		except Exception, e:
			print("Error when configuring the NAO!")
			print(str(e))
			exit(1)
	
