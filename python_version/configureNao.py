## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2017
# description: basic class for all Nao tasks.
## ---------------------------------------------------------------------

import sys
sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
from naoqi import ALProxy

class ConfigureNao(object):
	"""
	a basic class for all nao tasks, including motion, bisualization etc.
	"""
	def __init__(self, IP):
		self._IP = IP
		self._PORT = 9559
		self._cameraProxy = ALProxy("ALVideoDevice", self._IP, self._PORT)
		self._motionProxy = ALProxy("ALMotion", self._IP, self._PORT)
		self._postureProxy = ALProxy("ALRobotPosture", self._IP, self._PORT)
		self._tts = ALProxy("ALTextToSpeech",self._IP, self._PORT)
		self._memoryProxy = ALProxy("ALMemory", self._IP, self._PORT)
	
