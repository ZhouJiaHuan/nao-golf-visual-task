#coding: utf-8
## ---------------------------------------------------------------------
# author: Meringue
# date: 1/15/2018
# description: visual classes for Nao golf task.
## ---------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
#sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")
#os.chdir(os.getcwd())
import cv2
import numpy as np

import vision_definitions as vd
import time

from configureNao import ConfigureNao
from naoqi import ALProxy
import motion


class VisualBasis(ConfigureNao):
	"""
	a basic class for visual task.
	"""
	def __init__(self, IP, PORT=9559, cameraId=1, resolution=vd.kVGA):
		"""
		initilization. 
		Args:
			IP: NAO's IP
			cameraId: bottom camera (1,default) or top camera (0).
			resolution: kVGA, default: 640*480)
		Return: none
		"""     
		super(VisualBasis, self).__init__(IP, PORT)
		self._cameraId = cameraId
		self._cameraName = "CameraBottom" if self._cameraId==1 else "CameraTop"
		self._resolution = resolution
		self._colorSpace = vd.kBGRColorSpace
		self._fps = 20
		self._frameHeight = 0
		self._frameWidth = 0
		self._frameChannels = 0
		self._frameArray = None
		self._cameraPitchRange = 47.64/180*np.pi
		self._cameraYawRange = 60.97/180*np.pi
		self._cameraProxy.setActiveCamera(self._cameraId)
							 
	def updateFrame(self, client="python_client"):
		"""
		get a new image from the specified camera and save it in self._frame.
		Args:
			client: client name.
		Return: none.
		"""
		if self._cameraProxy.getActiveCamera() == self._cameraId:
			print("current camera has been actived.")
		else:
			self._cameraProxy.setActiveCamera(self._cameraId)
		self._videoClient = self._cameraProxy.subscribe(client, self._resolution, self._colorSpace, self._fps)
		frame = self._cameraProxy.getImageRemote(self._videoClient)
		self._cameraProxy.unsubscribe(self._videoClient)
		try:
			self._frameWidth = frame[0]
			self._frameHeight = frame[1]
			self._frameChannels = frame[2]
			self._frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1],frame[0],frame[2]])
		except IndexError:
			raise
		
	def getFrameArray(self):
		"""
		get current frame.
		Return: 
			current frame array (numpy array).
		"""
		if self._frameArray is None:
			return np.array([])
		return self._frameArray
				
	def showFrame(self):
		"""
		show current frame image.
		"""
		if self._frameArray is None:
			print("please get an image from Nao with the method updateFrame()")
		else:
			cv2.imshow("current frame", self._frameArray)
			
	def printFrameData(self):
		"""
		print current frame data.
		"""
		print("frame height = ", self._frameHeight)
		print("frame width = ", self._frameWidth)
		print("frame channels = ", self._frameChannels)
		print("frame shape = ", self._frameArray.shape)
		
	def saveFrame(self, framePath):
		"""
		save current frame to specified direction.		
		Arguments:
			framePath: image path.
		"""
		cv2.imwrite(framePath, self._frameArray)
		print("current frame image has been saved in", framePath)
			  
	def setParam(self, paramName=None, paramValue = None):
		raise NotImplementedError
	
	def setAllParamsToDefault(self):
		raise NotImplementedError
		

class BallDetect(VisualBasis):
	"""
	derived from VisualBasics, used to detect the ball.
	"""
	def __init__(self, IP, PORT=9559, cameraId=vd.kBottomCamera, resolution=vd.kVGA):
		"""
		initialization.
		"""
		super(BallDetect, self).__init__(IP, PORT, cameraId, resolution)
		self._ballData = {"centerX":0, "centerY":0, "radius":0}
		self._ballPosition= {"disX":0, "disY":0, "angle":0}
		self._ballRadius = 0.05

	def __getChannelAndBlur(self, color):
		"""
		get the specified channel and blur the result.
		Arguments:
			color: the color channel to split, only supports the color of red, geen and blue.   
		Return: 
			the specified color channel or None (when the color is not supported).
		"""
		try:
			channelB = self._frameArray[:,:,0]
			channelG = self._frameArray[:,:,1]
			channelR = self._frameArray[:,:,2]
		except:
			raise Exception("no image detected!")
		Hm = 6
		if color == "red":
			channelB = channelB*0.1*Hm
			channelG = channelG*0.1*Hm
			channelR = channelR - channelB - channelG
			channelR = 3*channelR
			channelR = cv2.GaussianBlur(channelR, (9,9), 1.5)
			channelR[channelR<0] = 0
			channelR[channelR>255] = 255
			return np.uint8(np.round(channelR))
		elif color == "blue":
			channelR = channelR*0.1*Hm
			channelG = channelG*0.1*Hm
			channelB = channelB - channelG - channelR
			channelB = 3*channelB            
			channelB = cv2.GaussianBlur(channelB, (9,9), 1.5)
			channelB[channelB<0] = 0
			channelB[channelB>255] = 255
			return np.uint8(np.round(channelB))
		elif color == "green":
			channelB = channelB*0.1*Hm
			channelR= channelR*0.1*Hm
			channelG = channelG - channelB - channelR
			channelG = 3*channelG
			channelG = cv2.GaussianBlur(channelG, (9,9), 1.5)
			channelG[channelG<0] = 0
			channelG[channelG>255] = 255
			return np.uint8(np.round(channelG))
		else:
			print("can not recognize the color!")
			print("supported color:red, green and blue.")
			return None

	def __binImageHSV(self, color):
		"""
		get binary image from the HSV image (transformed from BGR image)
		Args:
			color: the color for binarization.
		Return:
			binImage: binary image.
		"""
		try:
			frameArray = self._frameArray.copy()
			imgHSV = cv2.cvtColor(frameArray, cv2.COLOR_BGR2HSV)
		except:
			raise Exception("no image detected!")
		if color == "red":
			minHSV1=np.array([0,43,46])
			maxHSV1=np.array([10,255,255])
			minHSV2=np.array([156,43,46])
			maxHSV2=np.array([180,255,255])
			frameBin1 = cv2.inRange(imgHSV, minHSV1, maxHSV1)
			frameBin2 = cv2.inRange(imgHSV, minHSV2, maxHSV2)
			frameBin = np.maximum(frameBin1, frameBin2)
			frameBin = cv2.GaussianBlur(frameBin, (9,9), 1.5)
			cv2.imshow("bin image", frameBin)
			return frameBin
		else:
			raise Exception("not recognize the color!")            

	def __findCircles(self, img, minDist, minRadius, maxRadius):
		"""
		detect circles from an image.
		Arguments:
			img: image to be detected.
			minDist: minimum distance between the centers of the detected circles.
			minRadius: minimum circle radius.
			maxRadius: maximum circle radius.
		Return: 
			an uint16 numpy array shaped circleNum*3 if circleNum>0, ([[circleX, circleY,radius]])
			else return None.
		"""
		cv_version = cv2.__version__.split(".")[0]
		if cv_version == "3": # for OpenCV >= 3.0.0
			gradient_name = cv2.HOUGH_GRADIENT
		else:
			import cv2.cv as cv
			gradient_name = cv.CV_HOUGH_GRADIENT
		circles = cv2.HoughCircles(np.uint8(img), gradient_name, 1, minDist, 
								   param1=150, param2=15, minRadius=minRadius, maxRadius=maxRadius)
		if circles is None:
			return np.uint16([])
		else:
			return np.uint16(np.around(circles[0, ]))
	
	def __selectCircle(self, circles):
		"""
		select one circle in list type from all circles detected. 
		Args:
			circles: numpy array shaped (N, 3),　N is the number of circles.
		Return:
			selected circle or None (no circle is selected).
		"""
		if circles.shape[0] == 0:
			return circles
		if circles.shape[0] == 1:
			centerX = circles[0][0]
			centerY = circles[0][1]
			radius = circles[0][2]
			initX = centerX - 2*radius
			initY = centerY - 2*radius
			if initX<0 or initY<0 or (initX+4*radius)>self._frameWidth or (initY+4*radius)>self._frameHeight or radius<1:
				return circles	
		channelB = self._frameArray[:,:,0]
		channelG = self._frameArray[:,:,1]
		channelR = self._frameArray[:,:,2]
		rRatioMin = 1.0; circleSelected = np.uint16([])
		for circle in circles:
			centerX = circle[0]
			centerY = circle[1]
			radius = circle[2]
			initX = centerX - 2*radius
			initY = centerY - 2*radius
			if initX<0 or initY<0 or (initX+4*radius)>self._frameWidth or (initY+4*radius)>self._frameHeight or radius<1:
				continue	
			rectBallArea = self._frameArray[initY:initY+4*radius+1, initX:initX+4*radius+1,:]
			bFlat = np.float16(rectBallArea[:,:,0].flatten())
			gFlat = np.float16(rectBallArea[:,:,1].flatten())
			rFlat = np.float16(rectBallArea[:,:,2].flatten())
			rScore1 = np.uint8(rFlat>1.0*gFlat)
			rScore2 = np.uint8(rFlat>1.0*bFlat)
			rScore = float(np.sum(rScore1*rScore2))
			gScore = float(np.sum(np.uint8(gFlat>1.0*rFlat)))
			rRatio = rScore/len(rFlat)
			gRatio = gScore/len(gFlat) 
			# print("red ratio = ", rRatio)
			# print("green ratio = ", gRatio)
			if rRatio>=0.12 and gRatio>=0.1 and abs(rRatio-0.19)<abs(rRatioMin-0.19):
				circleSelected = circle
				rRatioMin = rRatio		
		return circleSelected
	
	def __updateBallPositionFitting(self, standState):
		"""
		compute and update the ball position with compensation.
		Args:
			standState: "standInit" or "standUp".
		"""
		bottomCameraDirection = {"standInit":49.2, "standUp":39.7} 
		ballRadius = self._ballRadius
		try:
			cameraDirection = bottomCameraDirection[standState]
		except KeyError:
			print("Error! unknown standState, please check the value of stand state!")
			raise
		else:
			if self._ballData["radius"] == 0:
				self._ballPosition= {"disX":0, "disY":0, "angle":0}
			else:
				centerX = self._ballData["centerX"]
				centerY = self._ballData["centerY"]
				radius = self._ballData["radius"]
				cameraPosition = self._motionProxy.getPosition("CameraBottom", 2, True)
				cameraX = cameraPosition[0]
				cameraY = cameraPosition[1]
				cameraHeight = cameraPosition[2]
				headPitches = self._motionProxy.getAngles("HeadPitch", True)
				headPitch = headPitches[0]
				headYaws = self._motionProxy.getAngles("HeadYaw", True)
				headYaw = headYaws[0]
				ballPitch = (centerY-240.0)*self._cameraPitchRange/480.0   # y (pitch angle)
				ballYaw = (320.0-centerX)*self._cameraYawRange/640.0    # x (yaw angle)
				dPitch = (cameraHeight-ballRadius)/np.tan(cameraDirection/180*np.pi+headPitch+ballPitch)
				dYaw = dPitch/np.cos(ballYaw)
				ballX = dYaw*np.cos(ballYaw+headYaw)+cameraX
				ballY = dYaw*np.sin(ballYaw+headYaw)+cameraY
				ballYaw = np.arctan2(ballY, ballX)
				self._ballPosition["disX"] = ballX                                
				if (standState == "standInit"):
					ky = 42.513*ballX**4 - 109.66*ballX**3 + 104.2*ballX**2 - 44.218*ballX + 8.5526               
					#ky = 12.604*ballX**4 - 37.962*ballX**3 + 43.163*ballX**2 - 22.688*ballX + 6.0526
					ballY = ky*ballY
					ballYaw = np.arctan2(ballY,ballX)                    
				self._ballPosition["disY"] = ballY
				self._ballPosition["angle"] = ballYaw

	def __updateBallPosition(self, standState):
		"""
		compute and update the ball position with the ball data in frame.
		standState: "standInit" or "standUp".
		"""
		
		bottomCameraDirection = {"standInit":49.2/180*np.pi, "standUp":39.7/180*np.pi} 
		try:
			cameraDirection = bottomCameraDirection[standState]
		except KeyError:
			print("Error! unknown standState, please check the value of stand state!")
			raise
		else:
			if self._ballData["radius"] == 0:
				self._ballPosition= {"disX":0, "disY":0, "angle":0}
			else:
				centerX = self._ballData["centerX"]
				centerY = self._ballData["centerY"]
				radius = self._ballData["radius"]
				cameraPos = self._motionProxy.getPosition(self._cameraName, motion.FRAME_WORLD, True)
				cameraX, cameraY, cameraHeight = cameraPos[:3]
				head_yaw, head_pitch = self._motionProxy.getAngles("Head", True)
				camera_pitch = head_pitch + cameraDirection
				img_center_x = self._frameWidth/2
				img_center_y = self._frameHeight/2
				center_x = self._ballData["centerX"]
				center_y = self._ballData["centerY"]
				img_pitch = (center_y-img_center_y)/(self._frameHeight)*self._cameraPitchRange
				img_yaw = (img_center_x-center_x)/(self._frameWidth)*self._cameraYawRange
				ball_pitch = camera_pitch + img_pitch
				ball_yaw = img_yaw + head_yaw
				print("ball yaw = ", ball_yaw/np.pi*180)
				dis_x = (cameraHeight-self._ballRadius)/np.tan(ball_pitch) + np.sqrt(cameraX**2+cameraY**2)
				dis_y = dis_x*np.sin(ball_yaw)
				dis_x = dis_x*np.cos(ball_yaw)
				self._ballPosition["disX"] = dis_x
				self._ballPosition["disY"] = dis_y
				self._ballPosition["angle"] = ball_yaw
			
							   
	def updateBallData(self, client="python_client", standState="standInit", color="red", color_space="BGR", fitting=False):
		"""
		update the ball data with the frame get from the bottom camera.
		Arguments:
			standState: ("standInit", default), "standInit" or "standUp".
			color: ("red", default) the color of ball to be detected.
			color_space: "BGR", "HSV".
			fittting: the method of localization.
		Return: a dict with ball data. for example: {"centerX":0, "centerY":0, "radius":0}.
		"""
		self.updateFrame(client)
		#cv2.imwrite("src_image.jpg", self._frameArray)
		minDist = int(self._frameHeight/30.0)
		minRadius = 1
		maxRadius = int(self._frameHeight/10.0)
		if color_space == "BGR":
			grayFrame = self.__getChannelAndBlur(color)
		else:
			grayFrame = self.__binImageHSV(color)
		#cv2.imshow("bin frame", grayFrame)
		#cv2.imwrite("bin_frame.jpg", grayFrame)
		#cv2.waitKey(20)
		circles = self.__findCircles(grayFrame, minDist, minRadius, maxRadius)
		circle = self.__selectCircle(circles)
		# print("circle = ", circle.shape)
		if circle.shape[0] == 0:
			print("no ball")
			self._ballData = {"centerX":0, "centerY":0, "radius":0}
			self._ballPosition= {"disX":0, "disY":0, "angle":0}
		else:
			circle = circle.reshape([-1,3])    
			self._ballData = {"centerX":circle[0][0], "centerY":circle[0][1], "radius":circle[0][2]}
			if fitting == True:
				self.__updateBallPositionFitting(standState=standState)
			else:
				self.__updateBallPosition(standState=standState)
		  	
	def getBallPostion(self):
		"""
		get ball position.
		
		Return: distance in x axis, distance in y axis and direction related to Nao.
		"""
		return [self._ballPosition["disX"], self._ballPosition["disY"], self._ballPosition["angle"]]
		
	def showBallPosition(self):        
		"""
		show ball data in the current frame.
		"""
		if self._ballData["radius"] == 0:
			print("no ball found.")
			cv2.imshow("ball position", self._frameArray)
		else:
			print("ball postion = ", (self._ballPosition["disX"], self._ballPosition["disY"]))
			print("ball direction = ", self._ballPosition["angle"])
			frameArray = self._frameArray
			cv2.circle(frameArray, (self._ballData["centerX"],self._ballData["centerY"]),
					   self._ballData["radius"], (250,150,150),2)
			cv2.circle(frameArray, (self._ballData["centerX"],self._ballData["centerY"]),
					   2, (50,250,50), 3)
			cv2.imshow("ball position", frameArray)
			#cv2.imwrite("ball_position.jpg", frameArray)


class StickDetect(VisualBasis):
	"""
	derived from VisualBasics, used to detect the stict.
	"""
	
	def __init__(self, IP, PORT=9559, cameraId=vd.kTopCamera, resolution=vd.kVGA):
		super(StickDetect, self).__init__(IP, PORT, cameraId, resolution)
		self._boundRect = []
		self._cropKeep = 1
		self._stickAngle = None # rad
		
	def __preprocess(self, minHSV, maxHSV, cropKeep, morphology):
		"""
		preprocess the current frame for stick detection.(binalization, crop etc.)
		Arguments:
			minHSV: the lower limit for binalization.
			maxHSV: the upper limit for binalization.
			cropKeep: crop ratio (>=0.5).
			morphology: erosion and dilation.
		Return:
			preprocessed image for stick detection.
		"""
		self._cropKeep = cropKeep
		frameArray = self._frameArray
		height = self._frameHeight
		width = self._frameWidth
		try:
			frameArray = frameArray[int((1-cropKeep)*height):,:]
		except IndexError:
			raise		
		frameHSV = cv2.cvtColor(frameArray, cv2.COLOR_BGR2HSV)
		frameBin = cv2.inRange(frameHSV, minHSV, maxHSV)
		kernelErosion = np.ones((5,5), np.uint8)
		kernelDilation = np.ones((5,5), np.uint8) 
		frameBin = cv2.erode(frameBin, kernelErosion, iterations=1)
		frameBin = cv2.dilate(frameBin, kernelDilation, iterations=1)
		frameBin = cv2.GaussianBlur(frameBin, (9,9), 0)
		#cv2.imshow("stick bin", frameBin)
		#cv2.waitKey(20)
		return frameBin
		
	def __findStick(self, frameBin, minPerimeter, minArea):
		"""
		find the yellow stick in the preprocessed frame.
		Args:
			frameBin: preprocessed frame.
			minPerimeter: minimum perimeter of detected stick.
			minArea: minimum area of detected stick.
		Return: detected stick marked with rectangle or [].
		"""
		rects = []
		if cv2.__version__.split(".")[0] == "3": # for OpenCV >= 3.0.0
			_, contours, _ = cv2.findContours(frameBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		else:
			contours, _ = cv2.findContours(frameBin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
		if len(contours) == 0:
			return rects
		for contour in contours:
			perimeter = cv2.arcLength(contour, True)
			area = cv2.contourArea(contour)
			if perimeter>minPerimeter and area>minArea:
				x,y,w,h = cv2.boundingRect(contour)
				rects.append([x,y,w,h])	
		if len(rects) == 0:
			return rects		
		rects = [rect for rect in rects if (1.0*rect[3]/rect[2])>0.8]
		if len(rects) == 0:
			return rects
		rects = np.array(rects)
		# print(rects)
		rect = rects[np.argmax(1.0*(rects[:,-1])/rects[:,-2]),]
		rect[1] += int(self._frameHeight *(1-self._cropKeep))
		return rect
		
	def updateStickData(self, client="test", minHSV=np.array([27,55,115]), maxHSV=np.array([45,255,255]), cropKeep=1, morphology=True):
		"""
		update the yellow stick data from the specified camera.
		Args:
			client: client name
			minHSV: the lower limit for binalization.
			maxHSV: the upper limit for binalization.
			cropKeep:  crop ratio (>=0.5).
			morphology: (True, default), erosion and dilation.
		"""
		self.updateFrame(client)
		minPerimeter = self._frameHeight/8.0
		minArea = self._frameHeight*self._frameWidth/1000.0
		frameBin = self.__preprocess(minHSV, maxHSV, cropKeep, morphology)
		rect = self.__findStick(frameBin, minPerimeter, minArea)
		if rect == []:
			self._boundRect = []
			self._stickAngle = None
		else:
			self._boundRect = rect
			centerX = rect[0]+rect[2]/2
			width = self._frameWidth *1.0
			self._stickAngle = (width/2-centerX)/width*self._cameraYawRange
			cameraPosition = self._motionProxy.getPosition("Head", 2, True)
			cameraY = cameraPosition[5]
			# print("cameraY:",cameraY * 180 / 3.14)
			self._stickAngle += cameraY
				
	def showStickPosition(self):
		"""
		show the stick  position in the current frame.
		"""
		if self._boundRect == []:
			print("no stick detected.")
			cv2.imshow("stick position", self._frameArray)
		else:
			[x,y,w,h] = self._boundRect
			frame = self._frameArray.copy()
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
			cv2.imshow("stick position", frame)
