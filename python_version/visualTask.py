#coding: utf-8
"""
visual classes for Nao golf task.
@author: Meringue
@date: 2018/1/15
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
#sys.path.append("/home/meringue/Softwares/pynaoqi-sdk/") #naoqi directory
sys.path.append("./")
import numpy as np
import vision_definitions as vd
import time
from configureNao import ConfigureNao
from naoqi import ALProxy
import motion
import math
import almath
import cv2

cv_version = cv2.__version__.split(".")[0]
if cv_version == "2": # for OpenCV 2
	import cv2.cv as cv

class VisualBasis(ConfigureNao):
	"""
	a basic class for visual task.
	"""
	def __init__(self, IP, PORT=9559, cameraId=vd.kBottomCamera, resolution=vd.kVGA):
		"""
		initilization. 

		Args:
			IP: NAO's IP
			cameraId: bottom camera (1,default) or top camera (0).
			resolution: kVGA, default: 640*480)
		Return: 
			none
		"""     
		super(VisualBasis, self).__init__(IP, PORT)
		self.cameraId = cameraId
		self.cameraName = "CameraBottom" if self.cameraId==vd.kBottomCamera else "CameraTop"
		self.resolution = resolution
		self.colorSpace = vd.kBGRColorSpace
		self.fps = 20
		self.frameHeight = 0
		self.frameWidth = 0
		self.frameChannels = 0
		self.frameArray = None
		self.cameraPitchRange = 47.64/180*np.pi
		self.cameraYawRange = 60.97/180*np.pi
		self.cameraProxy.setActiveCamera(self.cameraId)
							 
	def updateFrame(self, client="python_client"):
		"""
		get a new image from the specified camera and save it in self._frame.

		Args:
			client: client name.
		Return: 
			none.
		"""
		if self.cameraProxy.getActiveCamera() != self.cameraId:
			self.cameraProxy.setActiveCamera(self.cameraId)
		videoClient = self.cameraProxy.subscribe(client, self.resolution, self.colorSpace, self.fps)
		frame = self.cameraProxy.getImageRemote(videoClient)
		self.cameraProxy.unsubscribe(videoClient)
		try:
			self.frameWidth = frame[0]
			self.frameHeight = frame[1]
			self.frameChannels = frame[2]
			self.frameArray = np.frombuffer(frame[6], dtype=np.uint8).reshape([frame[1],frame[0],frame[2]])
		except IndexError:
			raise
		
	def getFrameArray(self):
		"""
		get current frame.

		Return: 
			current frame array (numpy array).
		"""
		if self.frameArray is None:
			return np.array([])
		return self.frameArray
				
	def showFrame(self):
		"""
		show current frame image.
		"""
		if self.frameArray is None:
			print("please get an image from Nao with the method updateFrame()")
		else:
			cv2.imshow("current frame", self.frameArray)
			
	def printFrameData(self):
		"""
		print current frame data.
		"""
		print("frame height = ", self.frameHeight)
		print("frame width = ", self.frameWidth)
		print("frame channels = ", self.frameChannels)
		print("frame shape = ", self.frameArray.shape)
		
	def saveFrame(self, framePath):
		"""
		save current frame to specified direction.

		Arguments:
			framePath: image path.
		"""
		cv2.imwrite(framePath, self.frameArray)
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
		self.ballData = {"centerX":0, "centerY":0, "radius":0}
		self.ballPosition= {"disX":0, "disY":0, "angle":0}
		self.ballRadius = 0.025

	def __getChannelAndBlur(self, color):
		"""
		get the specified channel and blur the result.

		Args:
			color: the color channel to split, only supports the color of red, geen and blue.   
		Return: 
			the specified color channel or None (when the color is not supported).
		"""
		try:
			channelB = self.frameArray[:,:,0]
			channelG = self.frameArray[:,:,1]
			channelR = self.frameArray[:,:,2]
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

	def __binImageHSV(self, minHSV1, maxHSV1, minHSV2, maxHSV2):
		"""
		get binary image from the HSV image (transformed from BGR image)

		Args:
			minHSV1, maxHSV1, minHSV2, maxHSV2: parameters [np.array] for red ball detection
		Return:
			binImage: binary image.
		"""
		try:
			frameArray = self.frameArray.copy()
			imgHSV = cv2.cvtColor(frameArray, cv2.COLOR_BGR2HSV)
		except:
			raise Exception("no image detected!")
		frameBin1 = cv2.inRange(imgHSV, minHSV1, maxHSV1)
		frameBin2 = cv2.inRange(imgHSV, minHSV2, maxHSV2)
		frameBin = np.maximum(frameBin1, frameBin2)
		frameBin = cv2.GaussianBlur(frameBin, (9,9), 1.5)
		#cv2.imshow("bin image", frameBin)
		return frameBin        

	def __findCircles(self, img, minDist, minRadius, maxRadius):
		"""
		detect circles from an image.

		Args:
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
			gradient_name = cv.CV_HOUGH_GRADIENT
		circles = cv2.HoughCircles(np.uint8(img), gradient_name, 1, \
								   minDist, param1=150, param2=15, \
								   minRadius=minRadius, maxRadius=maxRadius)
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
			if (initX<0 or initY<0 or (initX+4*radius)>self.frameWidth or \
			   (initY+4*radius)>self.frameHeight or radius<1):
				return circles	
		channelB = self.frameArray[:,:,0]
		channelG = self.frameArray[:,:,1]
		channelR = self.frameArray[:,:,2]
		rRatioMin = 1.0; circleSelected = np.uint16([])
		for circle in circles:
			centerX = circle[0]
			centerY = circle[1]
			radius = circle[2]
			initX = centerX - 2*radius
			initY = centerY - 2*radius
			if initX<0 or initY<0 or (initX+4*radius)>self.frameWidth or \
			   (initY+4*radius)>self.frameHeight or radius<1:
				continue	
			rectBallArea = self.frameArray[initY:initY+4*radius+1, initX:initX+4*radius+1,:]
			bFlat = np.float16(rectBallArea[:,:,0].flatten())
			gFlat = np.float16(rectBallArea[:,:,1].flatten())
			rFlat = np.float16(rectBallArea[:,:,2].flatten())
			rScore1 = np.uint8(rFlat>1.0*gFlat)
			rScore2 = np.uint8(rFlat>1.0*bFlat)
			rScore = float(np.sum(rScore1*rScore2))
			gScore = float(np.sum(np.uint8(gFlat>1.0*rFlat)))
			rRatio = rScore/len(rFlat)
			gRatio = gScore/len(gFlat) 
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
		ballRadius = self.ballRadius
		try:
			cameraDirection = bottomCameraDirection[standState]
		except KeyError:
			print("Error! unknown standState, please check the value of stand state!")
			raise
		else:
			if self.ballData["radius"] == 0:
				self.ballPosition= {"disX":0, "disY":0, "angle":0}
			else:
				centerX = self.ballData["centerX"]
				centerY = self.ballData["centerY"]
				radius = self.ballData["radius"]
				cameraPosition = self.motionProxy.getPosition("CameraBottom", 2, True)
				cameraX = cameraPosition[0]
				cameraY = cameraPosition[1]
				cameraHeight = cameraPosition[2]
				headPitches = self.motionProxy.getAngles("HeadPitch", True)
				headPitch = headPitches[0]
				headYaws = self.motionProxy.getAngles("HeadYaw", True)
				headYaw = headYaws[0]
				ballPitch = (centerY-240.0)*self.cameraPitchRange/480.0   # y (pitch angle)
				ballYaw = (320.0-centerX)*self.cameraYawRange/640.0    # x (yaw angle)
				dPitch = (cameraHeight-ballRadius)/np.tan(cameraDirection/180*np.pi+headPitch+ballPitch)
				dYaw = dPitch/np.cos(ballYaw)
				ballX = dYaw*np.cos(ballYaw+headYaw)+cameraX
				ballY = dYaw*np.sin(ballYaw+headYaw)+cameraY
				ballYaw = np.arctan2(ballY, ballX)
				self.ballPosition["disX"] = ballX                                
				if (standState == "standInit"):
					ky = 42.513*ballX**4 - 109.66*ballX**3 + 104.2*ballX**2 - 44.218*ballX + 8.5526               
					#ky = 12.604*ballX**4 - 37.962*ballX**3 + 43.163*ballX**2 - 22.688*ballX + 6.0526
					ballY = ky*ballY
					ballYaw = np.arctan2(ballY,ballX)                    
				self.ballPosition["disY"] = ballY
				self.ballPosition["angle"] = ballYaw

	def __updateBallPosition(self, standState): # test phase
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
			if self.ballData["radius"] == 0:
				self.ballPosition= {"disX":0, "disY":0, "angle":0}
			else:
				centerX = self.ballData["centerX"]
				centerY = self.ballData["centerY"]
				radius = self.ballData["radius"]
				cameraPos = self.motionProxy.getPosition(self.cameraName, motion.FRAME_WORLD, True)
				cameraX, cameraY, cameraHeight = cameraPos[:3]
				headYaw, headPitch = self.motionProxy.getAngles("Head", True)
				cameraPitch = headPitch + cameraDirection
				imgCenterX = self.frameWidth/2
				imgCenterY = self.frameHeight/2
				centerX = self.ballData["centerX"]
				centerY = self.ballData["centerY"]
				imgPitch = (centerY-imgCenterY)/(self.frameHeight)*self.cameraPitchRange
				imgYaw = (imgCenterX-centerX)/(self.frameWidth)*self.cameraYawRange
				ballPitch = cameraPitch + imgPitch
				#ballPitch = 38/180.0*3.14
				ballYaw = imgYaw + headYaw
				#ballYaw = 31/180.0*3.14
				dist = (cameraHeight-self.ballRadius)/np.tan(ballPitch) + np.sqrt(cameraX**2+cameraY**2)
				#print("height = ", cameraHeight)
				#print("cameraPitch = ", cameraPitch*180/3.14)
				#print("imgYaw = ", imgYaw/3.14*180)
				#print("headYaw = ", headYaw/3.14*180)
				#print("ballYaw = ",ballYaw/3.14*180)
				#print("ballPitch = ", ballPitch/3.14*180)
				disX = dist*np.cos(ballYaw)
				disY = dist*np.sin(ballYaw)
				#print("disX = ", disX)
				#print("disY = ", disY)
				self.ballPosition["disX"] = disX
				self.ballPosition["disY"] = disY
				self.ballPosition["angle"] = ballYaw
								   
	def updateBallData(self, client="python_client", standState="standInit", color="red", 
					   colorSpace="BGR", fitting=False, minHSV1=np.array([0,43,46]), 
					   maxHSV1=np.array([10,255,255]), minHSV2=np.array([156,43,46]), 
					   maxHSV2=np.array([180,255,255]), saveFrameBin=False):
		"""
		update the ball data with the frame get from the bottom camera.

		Args:
			standState: ("standInit", default), "standInit" or "standUp".
			color: ("red", default) the color of ball to be detected.
			colorSpace: "BGR", "HSV".
			fittting: the method of localization.
			minHSV1, maxHSV1, minHSV2, maxHSV2: only for HSV color space.
			saveFrameBin: save the preprocessed frame or not.
		Return: 
			a dict with ball data. for example: {"centerX":0, "centerY":0, "radius":0}.
		"""
		self.updateFrame(client)
		#cv2.imwrite("src_image.jpg", self._frameArray)
		minDist = int(self.frameHeight/30.0)
		minRadius = 1
		maxRadius = int(self.frameHeight/10.0)
		if colorSpace == "BGR":
			grayFrame = self.__getChannelAndBlur(color)
		else:
			grayFrame = self.__binImageHSV(minHSV1, maxHSV1, minHSV2, maxHSV2)
		if saveFrameBin:
			self._frameBin = grayFrame.copy()
		#cv2.imshow("bin frame", grayFrame)
		#cv2.imwrite("bin_frame.jpg", grayFrame)
		#cv2.waitKey(20)
		circles = self.__findCircles(grayFrame, minDist, minRadius, maxRadius)
		circle = self.__selectCircle(circles)
		# print("circle = ", circle.shape)
		if circle.shape[0] == 0:
			#print("no ball")
			self.ballData = {"centerX":0, "centerY":0, "radius":0}
			self.ballPosition= {"disX":0, "disY":0, "angle":0}
		else:
			circle = circle.reshape([-1,3])    
			self.ballData = {"centerX":circle[0][0], "centerY":circle[0][1], "radius":circle[0][2]}
			if fitting == True:
				self.__updateBallPositionFitting(standState=standState)
			else:
				self.__updateBallPosition(standState=standState)
		  	
	def getBallPosition(self):
		"""
		get ball position.

		Return: 
			distance in x axis, distance in y axis and direction related to Nao.
		"""
		disX = self.ballPosition["disX"]
		disY = self.ballPosition["disY"]
		angle = self.ballPosition["angle"]
		return [disX, disY, angle]

	def getBallInfoInImage(self):
		"""
		get ball information in image.

		Return:
			a list of centerX, centerY and radius of the red ball.
		"""
		centerX = self.ballData["centerX"]
		centerY = self.ballData["centerY"]
		radius = self.ballData["radius"]
		return [centerX, centerY, radius]
		
	def showBallPosition(self):        
		"""
		show ball data in the current frame.
		"""
		if self.ballData["radius"] == 0:
			#print("no ball found.")
			print("ball postion = ", (self.ballPosition["disX"], self.ballPosition["disY"]))
			cv2.imshow("ball position", self.frameArray)
		else:
			#print("ballX = ", self.ballData["centerX"])
			#print("ballY = ", self.ballData["centerY"])
			print("ball postion = ", (self.ballPosition["disX"], self.ballPosition["disY"]))
			#print("ball direction = ", self.ballPosition["angle"]*180/3.14)
			frameArray = self.frameArray
			cv2.circle(frameArray, (self.ballData["centerX"],self.ballData["centerY"]),
					   self.ballData["radius"], (250,150,150),2)
			cv2.circle(frameArray, (self.ballData["centerX"],self.ballData["centerY"]),
					   2, (50,250,50), 3)
			cv2.imshow("ball position", frameArray)
			#cv2.imwrite("ball_position.jpg", frameArray)
	
	def sliderHSV(self, client):
		"""
		slider for ball detection in HSV color space.

		Args:
			client: client name.
		"""
		def __nothing():
			pass
		windowName = "slider for ball detection"
		cv2.namedWindow(windowName)
		cv2.createTrackbar("minS1", windowName, 43, 60, __nothing)
		cv2.createTrackbar("minV1", windowName, 46, 65, __nothing)
		cv2.createTrackbar("maxH1", windowName, 10, 20, __nothing)
		cv2.createTrackbar("minH2", windowName, 156, 175, __nothing)
		while 1:
			self.updateFrame(client)
			minS1 = cv2.getTrackbarPos("minS1", windowName)
			minV1 = cv2.getTrackbarPos("minV1", windowName)
			maxH1 = cv2.getTrackbarPos("maxH1", windowName)
			minH2 = cv2.getTrackbarPos("minH2", windowName)
			minHSV1 = np.array([0, minS1, minV1])
			maxHSV1 = np.array([maxH1, 255, 255])
			minHSV2 = np.array([minH2, minS1, minV1])
			maxHSV2=np.array([180,255,255])
			self.updateBallData(client, colorSpace="HSV", minHSV1=minHSV1, 
								maxHSV1=maxHSV1, minHSV2=minHSV2, 
								maxHSV2=maxHSV2, saveFrameBin=True, fitting=True)
			cv2.imshow(windowName, self._frameBin)
			self.showBallPosition()
			k = cv2.waitKey(10) & 0xFF
			if k == 27:
				break
		cv2.destroyAllWindows()


class StickDetect(VisualBasis):
	"""
	derived from VisualBasics, used to detect the stict.
	"""
	
	def __init__(self, IP, PORT=9559, cameraId=vd.kTopCamera, resolution=vd.kVGA):
		super(StickDetect, self).__init__(IP, PORT, cameraId, resolution)
		self.boundRect = []
		self.cropKeep = 1
		self.stickAngle = 0.0 # rad
		
	def __preprocess(self, minHSV, maxHSV, cropKeep, morphology):
		"""
		preprocess the current frame for stick detection.(binalization, crop etc.)

		Args:
			minHSV: the lower limit for binalization.
			maxHSV: the upper limit for binalization.
			cropKeep: crop ratio (>=0.5).
			morphology: erosion and dilation.
		Return:
			preprocessed image for stick detection.
		"""
		self.cropKeep = cropKeep
		frameArray = self.frameArray
		height = self.frameHeight
		width = self.frameWidth
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
		rect = rects[np.argmax(1.0*(rects[:,-1])/rects[:,-2]),]
		rect[1] += int(self.frameHeight *(1-self.cropKeep))
		return rect
		
	def updateStickData(self, client="test", minHSV=np.array([27,55,115]), 
						maxHSV=np.array([45,255,255]), cropKeep=0.75, 
						morphology=True, savePreprocessImg=False):
		"""
		update the yellow stick data from the specified camera.

		Args:
			client: client name
			minHSV: the lower limit for binalization.
			maxHSV: the upper limit for binalization.
			cropKeep:  crop ratio (>=0.5).
			morphology: (True, default), erosion and dilation.
			savePreprocessImg: save the preprocessed image or not.
		"""
		self.updateFrame(client)
		minPerimeter = self.frameHeight/8.0
		minArea = self.frameHeight*self.frameWidth/1000.0
		frameBin = self.__preprocess(minHSV, maxHSV, cropKeep, morphology)
		if savePreprocessImg:
			self._frameBin = frameBin.copy()
		rect = self.__findStick(frameBin, minPerimeter, minArea)
		if rect == []:
			self.boundRect = []
			self.stickAngle = 0.0
		else:
			self.boundRect = rect
			centerX = rect[0]+rect[2]/2
			width = self.frameWidth *1.0
			self.stickAngle = (width/2-centerX)/width*self.cameraYawRange
			cameraPosition = self.motionProxy.getPosition("Head", 2, True)
			cameraY = cameraPosition[5]
			self.stickAngle += cameraY
				
	def showStickPosition(self):
		"""
		show the stick  position in the current frame.
		"""
		if self.boundRect == []:
			#print("no stick detected.")
			cv2.imshow("stick position", self.frameArray)
		else:
			[x,y,w,h] = self.boundRect
			frame = self.frameArray.copy()
			cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
			cv2.imshow("stick position", frame)

	def slider(self, client):
		"""
		slider for stick detection in HSV color space.
		
		Args:
			client: client name.
		"""
		def __nothing():
			pass
		windowName = "slider for stick detection"
		cv2.namedWindow(windowName)
		cv2.createTrackbar("minH", windowName, 27, 45, __nothing)
		cv2.createTrackbar("minS", windowName, 55, 75, __nothing)
		cv2.createTrackbar("minV", windowName, 115, 150, __nothing)
		cv2.createTrackbar("maxH", windowName, 45, 70, __nothing)
		while 1:
			self.updateFrame(client)
			minH = cv2.getTrackbarPos("minH", windowName)
			minS = cv2.getTrackbarPos("minS", windowName)
			minV = cv2.getTrackbarPos("minV", windowName)
			maxH = cv2.getTrackbarPos("maxH", windowName)
			minHSV = np.array([minH, minS, minV])
			maxHSV = np.array([maxH, 255, 255])
			self.updateStickData(client, minHSV, maxHSV, savePreprocessImg=True)
			cv2.imshow(windowName, self._frameBin)
			self.showStickPosition()
			k = cv2.waitKey(10) & 0xFF
			if k == 27:
				break
		cv2.destroyAllWindows()


class LandMarkDetect(ConfigureNao):
	"""
	detect the landMark.
	"""
	def __init__(self, IP, PORT=9559, cameraId=vd.kTopCamera, landMarkSize=0.105):
		super(LandMarkDetect, self).__init__(IP, PORT)
		self.cameraId = cameraId
		self.cameraName = "CameraTop" if cameraId==vd.kTopCamera else "CameraBottom"
		self.landMarkSize = landMarkSize
		self.disX = 0
		self.disY = 0
		self.dist = 0
		self.yawAngle = 0
		self.cameraProxy.setActiveCamera(self.cameraId)
		
	def updateLandMarkData(self, client="landMark"):
		"""
		update landMark information

		Args:
			client: client name
		Return:
			None.
		"""
		if self.cameraProxy.getActiveCamera() != self.cameraId:
			self.cameraProxy.setActiveCamera(self.cameraId)
		self.landMarkProxy.subscribe(client)
		markData = self.memoryProxy.getData("LandmarkDetected")
		self.cameraProxy.unsubscribe(client)
		if (markData is None or len(markData)==0):
			self.disX = 0
			self.disY = 0
			self.dist = 0
			self.yawAngle = 0
		else:
			wzCamera = markData[1][0][0][1]
			wyCamera = markData[1][0][0][2]
			angularSize = markData[1][0][0][3]
			distCameraToLandmark = self.landMarkSize / ( 2 * math.tan( angularSize / 2))
			transform = self.motionProxy.getTransform(self.cameraName, 2, True)
			transformList = almath.vectorFloat(transform)
			robotToCamera = almath.Transform(transformList)
			cameraToLandmarkRotTrans = almath.Transform_from3DRotation(0, wyCamera, wzCamera)
			cameraToLandmarkTranslationTrans = almath.Transform(distCameraToLandmark, 0, 0)
			robotToLandmark = robotToCamera * \
							  cameraToLandmarkRotTrans * \
							  cameraToLandmarkTranslationTrans
			self.disX = robotToLandmark.r1_c4
			self.disY = robotToLandmark.r2_c4
			self.dist = np.sqrt(self.disX**2 + self.disY**2)
			self.yawAngle = math.atan2(self.disY, self.disX)

	def getLandMarkData(self):
		"""
		get landMark information.

		Return:
			a list of disX, disY, dis, and yaw angle.
		"""
		return [self.disX, self.disY, self.dist, self.yawAngle]

	def showLandMarkData(self):
		"""
		show landmark information detected.
		"""
		print("disX = ", self.disX)
		print("disY = ", self.disY)
		print("dis = ", self.dist)
		print("yaw angle = ", self.yawAngle*180.0/np.pi)