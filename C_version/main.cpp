/*
 * Copyright (c) 2012-2015 Aldebaran Robotics. All rights reserved.
 * Use of this source code is governed by a BSD-style license that can be
 * found in the COPYING file.
 */


#include <iostream>
#include <math.h>
#include <string>
#include <cmath>
#include<vector>
#include <process.h>
#include <Windows.h>
#include "configureNao.h"
#include "visualTask.h"

#include <time.h>

using namespace std;
using namespace AL;
using namespace cv;

ALVideoDeviceProxy cameraProxy(_robotIp, 9559);
ALTextToSpeechProxy tts(_robotIp, 9559);
ALMotionProxy motionProxy(_robotIp, 9559); 
ALRobotPostureProxy postureProxy(_robotIp, 9559);
//AL::ALMemoryProxy memoryProxy(robotIP, 9559);     
//  
//AL::ALRobotPostureProxy postureProxy(robotIP, 9559);
//AL::ALTextToSpeechProxy tts(robotIP, 9559);
//AL::ALVideoDeviceProxy cameraProxy(robotIP, 9559);
//AL::ALVisualCompassProxy compassProxy(robotIP,9559);
//AL::ALLandMarkDetectionProxy landmarkProxy(robotIP, 9559);


float* redballData;
float* yellowstickData;

void ThreadFun1(void *paramter)
{
	tts.say("searching");
	cout<<"searching\n";
	for(;;)
	{
		//Sleep(1000);
		motionProxy.angleInterpolationWithSpeed("HeadYaw", -0.785f, 0.02f);	
		
	}
	 _endthread();
}

void ThreadFun2(void *parameter)
{
	cout<< "***************** red ball detection test ****************\n";
	for(;;)
	{
		cout<< "*********************************\n";
		for (int exposureAlgorithnId = 0; exposureAlgorithnId <= 3; exposureAlgorithnId++)
		{
			Mat frame = getFrame("test", 1, kBGRColorSpace, 1);

			//cout<<"************" << "exposure time = "<< exposureAlgorithnId*600+300 << "************" <<endl;
			//Mat frame = getFrameWithExposureTime("test", 1, kBGRColorSpace, 1, exposureAlgorithnId*600+300);

			//cout<<"************" << "exposure algorithm index = "<< exposureAlgorithnId << "************" <<endl;
			//Mat frame = getFrameWithExposureAlgorithm("test", 1, kBGRColorSpace, 1, exposureAlgorithnId);

			vector<Vec3f> circles, circlesVerified;
			Vec3f circleSelect;
			circles = ballDetect(frame, 6, RED, BGR, true);
			circlesVerified = verifieCircles(circles, frame);
			redballData = computePosition(frame, circlesVerified, StandInit);

			int flag = redballData[4];
			switch (flag)
			{
				case 0:
					tts.say("no balls");
					break;
				case 1:
					tts.say("find the ball");
					cout<<"x = "<< redballData[0]<<endl;
					cout<<"y = "<< redballData[1]<<endl;
					cout<<"d = "<< redballData[2]<<endl;
					cout<<"yaw = "<< redballData[3]/pi*180<<endl;
					break;
				default:
					tts.say("many balls");
			}
			//cin.get();
		}

	}
	_endthread();

}

int main()
{
	/*
	cout<< "***************** thread test ****************\n";
	postureProxy.goToPosture("StandInit", 0.1f);
	motionProxy.angleInterpolationWithSpeed("HeadYaw", 0.785f, 0.1f);
	motionProxy.angleInterpolationWithSpeed("HeadPitch", 0.373, 0.1f);
	_beginthread(ThreadFun1, 0, NULL);
	_beginthread(ThreadFun2, 0, NULL);
	while(1)
	{
	}
	*/
	

	
	
	cout<< "***************** red ball detection test ****************\n";
	postureProxy.goToPosture("StandInit", 0.2); // StandInit, StandUp
	motionProxy.setMoveArmsEnabled(true, false);		

	//motionProxy.angleInterpolationWithSpeed("HeadYaw",0.0f, 0.1f);	
	//motionProxy.angleInterpolationWithSpeed("HeadPitch", 0.373, 0.2f);
	for(;;)
	{
		cout<< "*********************************\n";
		for (int exposureAlgorithnId = 0; exposureAlgorithnId <= 3; exposureAlgorithnId++)
		{
			//motionProxy.angleInterpolationWithSpeed("HeadYaw",0.0f, 0.1f);	
			//motionProxy.angleInterpolationWithSpeed("HeadPitch", 0.373, 0.2f);
			vector<float> Rotation1, Rotation2;
			float pitch, yaw;
			Rotation1 = motionProxy.getAngles("HeadPitch", true); // Z axis
			pitch=Rotation1.at(0);
			Rotation2 = motionProxy.getAngles("HeadYaw", true); // Y axis
			yaw=Rotation2.at(0);
			cout<<"pitch = "<<pitch<<endl;
			cout<<"yaw = "<<yaw<<endl;

			time_t time1, time2;
			time1 = time(NULL);
			Mat frame = getFrame("test", 1, kBGRColorSpace, 1);
			time2 = time(NULL);
			cout<<"get frame time = "<< (time2-time1)<<endl;
			//cout<<"************" << "exposure time = "<< exposureAlgorithnId*400+1 << "************" <<endl;
			//Mat frame = getFrameWithExposureTime("test", 1, kBGRColorSpace, 1, exposureAlgorithnId*400+1);

			//cout<<"************" << "exposure algorithm index = "<< exposureAlgorithnId << "************" <<endl;
			//Mat frame = getFrameWithExposureAlgorithm("test", 1, kBGRColorSpace, 1, exposureAlgorithnId);

			vector<Vec3f> circles, circlesVerified;
			Vec3f circleSelect;
			circles = ballDetect(frame, 6, RED, HSV, false);
			circlesVerified = verifieCircles(circles, frame);
			redballData = computePosition(frame, circlesVerified, StandInit);

			int flag = redballData[4];
			switch (flag)
			{
				case 0:
					//tts.say("no balls");
					break;
				case 1:
					//tts.say("find the ball");
					
					cout<<"x = "<< redballData[0]<<endl;
					cout<<"y = "<< redballData[1]<<endl;
					cout<<"d = "<< redballData[2]<<endl;
					cout<<"yaw = "<< redballData[3]/pi*180<<endl;
					
					break;
				default:
					//tts.say("many balls");
					break;
			}
			waitKey(20);
			//destroyAllWindows();
			//cin.get();
		}
	}
	

	
	/*
	cout<< "***************** yellow stick detection test ****************\n";
	tts.say("stick  searching");
	postureProxy.goToPosture("StandInit", 0.2);
	motionProxy.setMoveArmsEnabled(true, false);			     
	motionProxy.angleInterpolationWithSpeed("HeadYaw",0.0f, 0.1f);	
	motionProxy.angleInterpolationWithSpeed("HeadPitch",-0.17f, 0.1f);
	for(;;)
	{
		cout<< "*********************************\n";
		for (int exposureAlgorithnId = 0; exposureAlgorithnId <= 3; exposureAlgorithnId++)
		{
			Mat frame = getFrame("test", 0, kBGRColorSpace, 1);

			//cout<<"************" << "exposure time = "<< exposureAlgorithnId*200 << "************" <<endl;
			//Mat frame = getFrameWithExposureTime("test", 0, kBGRColorSpace, 1, exposureAlgorithnId*200+1);

			//cout<<"************" << "exposure algorithm index = "<< exposureAlgorithnId << "************" <<endl;
			//Mat frame = getFrameWithExposureAlgorithm("test", 0, kBGRColorSpace, 1, exposureAlgorithnId);
			
			yellowstickData = stickDetect(frame);
			int flag = (int)yellowstickData[1];
			switch (flag)
			{
				case 0:
					tts.say("no sticks");
					break;
				case 1:
					tts.say("see the stick");
					cout<<"yaw = " << yellowstickData[0]/pi*180 << endl;
					break;
				default:
					cout<<"stick num = "<<flag<<endl;
					tts.say("many sticks");
			}
		}
		Sleep(1000);
	}
	*/

	return 0;
}