#include "visualTask.h"

// NAO Nao();
namespace
{
	ALMotionProxy _MotProxy(_robotIp, _PORT); 
	ALTextToSpeechProxy _tts(_robotIp, _PORT);
	ALVideoDeviceProxy _CameraProxy(_robotIp, _PORT);
}


const float cameraPitchRange = 47.64; 
const float cameraYawRange = 60.97;

/*
Description: get one frame from the camera.
Arguments:
clientName - default: "test".
cameraId - default: bottom camera (1).
ColorSpace - default: BGR (kBGRColorSpace).
Fps - default: 1
Return: one frame image.
*/
Mat getFrame(string clientName, const int cameraId, const int ColorSpace, const int Fps)
{
	_CameraProxy.setParameterToDefault(cameraId, kCameraAutoExpositionID);
	_CameraProxy.setParameterToDefault(cameraId, kCameraExposureID);
	_CameraProxy.setParameterToDefault(cameraId, kCameraExposureAlgorithmID);
	_CameraProxy.setActiveCamera(cameraId); // active the camera on the bottom.
	clientName = _CameraProxy.subscribe(clientName, kVGA, ColorSpace,  Fps);
	Mat imgHeader = Mat(Size(640, 480),CV_8UC3); 

	ALValue img = _CameraProxy.getImageRemote(clientName);// Retrieves the latest image from the video source
	while(!img.getSize())
	{  
		std::cout<<"img is empty"<<std::endl;
		img = _CameraProxy.getImageRemote(clientName);
	}
	imgHeader.data = (uchar*)img[6].GetBinary(); 
	Mat frame = imgHeader.clone(); 
	_CameraProxy.releaseImage(clientName);
	_CameraProxy.unsubscribe(clientName);
	return frame;
}



/*
Description: get one frame from the camera with specified exposure time.
Arguments:
clientName - default: "test".
cameraId - default: bottom camera (1).
ColorSpace - default: BGR (kBGRColorSpace).
Fps - default: 1
exposureTime - (0, 2500), default: 300.
Return: one frame image.
*/
Mat getFrameWithExposureTime(string clientName, const int cameraId, const int ColorSpace, const int Fps, int exposureTime)
{
	_CameraProxy.setParameterToDefault(cameraId, kCameraAutoExpositionID);
	_CameraProxy.setParameterToDefault(cameraId, kCameraExposureID);
	//_CameraProxy.setParameterToDefault(cameraId, kCameraExposureAlgorithmID);
	_CameraProxy.setParameter(cameraId, kCameraAutoExpositionID, 0);
	bool changeExposureTime = _CameraProxy.setParameter(cameraId, kCameraExposureID, exposureTime); // exposure time
	if (changeExposureTime)
	{
		cout<<"exposure time applied.\n";
	}
	_CameraProxy.setActiveCamera(cameraId); // active the camera on the bottom.
	clientName = _CameraProxy.subscribe(clientName, kVGA, ColorSpace,  Fps);

	Mat imgHeader = Mat(Size(640, 480),CV_8UC3); 
	ALValue img = _CameraProxy.getImageRemote(clientName);// Retrieves the latest image from the video source
	while(!img.getSize())
	{  
		std::cout<<"img is empty"<<std::endl;
		img = _CameraProxy.getImageRemote(clientName);
	}
	imgHeader.data = (uchar*)img[6].GetBinary(); 
	Mat frame = imgHeader.clone(); 
	_CameraProxy.releaseImage(clientName);
	_CameraProxy.unsubscribe(clientName);
	return frame;
}


/*
Description: get one frame from the camera with specified exposure time.
Arguments:
clientName - default: "test".
cameraId - default: bottom camera (1).
ColorSpace - default: BGR (kBGRColorSpace).
Fps - default: 1
exposureAlgorithm - [0, 1, 2, 3], default: 1
Return: one frame image.
*/
Mat getFrameWithExposureAlgorithm(string clientName, const int cameraId, const int ColorSpace, const int Fps, int exposureAlgorithm)
{
	_CameraProxy.setParameterToDefault(cameraId, kCameraAutoExpositionID);
	_CameraProxy.setParameterToDefault(cameraId, kCameraExposureID);
	_CameraProxy.setActiveCamera(cameraId); // active the camera on the bottom.
	bool changeExposureAlgorithm = _CameraProxy.setParameter(cameraId, kCameraExposureAlgorithmID, exposureAlgorithm); // exposureAlgorithm
	cout<<"bool = "<<changeExposureAlgorithm<<endl;
	if (changeExposureAlgorithm)
	{
		cout<<"exposure algorithm applied.\n";
	}
	//int ID = _CameraProxy.getParameter(clientName, kCameraExposureAlgorithmID);
	//cout<<"kCameraExposureAlgorithmID = " << _CameraProxy.getParameter(clientName, kCameraExposureAlgorithmID)<<endl;
	clientName = _CameraProxy.subscribe(clientName, kVGA, ColorSpace,  Fps);

	Mat imgHeader = Mat(Size(640, 480),CV_8UC3); 
	ALValue img = _CameraProxy.getImageRemote(clientName);// Retrieves the latest image from the video source
	while(!img.getSize())
	{  
		std::cout<<"img is empty"<<std::endl;
		img = _CameraProxy.getImageRemote(clientName);
	}
	imgHeader.data = (uchar*)img[6].GetBinary(); 
	Mat frame = imgHeader.clone(); 
	_CameraProxy.releaseImage(clientName);
	_CameraProxy.unsubscribe(clientName);
	return frame;
}


/*
Description: release frame and unsubscribe the camera.
Arguments:
clientName - default: "test".
Return: none.
*/
void releaseFrame(string clientName)
{
	_CameraProxy.releaseImage(clientName);
	_CameraProxy.unsubscribe(clientName);
	destroyAllWindows();
}


/************************ for ball detection *******************************/
/*
Description: split the image to BGR channel with the param of Hm.
Arguments:
srcImage - BGR image.
Hm - param to split the specified color.
color - enum type {BLUE, GREEN, RED}.
Return: one-channel gray image
*/
Mat splitChannelBGR(Mat srcImage, int Hm, COLOR color)
{
	Mat _srcImage = srcImage.clone();
	Mat channels[3], srcImageChannel;
	split(_srcImage, channels);

	switch (color)
	{
	case BLUE:
		channels[1] = channels[1].mul(.1*Hm);
		channels[2] = channels[2].mul(.1*Hm);
		channels[0] = channels[0] - channels[1] - channels[2];
		channels[0] = 3 * channels[0];
		srcImageChannel = channels[0];
		break;
	case GREEN:
		channels[0] = channels[0].mul(.1*Hm);
		channels[2] = channels[2].mul(.1*Hm);
		channels[1] = channels[1] - channels[0] - channels[2];
		channels[1] = 3 * channels[1];
		srcImageChannel = channels[1];
		break;
	case RED:
		channels[0] = channels[0].mul(.1*Hm);
		channels[1] = channels[1].mul(.1*Hm);
		channels[2] = channels[2] - channels[0] - channels[1];
		channels[2] = 3 * channels[2];
		srcImageChannel = channels[2];
		break;
	default:
		break;
	}
	GaussianBlur(srcImageChannel, srcImageChannel, Size(9, 9), 2, 2);
	imshow("srcImageChannel", srcImageChannel);
	waitKey(100);
	return srcImageChannel;	 
}


/*
Description: get binary image from an HSV image (transformed from BGR image).
Arguments:
srcImage - BGR image.
color - enum type {BLUE, GREEN, RED}.
Return: binary image
*/
Mat binImageHSV(Mat srcImage, COLOR color)
{
	Mat _srcImage = srcImage.clone();
	cvtColor(_srcImage, _srcImage, CV_BGR2HSV);
	Mat binImage, binImage1;
	int h, s, v;
	int H, S, V;
	int h1 = 145, H1 = 180;

	switch(color)
	{
	case RED:
		h = 0; s = 43; v = 46;
		H = 10; S = 255; V = 255;
		inRange(_srcImage, Scalar(h,s,v), Scalar(H,S,V), binImage);
		inRange(_srcImage, Scalar(h1,s,v), Scalar(H1,S,V), binImage1);
		binImage = binImage + binImage1;
		break;
	default:
		break;
	}
	GaussianBlur(binImage, binImage, Size(9, 9), 2, 2);
	imshow("binaray image", binImage);
	return binImage;
}

/*
Description: find circles with Hough transformation.
Arguments:
srcImage - one-channel gray image.
minDist - the minimum distance between different centers of circle (default: 16).
minRadius - the minimul radius detected (default: 0).
maxRadius - the maximul radius detected (default: 50).
Return: circles detected(vector<Vec3f>:centerX, centerY, radius)
*/
vector<Vec3f> findCircles(Mat srcImageGray, double minDist, double minRadius, double maxRadius)
{
	vector<Vec3f> circles;
	HoughCircles(srcImageGray, circles, CV_HOUGH_GRADIENT, 1, minDist, 200, 16, minRadius, maxRadius);
	return circles;
}


/*
Description: draw all circles on image.
Arguments:
srcImage - a Mat image need to be drawn on.
circles - all circles information (centerX, centerY, radius).
Return: none
*/
void drawCircles(Mat srcImage, vector<Vec3f> circles)
{
	Mat _srcImage = srcImage.clone();
	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		circle(_srcImage, center, radius, Scalar(255, 0, 255), 3, 10);
		circle(_srcImage, center, 3, Scalar(0, 255, 255), 3);
		cout << "circle " << i << ":" << endl;
		cout << "center position = " << center << endl;
		cout << "radius = " << circles[i][2] << endl;
	}
	imshow("srcImage with circles", _srcImage);
	waitKey(1);
}


/*
Description: detect ball with specified color.
Arguments:
frame - one frame image in BGR color space get from Nao's camera.
Hm - param to split the specified color (default: 6).
color - enum type {BLUE, GREEN, RED} (default: RED).
drawResult - show the circles on source image or not.
Return: all ball's information (x, y, radius).
*/
vector<Vec3f> ballDetect(Mat frame, int Hm, COLOR color, COLORSPACE colorSpace, bool drawResult)
{
	vector<Vec3f> circles;
	Mat frameGray;
	switch(colorSpace)
	{
	case BGR:
		frameGray = splitChannelBGR(frame, Hm, color);
		break;
	case HSV:
		frameGray = binImageHSV(frame, color);
		break;
	default:
		break;
	}
	
	circles = findCircles(frameGray);
	if (drawResult)
	{
		drawCircles(frame, circles);
	}
	return circles;
}


/*
Description: verify all possible balls.
Arguments:
circles - all possible circles in frame.
frame - one frame image in BGR color space get from Nao's camera.
Return: possible circles.
*/
vector<Vec3f> verifieCircles(vector<Vec3f> circles, Mat frame)
{
	vector<Vec3f> circlesSelect;
	Mat channels[3];
	float tot_color;
	int initX, initY;
	int centerX, centerY, radius;

	if (circles.size() == 0)
	{
		return circles;
	}

	if (circles.size()==1)
	{
		
		centerX = (int)circles[0][0];
		centerY = (int)circles[0][1];
		radius = (int)circles[0][2];
		initX = centerX - 2 * radius;
		initY = centerY - 2 * radius;
		if ((initX<0) || (initY<0) || (initX + 4 * radius>640) || (initY + 4 * radius>480) || (radius<1))
		{
			return circles;
		}
	}

	split(frame, channels);
	float minRatio_RED = 1;
	for (size_t i = 0; i < circles.size(); i++)
	{
		centerX = (int)circles[i][0];
		centerY = (int)circles[i][1];
		radius = (int)circles[i][2];
		initX = centerX - 2 * radius;
		initY = centerY - 2 * radius;

		int countR = 0, countG = 0;
		if ((initX<0) || (initY<0) || (initX + 4 * radius>640) || (initY + 4 * radius>480) || (radius<1))
		{
			printf("circle %d: nothing to do.\n", i);
			continue;
		}

		int count = 0;
		for (int j = initX; j < initX + 4 * radius; j++)
		{
			for (int k = initY; k < initY + 4 * radius; k++)
			{
				float r = channels[2].at<unsigned char>(k, j);
				float g = channels[1].at<unsigned char>(k, j);
				float b = channels[0].at<unsigned char>(k, j);
				tot_color = r*r + g*g + b*b;
				r = r / sqrt(tot_color);
				g = g / sqrt(tot_color);
				b = b / sqrt(tot_color);
				if ((r > 1.2*g) && (r > 1.2*b))
					countR++;
				if ((g > 1.2*r) && (g > 0.0*b))
					countG++;
			}
		}
		float ratio_RED = (float)countR / (16.*radius*radius + 0.0001);
		float ratio_GREEN = (float)countG / (16.*radius*radius + 0.0001);
		printf("circle %d R %f G %f\n", i, ratio_RED, ratio_GREEN);

		if ((ratio_RED >= 0.15) && (ratio_GREEN >= 0.1) && (ratio_RED<minRatio_RED))
		{
			circlesSelect.clear();
			circlesSelect.push_back(circles[i]);
			minRatio_RED = ratio_RED;
			printf("selected circle: %d R %f G %f\n", i, ratio_RED, ratio_GREEN);
		}
	}
	drawCircles(frame, circlesSelect);
	//waitKey(50);
	return circlesSelect;
}


/*
Description: compute the ball position related to Nao.
Arguments:
frame - one frame image in BGR color space get from Nao's camera.
circles - all balls detected.
Return: position of all balls related to Nao (x, y, distance, yaw, flag(number of balls)).
*/
float* computePosition(Mat frame, vector<Vec3f> circles, STANDSTATE standState)
{   
	float* ballPosition = new float [5];
	for (int i = 0; i<5; i++)
	{
		*(ballPosition+i) = 0;
	}

	if (circles.size() == 0)
	{
		cout<<"no balls.\n";
		return ballPosition;
	}

	if (circles.size() > 1)
	{
		ballPosition[4] = circles.size();
		cout<<"many balls.\n";
		return ballPosition;
	}

	const float bottomCameraDirection = (standState == StandInit? 49.2:39.7);
	//float bottomCameraDirection = 49.2; // stand_state = "stand init"
	//if (standState ==  StandUp) // stan_state = "stand up"
	//{
	//	bottomCameraDirection = 39.7;
	//}
	float ballCenterX = circles[0][0];
	float ballCenterY = circles[0][1];
	float ballX, ballY, ballYaw, ballPitch, ballDistance;
	float cameraHeight, cameraX, cameraY;
	vector<float> headPitches, headYaws;
	float headPitch, headYaw;
	float dPitch, dYaw;
	const float ballRadius = 0.02;

	vector<float> cameraPosition = _MotProxy.getPosition("CameraBottom",2,true);
	//vector<float> cameraPosition = Nao.MotProxy->getPosition("CameraBottom",2,true);
	cameraX = cameraPosition.at(0); // X
	cameraY = cameraPosition.at(1); // Y (from right to left)
	cameraHeight = cameraPosition.at(2); // Z

	headPitches = _MotProxy.getAngles("HeadPitch", true);
	//headPitches = Nao.MotProxy->getAngles("HeadPitch", true);
	headPitch = headPitches.at(0);
	headYaws = _MotProxy.getAngles("HeadYaw", true);
	//headYaws = Nao.MotProxy->getAngles("HeadYaw", true);
	headYaw = headYaws.at(0);

	ballPitch = (ballCenterY-240.0)*cameraPitchRange/480.0*PI/180; //y (pitch angle)
	ballYaw = (320.0-ballCenterX)*cameraYawRange/640.0*PI/180; //x (yaw angle)

	dPitch = (cameraHeight-ballRadius)/tan(bottomCameraDirection*PI/180+headPitch+ballPitch);//¶Ô±È
	dYaw = dPitch/cos(ballYaw);
	ballX = dYaw*cos(ballYaw+headYaw)+cameraX;
	ballY = dYaw*sin(ballYaw+headYaw)+cameraY;
	ballYaw = atan2(ballY,ballX);

	// compensation for ballY and ballYaw (stand_state = 0: stand init)
	
	if (standState == StandInit)
	{
		double ky;
		ky = 12.604*ballX*ballX*ballX*ballX - 37.962*ballX*ballX*ballX + 43.163*ballX*ballX - 22.688*ballX + 6.0526;
		ballY = ky*ballY;
		ballYaw = atan2(ballY,ballX);
	}
	
	ballDistance = sqrt(ballX*ballX+ballY*ballY); 
	*(ballPosition) = ballX;
	*(ballPosition+1) = ballY;
	*(ballPosition+2) = ballDistance;
	*(ballPosition+3) = ballYaw;
	*(ballPosition+4) = circles.size();
	return ballPosition;
}


/************************* for yellow stick searching *********************/
/*
Description: preprocess the image for yellow stick searching.
Arguments:
frame - one frame image in BGR color space get from Nao's camera.
minHSV, maxHSV - params for binaryzation.
Return: preprocessed binary image for yellow stick seaching.
*/
Mat preprocess(Mat frame, Scalar minHSV, Scalar maxHSV)
{
	int frameRows = frame.rows;
	int frameCols = frame.cols;
	Mat frameCropHSV, frameCropBin;
	Mat frameCrop = frame(Rect(0,frameRows/2,frameCols,frameRows/2)); // crop
	cvtColor(frameCrop, frameCropHSV, CV_BGR2HSV); //BGR -> HSV
	inRange(frameCropHSV, minHSV, maxHSV, frameCropBin); // binaryzation
	Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));
	Mat element2 = getStructuringElement(MORPH_RECT, Size(6, 6));
	erode(frameCropBin, frameCropBin, element1);
	dilate(frameCropBin, frameCropBin, element2);
	GaussianBlur(frameCropBin, frameCropBin, Size(9, 9), 0);
	//blur(frameCropBin,frameCropBin, Size(3, 3)); // blur
	return frameCropBin;
}


/*
Description: find sticks in a binary image.
Arguments:
frameBin - one binary preprocessed frame image.
Return: all sticks marked with rectangles.
*/
vector<Rect> findSticks(Mat frameBin, int minPerimeter, int minArea)
{
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(frameBin, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);

	vector<vector<Point>> contours_poly(contours.size());
	vector<Rect> boundRect;
	float maxRatio = 0.8, currentRatio;
	for (size_t i = 0; i < contours.size(); i++)
		{
			float currentPerimeter = arcLength(contours[i], true);
			float currentArea = contourArea(contours[i]);
			if((currentPerimeter > minPerimeter) && (currentArea > minArea) )
			{
				approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
				Rect currentRect = boundingRect(Mat(contours_poly[i]));
				currentRatio = currentRect.height*1.0/currentRect.width;
				cout<<"ratio = " << currentRatio <<endl;
				if ((currentRatio>maxRatio) && (currentRect.width<200))
				{
					//cout<<"height = "<<currentRect.height<<endl;
					//cout<<"width = "<<currentRect.width<<endl;
					boundRect.clear();
					boundRect.push_back(currentRect);
					maxRatio = currentRatio;
				}
			}
		}
	return boundRect;
}

/*
Description: draw sticks in source image.
Arguments:
frame - one frame image in BGR color space get from Nao's camera.
boundRect - rectangles used for marking the stick.
Return: none.
*/
void drawSticks(Mat frame, vector<Rect> boundRect)
{
	RNG rng(123);
	Point point;
	int frameRows = frame.rows;
	int frameCols = frame.cols;
	Mat frameCrop = frame(Rect(0,frameRows/2,frameCols,frameRows/2));
	for(size_t i = 0; i < boundRect.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		point.x=(boundRect[i].tl().x + boundRect[i].br().x) / 2;
		point.y=(boundRect[i].tl().y + boundRect[i].br().y) / 2;
		rectangle(frameCrop, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
		circle(frameCrop, point, 4, color, -1, 8, 0);
	}
	imshow("srcImage with marked sticks",frameCrop);
	waitKey(1000);
	destroyAllWindows();
}


/*
Description: detect yellow stick from given image.
Arguments:
frame - one frame image in BGR color space get from Nao's camera.
minHSV, maxHSV - params for binaryzation.
drawResult - show the circles on source image or not.
Return: stick data ([direction, flag]).
*/
float* stickDetect(Mat frame, Scalar minHSV, Scalar maxHSV, int minPerimeter, int minArea, bool drawResult)
{
	float* stickData = new float [2];
	*stickData = 0;
	*(stickData + 1) = 0;
	Mat frameBin;
	vector<Rect> boundRect;
	Point point;
	frameBin = preprocess(frame, minHSV, maxHSV);
	imshow("frameBin", frameBin);
	waitKey(1000);
	boundRect = findSticks(frameBin, minPerimeter, minArea);

	if (drawResult)
	{
		drawSticks(frame, boundRect);
	}

	switch (boundRect.size())
	{
	case 0:
		cout<<"no sticks.\n";
		return stickData;
	case 1:
		cout<< "see the stick.\n";
		point.x=(boundRect[0].tl().x + boundRect[0].br().x) / 2;
		point.y=(boundRect[0].tl().y + boundRect[0].br().y) / 2;		
		*stickData =(320.0f-point.x)*cameraYawRange/640.0f*PI/180.0;
		*(stickData + 1) = 1;
		return stickData;
	default:
		cout<<"many sticks.\n";
		*(stickData + 1) = boundRect.size();
		return stickData;
	}
}