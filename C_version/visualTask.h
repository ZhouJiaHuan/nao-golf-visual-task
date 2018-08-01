#ifndef VISUALTASK_H
#define VISUALTASK_H

// #define  robotIP  "192.168.1.102"
#include <iostream>
#include <string>
#include "configureNao.h"

using namespace std;
using namespace AL;
using namespace cv;



#define pi 3.1415926
#define PI pi


enum COLOR {BLUE, GREEN, RED};
enum STANDSTATE {StandInit, StandUp};
enum COLORSPACE {BGR, HSV};

Mat getFrame(string clientName = "test", const int cameraId = 1, const int ColorSpace = kBGRColorSpace, const int Fps = 1);
Mat getFrameWithExposureTime(string clientName = "test", const int cameraId = 1, const int ColorSpace = kBGRColorSpace, const int Fps = 1,  int exposureTime = 300);
Mat getFrameWithExposureAlgorithm(string clientName = "test", const int cameraId = 1, const int ColorSpace = kBGRColorSpace, const int Fps = 1, int exposureAlgorithm = 1);
void releaseFrame(string clientName = "test");


/* for ball detection */
Mat splitChannelBGR(Mat, int, COLOR);
Mat binImageHSV(Mat, COLOR);
vector<Vec3f> findCircles(Mat, double minDist = 16, double minRadius = 0, double maxRadius = 80);
void drawCircles(Mat, vector<Vec3f>);
vector<Vec3f> verifieCircles(vector<Vec3f>, Mat);
vector<Vec3f> ballDetect(Mat, int Hm = 6, COLOR color = RED, COLORSPACE colorSpace = BGR, bool drawResult = true);
float* computePosition(Mat, vector<Vec3f> circles, STANDSTATE standState = StandInit);


/* for yellow stick searching */
Mat preprocess(Mat, Scalar minHSV, Scalar maxHSV);
vector<Rect> findSticks(Mat, int minPerimeter, int minArea);
void drawSticks(Mat, vector<Rect>);
float* stickDetect(Mat frame, Scalar minHSV = Scalar(28, 55, 115), Scalar maxHSV = Scalar(45, 255, 255), 
				   int minPerimeter = 80, int minArea = 150, bool drawResult = true);
#endif