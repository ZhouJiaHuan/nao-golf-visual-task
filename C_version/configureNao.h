#ifndef CONFIGURENAO_H
#define CONFIGURENAO_H
#include <iostream>
#include <alvalue/alvalue.h>
#include <alcommon/alproxy.h>
#include <alcommon/albroker.h>
#include <alvision/alimage.h>
#include <alvision/alvisiondefinitions.h>
#include <althread/alcriticalsection.h>
#include <boost/shared_ptr.hpp>
#include <alproxies/alvideodeviceproxy.h>
#include <alproxies/almemoryproxy.h>
#include <alproxies/almotionproxy.h>
#include <althread/almutex.h>
#include <alproxies/altexttospeechproxy.h>
#include <althread/almutex.h>
#include <alerror/alerror.h>
#include <alproxies/alpreferencemanagerproxy.h>
#include <alproxies/alfsrproxy.h>
#include <qi/log.hpp>
#include <alerror/alerror.h>
#include <alproxies/almemoryproxy.h>
#include <alproxies/almotionproxy.h>
#include <alproxies/alrobotpostureproxy.h>
#include <alproxies/altexttospeechproxy.h>
#include <alproxies/alredballdetectionproxy.h>
//#include <alproxies/alnotificationmanagerproxy.h>
#include <process.h>
#include <alproxies/albehaviormanagerproxy.h>
#include <alproxies/alvisualcompassproxy.h>

#include <alproxies/alvideodeviceproxy.h>
#include <alvision/alvisiondefinitions.h>
#include <alvision/alimage.h>
#include <alproxies/alsystemproxy.h>

#include <alproxies/dcmproxy.h>
#include <alproxies/allandmarkdetectionproxy.h>
#include <almath/types/altransform.h>
#include <opencv2/opencv.hpp>
using namespace AL;
using namespace std;

#define _robotIp "192.168.1.101"
#define _PORT 9559

#endif