/*
 * com_eprotea_icrengine_ICREngine.c
 *
 *  Created on: Feb 2, 2015
 *      Author: thienlong
 */




#include "com_eprotea_icrengine_ICREngine.h"

#include <ICREngine.h>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include <string>

#include "util/misc.h"
using namespace cv;
static icr::ICREngine engine;

JNIEXPORT void JNICALL Java_com_eprotea_icrengine_ICREngine_predictCA
  (JNIEnv *env, jobject thisObj, jbyteArray chqImg, jdoubleArray output) {
	jbyte* imgData = env->GetByteArrayElements(chqImg, NULL);
	int length = env->GetArrayLength(chqImg);
	Mat img = imdecode(Mat(1, length, CV_8UC1, imgData), 0);
	env->ReleaseByteArrayElements(chqImg, imgData, 0);
	if (img.empty()) {
		jclass Exception = env->FindClass("com/eprotea/icrengine/ICRException");
		env->ThrowNew(Exception,"Invalid image data format");
		return;
	} else if (img.cols < 1300 || img.rows < 650) {
		jclass Exception = env->FindClass("com/eprotea/icrengine/ICRException");
		env->ThrowNew(Exception,"Image is too small. Expected Size is around 700x1410");
	}
	string amount = engine.recognite(img);
	amount = removeDelimiter(amount);
	if (amount.empty()) {
		return;
	}
	jdouble* result = env->GetDoubleArrayElements(output, NULL);
	istringstream iss(amount);
	iss >> result[0];
	env->ReleaseDoubleArrayElements(output, result, 0);
}

JNIEXPORT void JNICALL Java_com_eprotea_icrengine_ICREngine_loadModels
  (JNIEnv *env, jclass classObj, jstring modelPath) {
	const char* cmPath = env->GetStringUTFChars(modelPath, NULL);
	icr::ICREngine::loadModels(cmPath);
	env->ReleaseStringUTFChars(modelPath, cmPath);
}
