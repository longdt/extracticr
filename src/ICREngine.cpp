/*
 * ICREngine.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: thienlong
 */

#include <ICREngine.h>
#include <opencv2/highgui/highgui.hpp>
#include "recognizer.h"
#include "blob.h"
#include "CARLocator.h"
#include "preprocessor.h"

#include "NumberRecognizer.h"
digit_recognizer recognizer;


bool isPeriod(cv::Rect carBox, int middleLine, Blob& blob) {
	cv::Rect brect = blob.boundingRect();
	if (middleLine > brect.y + brect.height) {
		return false;
	}
	float score = blob.points.size() / (float) (brect.width * brect.height);
	score *= std::min(brect.width, brect.height) / (float) std::max(brect.width, brect.height);
	return score > 0.8;
}

void removeDelimiter(cv::Rect carBox, int middleLine, Blobs& blobs, float slantAngle) {
	for (int idx = blobs.size() - 1; idx >= 0; --idx) {
		Blob& blob = *(blobs[idx]);
		cv::Rect brect = blob.boundingRect();
		if (middleLine > brect.y + brect.height) {
			return;
		}
		float score = blob.points.size() / (float) (brect.width * brect.height);
		score *= std::min(brect.width, brect.height) / (float) std::max(brect.width, brect.height);
		if(score <= 0.8) {
			cv::Mat digit = makeDigitMat(blob, slantAngle);
			digit_recognizer::result rs = recognizer.predict(digit);
			if (rs.label() == 1) {
				blobs.erase(idx);
			}
		}
	}
}

namespace icr {

int detectTerminator(Blobs& blobs) {
	if (blobs.size() <= 1) {
		return -1;
	}
	int firstHeighest = 0;
	int secondHeighest = 0;
	int index = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		cv::Rect rect = blobs[i]->boundingRect();
		if (firstHeighest < rect.height) {
			secondHeighest = firstHeighest;
			firstHeighest = rect.height;
			index = i;
		}
	}
	if (firstHeighest / (float) secondHeighest > 2) {
		return index;
	}
	return -1;
}

int detectPeriod(Blobs& blobs) {
	cv::Rect rect = blobs.boundingRect();
	int middleLine = rect.y + rect.height / 2;
	for (size_t i = 0; i < blobs.size(); ++i) {
		if (isPeriod(rect, middleLine, *blobs[i])) {
			return i;
		}
	}
	return -1;
}

ICREngine::ICREngine() {
}

ICREngine::~ICREngine() {
}

std::string ICREngine::recognite(cv::Mat& cheque) {
	CARLocator locator(cheque);
	Blobs blobs;
	locator.getHandwrittenBlobs(blobs);
	cv::Rect loc = blobs.boundingRect();
	blobs.move(-loc.x, -loc.y);
	float angle = deslant(loc.size(), blobs);
	cv::Mat car;
	drawBlobs(blobs, car);
	cv::imshow("hwimg", car);
	loc = blobs.boundingRect();
//	removeDelimiter(loc, loc.y + loc.height / 2, blobs, angle);
	groupVertical(blobs);
//	defragment(car, blobs);
	NumberRecognizer nr(blobs);
	//return extractDigit(blobs, angle);
	return nr.predict();
}


} /* namespace icr */
