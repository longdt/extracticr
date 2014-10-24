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
digit_recognizer recognizer;

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

bool isPeriod(cv::Rect carBox, int middleLine, Blob& blob) {
	cv::Rect brect = blob.boundingRect();
	if (middleLine < brect.y || middleLine > brect.y + brect.height) {
		return false;
	}
	float score = blob.points.size() / (float) (brect.width * brect.height);
	score *= std::min(brect.width, brect.height) / (float) std::max(brect.width, brect.height);
	return score > 0.8;
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

//std::string recognite(Blobs& blobs) {
//	int termIdx = detectTerminator(blobs);
//	if (termIdx != -1) {
//		blobs.erase(termIdx);
//	}
//	cv::Rect rect = boundingRect(blobs);
//	int middleLine = rect.y + rect.height / 2;
//	std::string result;
//	for (size_t i = 0; i < blobs.size(); ++i) {
//		if (isPeriod(rect, middleLine, *blobs[i])) {
//			result = result + ".";
//			continue;
//		}
//		result =
//	}
//}

ICREngine::ICREngine() {
	computeDigitWidth("/media/thienlong/linux/CAR/cvl-digits/train", digitStatistics);
}

ICREngine::~ICREngine() {
	// TODO Auto-generated destructor stub
}

std::string ICREngine::recognite(cv::Mat& cheque) {
	CARLocator locator(cheque);
	Blobs blobs;
	locator.getHandwrittenBlobs(blobs);
	cv::Rect loc = blobs.boundingRect();
	blobs.move(-loc.x, -loc.y);
	float angle = deslant(loc.size(), blobs);
	auto car = drawBlob(blobs);
	cv::imshow("hwimg", car);
	std::vector<int> labels;
	groupVertical(blobs, labels);
//	defragment(car, blobs);
	return extractDigit(blobs, angle);
}


} /* namespace icr */
