/*
 * PrintedEngine.cpp
 *
 *  Created on: Jun 9, 2015
 *      Author: thienlong
 */

#include "ICREngine.h"

#include "CARLocator.h"

#include "NumberRecognizer.h"
namespace icr {
PrintedEngine::PrintedEngine(int type) : ICREngine(type) {

}

std::string PrintedEngine::recognite(cv::Mat& cheque, float* confidence) {
	CARLocator* locator = new PrintedCARLocator(cheque);
	Blobs blobs;
	locator->getHandwrittenBlobs(blobs);
	delete locator;
	if (blobs.size() == 0) {
		return "";
	}




	cv::Rect loc = blobs.boundingRect();
	blobs.move(-loc.x, -loc.y);
	NumberRecognizer nr(blobs);
	return nr.predict(confidence);
}

}
