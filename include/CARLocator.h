/*
 * CARLocator.h
 *
 *  Created on: Oct 8, 2014
 *      Author: thienlong
 */

#ifndef CARLOCATOR_H_
#define CARLOCATOR_H_
#include "opencv2/core/core.hpp"

#include "blob.h"
namespace icr {
void trimNoises(Blobs& blobs);
bool intersect(int y0, int y1, cv::Rect rect);
class CARLocator {
protected:
	cv::Mat cheqImg;
	virtual cv::Rect getRMLocation(Blobs& blobs, cv::Rect& carLoc);
public:
	/*
	 * cheqImg is gray level image
	 * */
	explicit CARLocator(cv::Mat& cheqImg);
	virtual cv::Rect getCARLocation();
	virtual cv::Rect getMPRLocation();
	cv::Rect getRMLocation();
	virtual void getHandwrittenBlobs(Blobs& blobs);
	virtual ~CARLocator();
};

class PhCARLocator : public CARLocator {
private:
	bool boundingBox;
	cv::Mat mprImg;
public:
	explicit PhCARLocator(cv::Mat& cheqImg);
	virtual cv::Rect getRMLocation(Blobs& blobs, cv::Rect& carLoc);
	virtual cv::Rect getMPRLocation();
	virtual cv::Rect getCARLocation();
	virtual void getHandwrittenBlobs(Blobs& blobs);
};

class PrintedCARLocator : public CARLocator {
private:
	bool boundingBox;
	cv::Mat mprImg;
public:
	explicit PrintedCARLocator(cv::Mat& cheqImg);
	virtual void getHandwrittenBlobs(Blobs& blobs);
	virtual cv::Rect getCARLocation();
	virtual cv::Rect getRMLocation(Blobs& blobs, cv::Rect& carLoc);
};

} /* namespace icr */

#endif /* CARLOCATOR_H_ */
