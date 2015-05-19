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

class CARLocator {
protected:
	cv::Mat cheqImg;
	virtual cv::Rect getRMLocation(Blobs& blobs, cv::Rect& carLoc);
public:
	/*
	 * cheqImg is gray level image
	 * */
	CARLocator(cv::Mat& cheqImg);
	virtual cv::Rect getCARLocation();
	virtual cv::Rect getMPRLocation();
	cv::Rect getRMLocation();
	virtual void getHandwrittenBlobs(Blobs& blobs);
	virtual ~CARLocator();
};

class PhCARLocator : public CARLocator {
	using CARLocator::CARLocator;
private:
	bool boundingBox;
public:
	virtual cv::Rect getRMLocation(Blobs& blobs, cv::Rect& carLoc);
	virtual cv::Rect getMPRLocation();
	virtual cv::Rect getCARLocation();
	virtual void getHandwrittenBlobs(Blobs& blobs);
};

} /* namespace icr */

#endif /* CARLOCATOR_H_ */
