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
private:
	cv::Mat cheqImg;
	cv::Rect getRMLocation(Blobs& blobs, cv::Rect& carLoc);
public:
	/*
	 * cheqImg is gray level image
	 * */
	CARLocator(cv::Mat& cheqImg);
	cv::Rect getCARLocation();
	cv::Rect getMPRLocation();
	cv::Rect getRMLocation();
	void getHandwrittenBlobs(Blobs& blobs);
	virtual ~CARLocator();
};

} /* namespace icr */

#endif /* CARLOCATOR_H_ */
