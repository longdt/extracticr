/*
 * GeoContextModel.h
 *
 *  Created on: Nov 7, 2014
 *      Author: thienlong
 */

#ifndef INCLUDE_GEOCONTEXTMODEL_H_
#define INCLUDE_GEOCONTEXTMODEL_H_
#include "cnn/tiny_cnn.h"

#include "blob.h"
using namespace tiny_cnn;

namespace icr {
class GeoContext;
class GeoContextModel {
private:
	network<mse, gradient_descent> uigModel;
	network<mse, gradient_descent> bigModel;
	network<mse, gradient_descent> ucgModel;
	network<mse, gradient_descent> bcgModel;
public:
	GeoContextModel();
	float predict(const GeoContext& content);
	virtual ~GeoContextModel();
};

struct UCGContext {
	//ucg feature
	cv::Rect box;
	cv::Point2f gravityCenter;
	cv::Point2f geoCenter;
	cv::Point2f gravityCenterLine;
	cv::Point2f geoCenterLine;
	float meanProjectProfile[2];
	float meanOutProfile[4];
	float deviOutProfile[4];
};

class GeoContext {
private:
	float strHeight;
	bool hasPrev;
	UCGContext curUcg;
	//bcg feature
	UCGContext prevUcg;
	cv::Rect box2Segms;
public:
	GeoContext(float strHeight, Blobs& segms, int fromIdx, int endIdx);
	GeoContext(float strHeight, Blobs& segms, int fromIdx, int endIdx, GeoContext& prev);
	void getUCGVector(vec_t& output);
	void getUIGVector(vec_t& output);
	void getBCGVector(vec_t& output);
	void getBIGVector(vec_t& output);
};

} /* namespace icr */

#endif /* INCLUDE_GEOCONTEXTMODEL_H_ */
