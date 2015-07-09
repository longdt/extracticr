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
#include <vector>

#include "cnn/util.h"
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
	static std::vector<label_t> CLASS_MAP;
	GeoContextModel();
	void loadModel(std::string& mpath);
	void predictUnary(GeoContext& context, vec_t& ucg, vec_t& uig);
	void predictBinary(GeoContext& context, vec_t& bcg, vec_t& big);
	virtual ~GeoContextModel();
};

struct UCGContext {
	//ucg feature
	cv::Rect box;
	float innerGap;
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
	void getUCGVector(UCGContext& ucgCtx, vec_t& output);
public:
	GeoContext(float strHeight, Blobs& segms, int fromIdx, int endIdx);
	GeoContext(float strHeight, Blobs& segms, int fromIdx, int endIdx, GeoContext& prev);
	GeoContext(float strHeight, UCGContext& curUcg, UCGContext& prevUcg);
	void setPrevContext(GeoContext &ctx);
	void getUCGVector(vec_t& output);
	void getUIGVector(vec_t& output);
	void getBCGVector(vec_t& output);
	void getBIGVector(vec_t& output);
	float getStrHeight();
	cv::Rect getCurBoundingRect();
};

class NumberModel {
private:
	bool decimal;
public:
	NumberModel(bool decimal);
	float getScore(std::vector<label_t> labels, label_t l);
	float getFinalScore(std::vector<label_t> labels, label_t l);
};

} /* namespace icr */

#endif /* INCLUDE_GEOCONTEXTMODEL_H_ */
