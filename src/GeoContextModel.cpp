/*
 * GeoContextModel.cpp
 *
 *  Created on: Nov 7, 2014
 *      Author: thienlong
 */

#include <GeoContextModel.h>

#include "preprocessor.h"

#include "util/misc.h"
namespace icr {
std::vector<label_t> GeoContextModel::CLASS_MAP = {0, 1, 2, 0, 3, 4, 2, 5, 0, 5, 6, 7};
GeoContextModel::GeoContextModel() {
	//load uigModel
	auto F1 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(12, 20);
	auto F2 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(20, 4);
	uigModel.add(F1);
	uigModel.add(F2);
	std::ifstream uigfs("uig");
	if (uigfs.is_open()) {
		uigfs >> *F1 >> *F2;
		uigfs.close();
	}
	//load ucgModel
	F1 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(24, 100);
	F2 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(100, 8);
	ucgModel.add(F1);
	ucgModel.add(F2);
	std::ifstream ucgfs("ucg");
	if (ucgfs.is_open()) {
		ucgfs >> *F1 >> *F2;
		ucgfs.close();
	}
	//load bcgModel
	F1 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(72, 120);
	F2 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(120, 64);
	bcgModel.add(F1);
	bcgModel.add(F2);
	std::ifstream bcgfs("bcg");
	if (bcgfs.is_open()) {
		bcgfs >> *F1 >> *F2;
		bcgfs.close();
	}
	//load bigModel
	F1 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(13, 80);
	F2 = new fully_connected_layer<network<mse, gradient_descent>, sigmoid_activation>(80, 6);
	bigModel.add(F1);
	bigModel.add(F2);
	std::ifstream bigfs("big");
	if (bigfs.is_open()) {
		bigfs >> *F1 >> *F2;
		bigfs.close();
	}
}

GeoContextModel::~GeoContextModel() {
	// TODO Auto-generated destructor stub
}

void GeoContextModel::predictUnary(GeoContext& context, vec_t& ucg, vec_t& uig) {
	vec_t in;
	context.getUCGVector(in);
	ucgModel.predict(in, &ucg);
	context.getUIGVector(in);
	uigModel.predict(in, &uig);
}
void GeoContextModel::predictBinary(GeoContext& context, vec_t& bcg, vec_t& big) {
	vec_t in;
	context.getBCGVector(in);
	bcgModel.predict(in, &bcg);
	context.getBIGVector(in);
	bigModel.predict(in, &big);
}

void calcUpperOutProfile(cv::Mat& blobImg, float& mean, float& deviation) {
	average<int> avg;
	for (int c = 0; c < blobImg.cols; ++c) {
		for (int r = 0; r < blobImg.rows; ++r) {
			if (blobImg.at<uchar>(r, c) > 0) {
				avg.update(r);
				break;
			}
		}
	}
	mean = avg.mean();
	deviation = avg.sdeviation();
}

void generateUCGContext(Blobs& segms, int fromIdx, int endIdx, UCGContext& ucg) {
	Blob* blob = segms.newBlob(fromIdx, endIdx);
	ucg.box = blob->boundingRect();
	ucg.innerGap = blob->getInnerGap();
	ucg.geoCenter = cv::Point2f(ucg.box.x + ucg.box.width / 2.0f, ucg.box.y + ucg.box.height / 2.0f);
	ucg.gravityCenter = blob->getMassCenter();
	ucg.gravityCenterLine = segms.getMassCenter();
	cv::Rect lineBox = segms.boundingRect();
	ucg.geoCenterLine = cv::Point2f(lineBox.x + lineBox.width / 2.0f, lineBox.y + lineBox.height / 2.0f);
	//compute meanProjectProfile
	ucg.meanProjectProfile[0] = blob->points.size() / (float) ucg.box.width;
	ucg.meanProjectProfile[1] = blob->points.size() / (float) ucg.box.height;
	cv::Mat blobImg = cropBlob(*blob);
	calcUpperOutProfile(blobImg, ucg.meanOutProfile[0], ucg.deviOutProfile[0]);
	cv::Mat temp;
	cv::flip(blobImg, temp, 0);
	calcUpperOutProfile(temp, ucg.meanOutProfile[1], ucg.deviOutProfile[1]);
	cv::transpose(blobImg, temp);
	calcUpperOutProfile(temp, ucg.meanOutProfile[2], ucg.deviOutProfile[2]);
	cv::flip(temp, blobImg, 0);
	calcUpperOutProfile(blobImg, ucg.meanOutProfile[3], ucg.deviOutProfile[3]);
	delete blob;
}

GeoContext::GeoContext(float strHeight, Blobs& segms, int fromIdx, int endIdx) : hasPrev(false), strHeight(strHeight) {
	generateUCGContext(segms, fromIdx, endIdx, curUcg);
}

GeoContext::GeoContext(float strHeight, Blobs& segms, int fromIdx, int endIdx, GeoContext& prev) : prevUcg(prev.curUcg), hasPrev(true), strHeight(strHeight) {
	generateUCGContext(segms, fromIdx, endIdx, curUcg);
	box2Segms = boundingRect(prevUcg.box, curUcg.box);
}

GeoContext::GeoContext(float strHeight, UCGContext& curUcg, UCGContext& prevUcg) : strHeight(strHeight), curUcg(curUcg), prevUcg(prevUcg), hasPrev(true) {
	box2Segms = boundingRect(prevUcg.box, curUcg.box);
}

void GeoContext::setPrevContext(GeoContext &ctx) {
	hasPrev = true;
	prevUcg = ctx.curUcg;
	box2Segms = boundingRect(prevUcg.box, curUcg.box);
}

float GeoContext::getStrHeight() {
	return strHeight;
}

void GeoContext::getUCGVector(vec_t& output) {
	output.clear();
	getUCGVector(curUcg, output);
}

void GeoContext::getUCGVector(UCGContext& ucgCtx, vec_t& output) {
	output.push_back(ucgCtx.box.height / strHeight);
	output.push_back(ucgCtx.box.width / strHeight);
	output.push_back(ucgCtx.innerGap / strHeight);
	output.push_back((ucgCtx.gravityCenter.x - ucgCtx.box.x) / strHeight);
	output.push_back((ucgCtx.gravityCenter.y - ucgCtx.box.y) / strHeight);
	output.push_back(std::log(ucgCtx.box.height / (float) ucgCtx.box.width));
	output.push_back(std::sqrt(ucgCtx.box.area()) / strHeight);
	output.push_back(std::sqrt(ucgCtx.box.width * ucgCtx.box.width + ucgCtx.box.height * ucgCtx.box.height) / strHeight);
	output.push_back(std::abs(ucgCtx.geoCenter.x - ucgCtx.gravityCenter.x) / strHeight);
	output.push_back(std::abs(ucgCtx.geoCenter.y - ucgCtx.gravityCenter.y) / strHeight);
	output.push_back(std::abs(ucgCtx.gravityCenter.y - ucgCtx.gravityCenterLine.y) / strHeight);
	output.push_back(std::abs(ucgCtx.geoCenter.y - ucgCtx.geoCenterLine.y) / strHeight);
	output.push_back(std::abs(ucgCtx.box.y - ucgCtx.geoCenterLine.y) / strHeight);
	output.push_back(std::abs(ucgCtx.box.y + ucgCtx.box.height - ucgCtx.geoCenterLine.y) / strHeight);
	output.push_back(ucgCtx.meanProjectProfile[0] / strHeight);
	output.push_back(ucgCtx.meanProjectProfile[1] / strHeight);
	output.push_back(ucgCtx.meanOutProfile[0] / strHeight);
	output.push_back(ucgCtx.deviOutProfile[0] / strHeight);
	output.push_back(ucgCtx.meanOutProfile[1] / strHeight);
	output.push_back(ucgCtx.deviOutProfile[1] / strHeight);
	output.push_back(ucgCtx.meanOutProfile[2] / strHeight);
	output.push_back(ucgCtx.deviOutProfile[2] / strHeight);
	output.push_back(ucgCtx.meanOutProfile[3] / strHeight);
	output.push_back(ucgCtx.deviOutProfile[3] / strHeight);
}

void GeoContext::getUIGVector(vec_t& output) {
	output.clear();
	output.push_back(curUcg.box.height / strHeight);
	output.push_back(curUcg.box.width / strHeight);
	output.push_back(curUcg.innerGap / strHeight);
	output.push_back((curUcg.gravityCenter.x - curUcg.box.x) / strHeight);
	output.push_back((curUcg.gravityCenter.y - curUcg.box.y) / strHeight);
	output.push_back(std::log(curUcg.box.height / (float) curUcg.box.width));
	output.push_back(std::sqrt(curUcg.box.area()) / strHeight);
	output.push_back(std::sqrt(curUcg.box.width * curUcg.box.width + curUcg.box.height * curUcg.box.height) / strHeight);

	output.push_back(std::abs(curUcg.geoCenter.x - curUcg.gravityCenter.x) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.y - curUcg.gravityCenter.y) / strHeight);
	output.push_back(curUcg.meanProjectProfile[0] / strHeight);
	output.push_back(curUcg.meanProjectProfile[1] / strHeight);
}



void GeoContext::getBCGVector(vec_t& output) {
	output.clear();
	if (!hasPrev) {
		return;
	}
	output.push_back(std::abs(curUcg.box.y - prevUcg.box.y) / strHeight);
	output.push_back(std::abs(curUcg.box.y + curUcg.box.height - prevUcg.box.y - prevUcg.box.height) / strHeight);
	output.push_back(std::abs(curUcg.box.y - prevUcg.box.y - prevUcg.box.height) / strHeight);
	output.push_back(std::abs(curUcg.box.y + curUcg.box.height - prevUcg.box.y) / strHeight);
	output.push_back(std::abs(curUcg.box.x - prevUcg.box.x) / strHeight);
	output.push_back(std::abs(curUcg.box.x + curUcg.box.width - prevUcg.box.x - prevUcg.box.width) / strHeight);
	output.push_back(std::abs(curUcg.gravityCenter.x - prevUcg.gravityCenter.x) / strHeight);
	output.push_back(std::abs(curUcg.gravityCenter.y - prevUcg.gravityCenter.y) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.x - prevUcg.geoCenter.x) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.y - prevUcg.geoCenter.y) / strHeight);
	output.push_back(box2Segms.height / strHeight);
	output.push_back(box2Segms.width / strHeight);
	output.push_back((curUcg.box.x - prevUcg.box.x - prevUcg.box.width) / strHeight);
	output.push_back(curUcg.box.height /(float) prevUcg.box.height);
	output.push_back(curUcg.box.width / (float) prevUcg.box.width);
	output.push_back(std::sqrt(commonArea(curUcg.box, prevUcg.box)) / strHeight);
	output.push_back((curUcg.meanOutProfile[0] - prevUcg.meanOutProfile[0]) / strHeight);
	output.push_back((curUcg.meanOutProfile[1] - prevUcg.meanOutProfile[1]) / strHeight);
	output.push_back((curUcg.meanOutProfile[2] - prevUcg.meanOutProfile[2]) / strHeight);
	output.push_back((curUcg.meanOutProfile[3] - prevUcg.meanOutProfile[3]) / strHeight);
	output.push_back((curUcg.deviOutProfile[0] - prevUcg.deviOutProfile[0]) / strHeight);
	output.push_back((curUcg.deviOutProfile[1] - prevUcg.deviOutProfile[1]) / strHeight);
	output.push_back((curUcg.deviOutProfile[2] - prevUcg.deviOutProfile[2]) / strHeight);
	output.push_back((curUcg.deviOutProfile[3] - prevUcg.deviOutProfile[3]) / strHeight);
	getUCGVector(prevUcg, output);
	getUCGVector(curUcg, output);
}

void GeoContext::getBIGVector(vec_t& output) {
	output.clear();
	if (!hasPrev) {
		return;
	}
	output.push_back(std::abs(curUcg.box.y - prevUcg.box.y) / strHeight);
	output.push_back(std::abs(curUcg.box.y + curUcg.box.height - prevUcg.box.y - prevUcg.box.height) / strHeight);
	output.push_back(std::abs(curUcg.box.y - prevUcg.box.y - prevUcg.box.height) / strHeight);
	output.push_back(std::abs(curUcg.box.y + curUcg.box.height - prevUcg.box.y) / strHeight);
	output.push_back(std::abs(curUcg.box.x - prevUcg.box.x) / strHeight);
	output.push_back(std::abs(curUcg.box.x + curUcg.box.width - prevUcg.box.x - prevUcg.box.width) / strHeight);
	output.push_back(std::abs(curUcg.gravityCenter.x - prevUcg.gravityCenter.x) / strHeight);
	output.push_back(std::abs(curUcg.gravityCenter.y - prevUcg.gravityCenter.y) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.x - prevUcg.geoCenter.x) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.y - prevUcg.geoCenter.y) / strHeight);
	output.push_back(box2Segms.height / strHeight);
	output.push_back(box2Segms.width / strHeight);
	output.push_back((curUcg.box.x - prevUcg.box.x - prevUcg.box.width) / strHeight);
}

float NumberModel::getScore(std::vector<label_t> labels, label_t l) {
	//manual score implementation
	if (labels.empty() && (l >= 10 || l == 0)) {
		return -2.0;
	}
	if (l < 10) {
		return 0;
	}
	if (std::find(labels.begin(), labels.end(), 10) != labels.end()) {
		return -1.0;
	}
	auto r = std::find(labels.rbegin(), labels.rend(), 11);
	if (r != labels.rend() && std::distance(r.base(), labels.end()) != 4) {
		return -1.0;
	}
	return 0;
}

float NumberModel::getFinalScore(std::vector<label_t> labels, label_t l) {
	float score = getScore(labels, l);
	if (l == 10 || l == 11) {
		return score - 1;
	}
	//analytics period point
	auto iter = std::find(labels.begin(), labels.end(), 10);
	auto end = labels.end();
	if (iter != end) {
		return iter + 2 != end ? score - 1 : score + 1;
	}
	return score;
}

} /* namespace icr */
