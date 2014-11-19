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

GeoContextModel::GeoContextModel() {
	uigModel = make_mlp<mse, gradient_descent, tanh_activation>({ 32 * 32, 300, 2 });
	bigModel = make_mlp<mse, gradient_descent, tanh_activation>({ 32 * 32, 300, 2 });
	ucgModel = make_mlp<mse, gradient_descent, tanh_activation>({ 32 * 32, 300, 6 });
	bcgModel = make_mlp<mse, gradient_descent, tanh_activation>({ 32 * 32, 300, 6 });
}

GeoContextModel::~GeoContextModel() {
	// TODO Auto-generated destructor stub
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

float GeoContext::getStrHeight() {
	return strHeight;
}

void GeoContext::getUCGVector(vec_t& output) {
	output.clear();
	output.push_back(curUcg.box.height / strHeight);
	output.push_back(curUcg.box.width / strHeight);
	output.push_back((curUcg.gravityCenter.x - curUcg.box.x) / strHeight);
	output.push_back((curUcg.gravityCenter.y - curUcg.box.y) / strHeight);
	output.push_back(std::log(curUcg.box.height / (float) curUcg.box.width));
	output.push_back(std::sqrt(curUcg.box.area()) / strHeight);
	output.push_back(std::sqrt(curUcg.box.width * curUcg.box.width + curUcg.box.height * curUcg.box.height) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.x - curUcg.gravityCenter.x) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.y - curUcg.gravityCenter.y) / strHeight);
	output.push_back(std::abs(curUcg.gravityCenter.y - curUcg.gravityCenterLine.y) / strHeight);
	output.push_back(std::abs(curUcg.geoCenter.y - curUcg.geoCenterLine.y) / strHeight);
	output.push_back(std::abs(curUcg.box.y - curUcg.geoCenterLine.y) / strHeight);
	output.push_back(std::abs(curUcg.box.y + curUcg.box.height - curUcg.geoCenterLine.y) / strHeight);
	output.push_back(curUcg.meanProjectProfile[0] / strHeight);
	output.push_back(curUcg.meanProjectProfile[1] / strHeight);
	output.push_back(curUcg.meanOutProfile[0] / strHeight);
	output.push_back(curUcg.deviOutProfile[0] / strHeight);
	output.push_back(curUcg.meanOutProfile[1] / strHeight);
	output.push_back(curUcg.deviOutProfile[1] / strHeight);
	output.push_back(curUcg.meanOutProfile[2] / strHeight);
	output.push_back(curUcg.deviOutProfile[2] / strHeight);
	output.push_back(curUcg.meanOutProfile[3] / strHeight);
	output.push_back(curUcg.deviOutProfile[3] / strHeight);
}

void GeoContext::getUIGVector(vec_t& output) {
	output.clear();
	output.push_back(curUcg.box.height / strHeight);
	output.push_back(curUcg.box.width / strHeight);
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
	if (hasPrev) {
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
}

void GeoContext::getBIGVector(vec_t& output) {
	output.clear();
	if (hasPrev) {
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
	if (l == 10 && std::find(labels.begin(), labels.end(), l) != labels.end()) {
		return -1.0;
	}
	auto begin = labels.size() >= 3 ? labels.end() - 3 : labels.end() - labels.size();
	if (l == 11 && std::find(begin, labels.end(), l) != labels.end()) {
		return -1.0;
	}
	return 0;
}

float NumberModel::getFinalScore(std::vector<label_t> labels, label_t l) {
	float score = getScore(labels, l);
	//analytics period point
	auto iter = std::find(labels.begin(), labels.end(), 10);
	auto end = labels.end();
	if (iter + 2 == end) {
		return score + 0.5;
	}
	return score;
}

} /* namespace icr */
