/*
 * NumberRecognizer.cpp
 *
 *  Created on: Oct 29, 2014
 *      Author: thienlong
 */

#include <GeoContextModel.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "NumberRecognizer.h"
#include "opencv2/core/core.hpp"
#include "preprocessor.h"
#include "util/misc.h"
#include <list>


#include <boost/filesystem.hpp>
using boost::filesystem::path;
extern std::string chqName;
extern digit_recognizer recognizer;

namespace icr {
using namespace cv;
label_t getBIGLabel(label_t prev, label_t cur);
NumberRecognizer::NumberRecognizer(Blobs &blobs) {
	estHeightVertCenter(blobs, strHeight, middleLine);
	genOverSegm(blobs);
	//TODO debug
	cv::Mat car = drawBlobs(blobs);
	line(car, Point(0, middleLine), Point(car.cols -1, middleLine), Scalar(255, 0, 0));
	cv::imshow("hwimg", car);
	Mat img = drawBlobs(segms);
	namedWindow("segms", WINDOW_NORMAL);
	imshow("segms", img);
//	waitKey();
}



bool NumberRecognizer::isTouching(Blob& blob) {
	Rect r = blob.boundingRect();
	return (r.height > strHeight / 2 && r.width > strHeight * 0.8)
			|| (r.width /(float) r.height > 1.2);
}

bool isConcavePoint(Point prev, Point p, Point next) {
	float projectMidPoint = (prev.y - next.y) / ((float) (prev.x - next.x)) * (p.x - prev.x) + prev.y;
	return p.y > projectMidPoint;
}

void findConcavePoints(vector<Point>& approx, vector<Point>& cps) {
	for (size_t i = 1; i < approx.size() - 1; ++i) {
		if (isConcavePoint(approx[i -1], approx[i], approx[i + 1])) {
			cps.push_back(approx[i]);
		}
	}
}

void rollPath(const Mat& img, vector<Point>& approx) {
	//find btmLeft btmRight
	int btmLeft = 0;
	int btmRight = 0;
	for (size_t i = 0; i < approx.size(); ++i) {
		Point p = approx[i];
		if (p.x < 3 && p.y >= img.rows - 2) {
			btmLeft = i;
		} else if (p.x >= img.cols - 2 && p.y >= img.rows - 2) {
			btmRight = i;
		}
	}
	if (btmLeft == 0) {
		if (btmRight == approx.size() - 1)
			return;
	}
	vector<Point> result;
	if (btmLeft < btmRight) {
		result.insert(result.end(), approx.rbegin() + approx.size() - btmLeft - 1, approx.rend());
		result.insert(result.end(), approx.rbegin(), approx.rbegin() + approx.size() - btmRight);
	} else {
		result.insert(result.end(), approx.begin() + btmLeft, approx.end());
		result.insert(result.end(), approx.begin(), approx.begin() + btmRight);
	}
	approx = result;
}

#define MAX_SHIFT 2
Point shift2BG(const Mat& project, Point p) {
	p.y = p.y - 1;
	if (project.at<uchar>(p) == 0) {
		return p;
	}
	for (int i = 1; i <= MAX_SHIFT; ++i) {
		Point newP(p.x + i, p.y - i);
		if (project.at<uchar>(newP) == 0) {
			return newP;
		}
		newP.x = p.x - i;
		if (project.at<uchar>(newP) == 0) {
			return newP;
		}
	}
	return p;
}

void genUpperCuts(Mat& img, std::vector<int>& upperCuts) {
	Mat project = projectTop(img);
    //Extract the contours so that
    vector<vector<Point> > contours;
    findContours( project, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
    	return;
    }
	vector<Point> approx;
	cv::approxPolyDP(cv::Mat(contours[0]), approx, cv::arcLength(cv::Mat(contours[0]), true)*0.02, true);
	rollPath(project, approx);
	//debug
//	Mat drawing(project.rows, project.cols, CV_8UC3);
//	for (size_t i = 0; i < approx.size() - 1; ++i) {
//		if (i == 0) {
//			line(drawing, approx[i], approx[i + 1], Scalar( 255,0, 0), 1);
//		} else {
//			line(drawing, approx[i], approx[i + 1], Scalar(0, 255, 0), 1);
//		}
//	}
	//analytic feature points
	vector<Point> cps;
	findConcavePoints(approx, cps);
	//correct point (shift to background point)
	for (Point p : cps) {
		p = shift2BG(project, p);
		upperCuts.push_back(p.x);
	}
}

void genLowerCuts(Mat& img, std::vector<int>& lowerCuts) {
	Mat flipedImg;
	flip(img, flipedImg, 0);
	genUpperCuts(flipedImg, lowerCuts);
}

int mergeVertCut(const Mat& img, int x1, int x2) {
	int cntCut1 = 0;
	int cntCut2 = 0;
	for (int r = 0; r < img.rows; ++r) {
		if (img.at<uchar>(r, x1) > 0) {
			++cntCut1;
		}
		if (img.at<uchar>(r, x2) > 0) {
			++cntCut2;
		}
	}
	if (cntCut1 > cntCut2) {
		return x2;
	} else if (cntCut1 < cntCut2) {
		return x1;
	} else {
		return (x1 + x2) / 2;
	}
}

void NumberRecognizer::genVertCuts(Blob& blob, std::vector<int>& cuts) {
	Rect box = blob.boundingRect();
	Mat img = cropBlob(blob);
	std::vector<int> upperCuts;
	std::vector<int> lowerCuts;
	genUpperCuts(img, upperCuts);
	genLowerCuts(img, lowerCuts);
	float mergeThres = strHeight / 8.0f;
	for (int x : upperCuts) {
		int nearest = 0;
		for (size_t i = 0; i < lowerCuts.size(); ++i) {
			if (std::abs(x - lowerCuts[i]) < mergeThres) {
				nearest = lowerCuts[i];
				lowerCuts.erase(lowerCuts.begin() + i);
				break;
			}
		}
		x = (nearest == 0) ? x + box.x : mergeVertCut(img, x, nearest) + box.x;
		cuts.push_back(x);
	}
	//insert lowerCuts
	for (int x : lowerCuts) {
		cuts.push_back(x + box.x);
	}
}

Blob* getSubBlob(Blob& blob, int from, int end) {
	Blob* rs = new Blob();
	for (Point p : blob.points) {
		if (p.x >= from && p.x < end) {
			rs->points.push_back(p);
		}
	}
	return rs;
}

void NumberRecognizer::segment(Blobs& segms, Blob& blob) {
	std::vector<int> cuts;
	Rect box = blob.boundingRect();
	cuts.push_back(box.x);
	genVertCuts(blob, cuts);
	if (cuts.size() == 1) {
		segms.add(new Blob(blob));
		return;
	}
	std::sort(cuts.begin(), cuts.end());
	cuts.push_back(box.x + box.width);
	for (size_t i = 0, n = cuts.size() - 1; i < n; ++i) {
		Blob* subBlob = getSubBlob(blob, cuts[i], cuts[i + 1]);
		if (subBlob->points.empty()) {
			delete subBlob;
		} else {
			segms.add(subBlob);
		}
	}
}

void NumberRecognizer::genOverSegm(Blobs &blobs) {
	Blob* blob = NULL;
	for (int i = 0; i < blobs.size(); ++i) {
		blob = blobs[i];
		if (!isTouching(*blob)) {
			segms.add(new Blob(*blob));
			continue;
		}
		segment(segms, *blob);
	}
}

class Path {
	friend class NumberRecognizer;
	std::vector<int> path;
	std::vector<label_t> labels;
	float score;
public:
	Path() : score(0) {

	}

	float getScore() {
		return score / (path.size() - 1);
	}

	void init(int node) {
		path.push_back(node);
	}

	void add(int node, label_t l, float score) {
		path.push_back(node);
		this->score += score;
		labels.push_back(l);
	}

	const std::vector<int>& get() const {
		return path;
	}

	std::string string() {
		stringstream ss;
		for (label_t l : labels) {
			if (l < 10) {
				ss << std::to_string(l);
			} else if (l == 10){
				ss << ".";
			} else if (l == 11) {
				ss <<",";
			}
		}
		return ss.str();
	}
};

class Beam {
	std::list<Path> paths;
	int with;
public:
	Beam(int with) : with(with) {}
	bool empty() {
		return paths.empty();
	}

	std::vector<Path> popLowerestNodes() {
		std::vector<Path> rs;
		if (paths.empty()) {
			return rs;
		}
		auto it = paths.begin();
		rs.push_back(*it);
		int lowestNode =it->get().back();
		for (it = paths.erase(it); it != paths.end(); it = paths.erase(it)) {
			if (it->get().back() == lowestNode) {
				rs.push_back(*it);
			} else {
				break;
			}
		}
		return rs;
	}

	void add(std::vector<Path> paths) {
		for (Path& p : paths) {
			add(p);
		}
	}

	void add(const Path& p) {
		int endNode = p.get().back();
		auto it = paths.begin();
		int counter = 0;
		for (; it != paths.end() && it->get().back() <= endNode; ++it) {
			if (it->get().back() == endNode) {
				++counter;
			}
		}
		paths.insert(it, p);
		//remove worst p if exceeded
		if (counter == with) {
			--it;
			auto worstIt = it;
			float worstScore = 999999;
			for (; counter >= 0; --it, --counter) {
				float score = it->getScore();
				if (worstScore > score) {
					worstScore = score;
					worstIt = it;
				}
			}
			paths.erase(worstIt);
		}
	}
};

Path bestPath(std::vector<Path>& paths) {
	auto bestIt = paths.begin();
	float bestCost = -999999;
	for (auto it = paths.begin(), end = paths.end(); it != end; ++it) {
		float score = it->getScore();
		if (bestCost < score) {
			bestCost = score;
			bestIt = it;
		}
	}
	return *bestIt;
}

bool NumberRecognizer::isCandidatePattern(int from, int end) {
	Rect rect = segms.boundingRect(from, end);
	//verify condition
	bool valid = (rect.height >= strHeight / 3.0f) && (rect.width < 2.4 * strHeight) && (rect.width / (float) rect.height < 3);
	if (valid) {
		return true;
	}
	return true;
	Blob* blob = segms.newBlob(from, end);
	valid = isPeriod(rect, middleLine, *blob);
	delete blob;
	return valid;
}

digit_recognizer::result NumberRecognizer::recognizeBlob(Blobs& segms, int start, int end) {
	Mat src = segms.cropBlobs(start, end);
	auto digitMat = makeDigitMat(src);
	vec_t in;
	matToVect(digitMat, in);
	auto rs = recognizer.predict(in);
	//TODO DEBUG generate sample data
//	digitMat = 255 - digitMat;
//	path filePath = "temp/" + chqName;
//	if (!boost::filesystem::exists(filePath)) {
//		boost::filesystem::create_directory(filePath);
//	}
//	imwrite(filePath.string() + "/-" + std::to_string(start) + "_" + std::to_string(end) + ".png", digitMat);
	return rs;
}

std::pair<label_t, float> NumberRecognizer::predictNode(std::vector<label_t>& labels, digit_recognizer::result rs, GeoContext& gc, bool lastNode) {
	//update NumberModel score
	for (int i = 0; i < rs.out.size(); ++i) {
		rs.out[i] =1.1 * rs.out[i] +  0.4f * (lastNode ? nm.getFinalScore(labels, i) : nm.getScore(labels, i));
	}
	std::pair<label_t, float> labelScore;
//	labelScore.first = rs.label(true);
//	labelScore.second = rs.confidence();
//	return labelScore;
	vec_t uigScore, ucgScore;
	gcm.predictUnary(gc, ucgScore, uigScore);
	for (int i = 0; i < rs.out.size(); ++i) {
		int uigLabel = (i < 10 ? 1 : (i == 10 ? 2 : 3));
		rs.out[i] += 0.3 * ucgScore[GeoContextModel::CLASS_MAP[i]] + uigScore[uigLabel];
	}
	//binary class
	if (!labels.empty()) {
		vec_t bigScore, bcgScore;
		gcm.predictBinary(gc, bcgScore, bigScore);
		label_t prevLabel = labels.back();
		int bcgIdx = GeoContextModel::CLASS_MAP[prevLabel] * 8;
		for (int i = 0; i < rs.out.size(); ++i) {
			rs.out[i] += 0.3 * bcgScore[GeoContextModel::CLASS_MAP[i] + bcgIdx];
			label_t bigLabel = getBIGLabel(prevLabel, i);
			rs.out[i] += 0.1 * bigScore[bigLabel];
		}
	}
	labelScore.first = rs.label(true);
	labelScore.second = rs.confidence();
	return labelScore;
}

NumberModel NumberRecognizer::nm;
GeoContextModel NumberRecognizer::gcm;
void NumberRecognizer::expandPath(Beam& beam, const std::vector<Path>& paths, int node) {
	if (paths.empty()) {
		return;
	}
	int endNode = paths[0].get().back();
	bool lastNode = node == segms.size();
	try {
		auto result = recognizeBlob(segms, endNode, node);
		GeoContext gc(strHeight, segms, endNode, node);
		for (Path p : paths) {
			Path newP = p;
			int pathSize = newP.path.size();
			if (pathSize >= 2) {
				GeoContext prevGC(strHeight, segms, newP.path[pathSize - 2], newP.path[pathSize - 1]);
				gc.setPrevContext(prevGC);
			}
			std::pair<label_t, float> rs = predictNode(newP.labels, result, gc, lastNode);
			newP.add(node, rs.first, rs.second);
			beam.add(newP);
		}
	} catch (cv::Exception& e) {
	}
}

std::string NumberRecognizer::predict() {
	//implement beam search
	Beam beam(10);
	Path start;
	start.init(0);
	beam.add(start);
	while (!beam.empty()) {
		std::vector<Path> paths = beam.popLowerestNodes();
		int endNode = paths[0].get().back();
		if (endNode == segms.size()) {
			return bestPath(paths).string();
		}
		//expand node
		int end = endNode + 4;
		if (end > segms.size()) {
			end = segms.size();
		}
		//TODO Debug
		std::string bestP = bestPath(paths).string();
		if (bestP.compare("339") == 0) {
			int stub = 0;
		}
		std::cout << "DEBUG: " << bestP << std::endl;
		for (int i = endNode + 1; i <= end; ++i) {
			if (isCandidatePattern(endNode, i)) {
				expandPath(beam, paths, i);
			}
		}

	}
	return "";
}

int findLabelSegment(std::vector<Segment>& segms, int start, int end) {
	for (Segment s : segms) {
		if (s.start == start && s.end == end) {
			return s.label;
		}
	}
	return -1;
}

label_t getBIGLabel(label_t prev, label_t cur) {
	if (prev < 10 && cur < 10) {
		return 1;
	} else if (prev < 10 && cur == 10) {
		return 2;
	} else if (prev < 10 && cur == 11) {
		return 3;
	} else if (prev == 10 && cur < 10) {
		return 4;
	} else if (prev == 11 && cur < 10) {
		return 5;
	} else {
		return 0;
	}
}

void NumberRecognizer::genTrainData(std::vector<Segment>& segmsCnf, int dataType, std::vector<vec_t>& inputs, std::vector<label_t>& labels) {
	vec_t in;
	if (dataType == ICR_UCG) {
		for (Segment segm : segmsCnf) {
			GeoContext gc(strHeight, segms, segm.start, segm.end);
			gc.getUCGVector(in);
			inputs.push_back(in);
			labels.push_back(segm.label);
		}
	} else if (dataType == ICR_UIG) {
		int numSegms = segms.size();
		for (int start = 0; start < segms.size() - 1; ++start) {
			for (int end = start + 1, n = std::min(start + 4, numSegms); end <= n; ++end) {
				if (isCandidatePattern(start, end)) {
					GeoContext gc(strHeight, segms, start, end);
					gc.getUIGVector(in);
					inputs.push_back(in);
					int label = findLabelSegment(segmsCnf, start, end);
					if (label == -1) {
						label = 0;
					} else if (label < 10) {
						label = 1;
					} else if (label == 10) {
						label = 2;
					} else {
						label = 3;
					}
					labels.push_back(label);
				}
			}
		}
	} else if (dataType == ICR_BCG && !segmsCnf.empty()) {
		Segment segm = segmsCnf[0];
		GeoContext prev(strHeight, segms, segm.start, segm.end);
		for (int i = 1; i < segmsCnf.size(); ++i) {
			label_t label = GeoContextModel::CLASS_MAP[segm.label] * 8;
			segm = segmsCnf[i];
			label += GeoContextModel::CLASS_MAP[segm.label];
			GeoContext gc(strHeight, segms, segm.start, segm.end, prev);
			gc.getBCGVector(in);
			inputs.push_back(in);
			labels.push_back(label);
			prev = gc;
		}
	} else if (dataType == ICR_BIG && !segmsCnf.empty()) {
		Segment segm = segmsCnf[0];
		GeoContext prev(strHeight, segms, segm.start, segm.end);
		for (int i = 1; i < segmsCnf.size(); ++i) {
			label_t label = getBIGLabel(segm.label, segmsCnf[i].label);
			segm = segmsCnf[i];
			GeoContext gc(strHeight, segms, segm.start, segm.end, prev);
			gc.getBIGVector(in);
			inputs.push_back(in);
			labels.push_back(label);
			prev = gc;
		}
	} else if (dataType == ICR_BIG_IGNORE && !segmsCnf.empty()) {
		for (int i = 0; i < segmsCnf.size(); i += 2) {
			Segment segm = segmsCnf[i];
			GeoContext prev(strHeight, segms, segm.start, segm.end);
			segm = segmsCnf[i + 1];
			GeoContext gc(strHeight, segms, segm.start, segm.end, prev);
			gc.getBIGVector(in);
			inputs.push_back(in);
			labels.push_back(0);
		}
	}
}

NumberRecognizer::~NumberRecognizer() {
	// TODO Auto-generated destructor stub
}

} /* namespace icr */
