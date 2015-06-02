#include "preprocessor.h"
#include <opencv2/imgproc/imgproc.hpp>

#include "util/misc.h"
Blob::Blob() : needNewRect(true), innerGap(0) {
}

cv::Rect Blob::boundingRect() {
	if (needNewRect) {
		bound = cv::boundingRect(points);
		needNewRect = false;
	}
	return bound;
}
void Blob::add(const Blob& other) {
	needNewRect = true;
	if (!points.empty() && !other.points.empty()) {
		innerGap += distanceBlobs(*this, other);
	}
	points.insert(points.end(), other.points.begin(), other.points.end());
}
void Blob::add(const cv::Point2i& point) {
	needNewRect = true;
	points.push_back(point);
}

float Blob::getInnerGap() const {
	return innerGap;
}

void Blob::setModify(bool flag) {
	needNewRect = flag;
}

void Blob::move(int x, int y) {
	for (auto& p : points) {
		p.x += x;
		p.y += y;
	}
	if (!needNewRect) {
		bound.x += x;
		bound.y += y;
	}
}

cv::Point2f Blob::getMassCenter() {
	int size = points.size();
	cv::Point2f centroid;
	if (size > 0) {
		float sumX = 0;
		float sumY = 0;
	    for (auto point : points){
	        sumX += point.x;
	        sumY += point.y;
	    }
		centroid.x = sumX/size;
		centroid.y = sumY/size;
	}
	return centroid;
}

Blobs::Blobs() {}

Blobs::Blobs(Blobs& blobs) {
	blobs.clone(*this);
}

size_t Blobs::size() const {
	return blobs.size();
}

void Blobs::erase(int index) {
	Blob* b = blobs[index];
	delete b;
	blobs.erase(blobs.begin() + index);
}

Blob* Blobs::detach(int index) {
	Blob* b = blobs[index];
	blobs.erase(blobs.begin() + index);
	return b;
}

Blob* Blobs::operator[] (int index) const {
	return blobs[index];
}

void Blobs::sort(bool (*compFunct)(Blob* blob1, Blob* blob2)) {
	std::sort(blobs.begin(), blobs.end(), compFunct);
}

void Blobs::add(Blob* blob) {
	blobs.push_back(blob);
}

void Blobs::clone(Blobs& other) const {
	other.clear();
	for (Blob* b : blobs) {
		other.add(new Blob(*b));
	}
}

void Blobs::clear() {
	for (Blob* blob : blobs) {
		delete blob;
	}
	blobs.clear();
}
Blobs::~Blobs() {
	clear();
}

void findBlobs(const cv::Mat &binary, Blobs &blobs) {
	blobs.clear();
	// Fill the label_image with the blobs
	// 0  - background
	// 1  - unlabelled foreground
	// 2+ - labelled foreground

	cv::Mat label_image;
	binary.convertTo(label_image, CV_32SC1);

	int label_count = 2; // starts at 2 because 0,1 are used already

	for (int y = 0; y < label_image.rows; y++) {
		int *row = (int*)label_image.ptr(y);
		for (int x = 0; x < label_image.cols; x++) {
			if (row[x] != 1) {
				continue;
			}

			cv::Rect rect;
			cv::floodFill(label_image, cv::Point(x, y), label_count, &rect, 0, 0, 4);

			Blob *blob = new Blob();
			int maxX = 0, minX = binary.cols, maxY = 0, minY = binary.rows;
			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if (row2[j] != label_count) {
						continue;
					}
					blob->add(cv::Point2i(j, i));
				}
			}
			blobs.add(blob);
			label_count++;
		}
	}
}

#define MAX_INT 0xfffffff

cv::Rect Blobs::boundingRect() const {
	int minX = MAX_INT;
	int minY = MAX_INT;
	int maxX = 0;
	int maxY = 0;
	for (int i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		auto rect = b->boundingRect();
		if (minX > rect.x) {
			minX = rect.x;
		}
		if (minY > rect.y) {
			minY = rect.y;
		}
		if (maxX < rect.x + rect.width) {
			maxX = rect.x + rect.width;
		}
		if (maxY < rect.y + rect.height) {
			maxY = rect.y + rect.height;
		}
	}
	return maxX == 0 ? cv::Rect() : cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

cv::Rect Blobs::boundingRect(int from, int end) const {
	cv::Rect rect = blobs[from]->boundingRect();
	for (int i = from + 1; i < end; ++i) {
		cv::Rect other = blobs[i]->boundingRect();
		int maxX = std::max(rect.x + rect.width, other.x + other.width);
		int maxY = std::max(rect.y + rect.height, other.y + other.height);
		rect.x = std::min(rect.x, other.x);
		rect.y = std::min(rect.y, other.y);
		rect.width = maxX - rect.x;
		rect.height = maxY - rect.y;
	}
	return rect;
}

void Blobs::move(int x, int y) {
	for (size_t i = 0; i < blobs.size(); ++i) {
		blobs[i]->move(x, y);
	}
}

Blob* Blobs::newBlob(int from, int end) const {
	Blob* b = new Blob();
	for (int i = from; i < end; ++i) {
		b->add(*(blobs[i]));
	}
	return b;
}

cv::Mat Blobs::cropBlobs(int from, int end) {
	cv::Rect bound = boundingRect(from, end);
	cv::Mat rs = cv::Mat::zeros(bound.height, bound.width, CV_8UC1);
	for (int i = from; i < end; ++i) {
		Blob& blob = *(blobs[i]);
		for (size_t j = 0; j < blob.points.size(); j++) {
			int x = blob.points[j].x - bound.x;
			int y = blob.points[j].y - bound.y;
			rs.at<uchar>(y, x) = 255;
		}
	}
	return rs;
}

cv::Point2f Blobs::getMassCenter() const {
	cv::Point2f centroid;
	float sumX = 0;
	float sumY = 0;
	int size = 0;
	for (int i = 0; i < blobs.size(); ++i) {
		Blob* blob = blobs[i];
		size += blob->points.size();
	    for (auto point : blob->points){
	        sumX += point.x;
	        sumY += point.y;
	    }
	}
	centroid.x = sumX/size;
	centroid.y = sumY/size;
	return centroid;
}

cv::Rect boundingRect(Blob& b1, Blob& b2) {
	cv::Rect r1 = b1.boundingRect();
	cv::Rect r2 = b2.boundingRect();
	int minX = std::min(r1.x, r2.x);
	int minY = std::min(r1.y, r2.y);
	int maxX = std::max(r1.x + r1.width, r2.x + r2.width);
	int maxY = std::max(r1.y + r1.height, r2.y + r2.height);
	return maxX == 0 ? cv::Rect() : cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

cv::Rect boundingRect(cv::Rect r1, cv::Rect r2) {
	int minX = std::min(r1.x, r2.x);
	int minY = std::min(r1.y, r2.y);
	int maxX = std::max(r1.x + r1.width, r2.x + r2.width);
	int maxY = std::max(r1.y + r1.height, r2.y + r2.height);
	return maxX == 0 ? cv::Rect() : cv::Rect(minX, minY, maxX - minX, maxY - minY);
}

void estHeightVertCenter(Blobs& blobs, float& strHeight, float& middleLine) {
	if (blobs.size() == 1) {
		strHeight = blobs[0]->boundingRect().height;
		middleLine = strHeight / 2;
		return;
	}
	float heightScore = 0;
	int sumHeightW = 0;
	float middleLineScore = 0;
	int sumMLW = 0;
	for (int i = 0, n = blobs.size() - 1; i < n; ++i) {
		cv::Rect r1 = blobs[i]->boundingRect();
		cv::Rect r2 = blobs[i + 1]->boundingRect();
		cv::Rect compose = boundingRect(*blobs[i], *blobs[i + 1]);
		heightScore += compose.height * (r1.width + r2.width);
		sumHeightW += r1.width + r2.width;
		middleLineScore += (compose.y + compose.height / 2) * (r1.area() + r2.area());
		sumMLW += r1.area() + r2.area();
	}
	strHeight = heightScore / sumHeightW;
	middleLine = middleLineScore / sumMLW;
}

cv::Mat drawBlobs(const Blobs& blobs) {
	cv::Mat output;
	drawBlobs(blobs, output);
	return output;
}

void drawBlobs(const Blobs& blobs, cv::Mat& output) {
	cv::Rect rect = blobs.boundingRect();
	output = cv::Mat::zeros(rect.y + rect.height, rect.x + rect.width, CV_8UC3);
	// Randomy color the blobs
	for (size_t i = 0; i < blobs.size(); i++) {
		unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));

		for (size_t j = 0; j < blobs[i]->points.size(); j++) {
			int x = blobs[i]->points[j].x;
			int y = blobs[i]->points[j].y;

			output.at<cv::Vec3b>(y, x)[0] = b;
			output.at<cv::Vec3b>(y, x)[1] = g;
			output.at<cv::Vec3b>(y, x)[2] = r;
		}
	}
}

cv::Mat drawBinaryBlobs(const Blobs& blobs) {
	cv::Rect rect = blobs.boundingRect();
	cv::Mat output = cv::Mat::zeros(rect.y + rect.height, rect.x + rect.width, CV_8UC1);
	for (size_t i = 0; i < blobs.size(); i++) {
		for (auto p : blobs[i]->points) {
			output.at<uchar>(p) = 255;
		}
	}
	return output;
}

void drawBinaryBlobs(const Blobs& blobs, cv::Mat& output) {
	output = cv::Scalar::all(0);
	for (size_t i = 0; i < blobs.size(); i++) {
		for (auto p : blobs[i]->points) {
			output.at<uchar>(p) = 255;
		}
	}
}

class DisjointDigit {
public:
	bool operator() (Blob* blob1, Blob* blob2) {
		cv::Rect b1 = blob1->boundingRect();
		cv::Rect b2 = blob2->boundingRect();
		int maxX1 = b1.x + b1.width;
		int maxX2 = b2.x + b2.width;
		if (b1.x >= maxX2 || b2.x >= maxX1) {
			return false;
		}

		int min = std::max(b1.x, b2.x);
		int max = std::min(maxX1, maxX2);
		float overLap = (max - min);
		return (overLap / b1.width + overLap / b2.width) > 0.7 ? true : (b1.y + b1.height <= b2.y || b2.y + b2.height <= b1.y);
	}
};

class OverlapRejoin {
public:
	bool operator() (Blob* blob1, Blob* blob2) {
		cv::Rect b1 = blob1->boundingRect();
		cv::Rect b2 = blob2->boundingRect();
		int maxX1 = b1.x + b1.width;
		int maxX2 = b2.x + b2.width;
		if (b1.x >= maxX2 || b2.x >= maxX1) {
			return false;
		}
		int min = std::max(b1.x, b2.x);
		int max = std::min(maxX1, maxX2);
		float overLap = (max - min);
		int minBlobWidth = std::min(b1.width, b2.width);
		float dist = std::abs((b1.x + b1.width / 2) - (b2.x + b2.width / 2));
		int span = std::max(maxX1, maxX2) - std::min(b1.x, b2.x);
		float novlp = overLap / minBlobWidth - dist / span;
		return novlp > 0.55;
	}
};

int numLabel(std::vector<int>& labels, int label) {
	int counter = 0;
	for (size_t i = 0; i < labels.size(); ++i) {
		if (labels[i] == label) {
			++counter;
		}
	}
	return counter;
}

inline bool isSmallBlob(Blob& blob) {
	cv::Rect bound = blob.boundingRect();
	return bound.width <= 3 || bound.height <= 3;
}

void removeSmallBlobs(Blobs& blobs) {
	double perimaterSum = 0;
	double baseLineSum = 0;
	int i = 0;
	for (; i < blobs.size(); ++i) {
		cv::Rect bound = blobs[i]->boundingRect();
		perimaterSum += (bound.width + bound.height);
		baseLineSum += bound.y + bound.height;
	}
	double perimaterAverg = perimaterSum / blobs.size();
	double baseLineAverg = baseLineSum / blobs.size();
	i = 0;
	double rate = 0;
	while (i < blobs.size()) {
		cv::Rect bound = blobs[i]->boundingRect();
		rate = (bound.width + bound.height) / perimaterAverg;
		if (bound.width < 3 || bound.height <= 3 || rate < 0.2 || (rate < 0.5 && (bound.y + bound.height) < baseLineAverg * 0.9)) {
			blobs.erase(i);
			continue;
		}
		++i;
	}
}

void removeTailNoise(Blobs& blobs, float strHeight, float middleLine) {
	//remove small tail blobs
	int lastBlobIdx = blobs.size() - 1;
	for (; lastBlobIdx >= 0; --lastBlobIdx) {
		cv::Rect box = blobs[lastBlobIdx]->boundingRect();
		//(box.y <= middleLine && box.y + box.height > middleLine)
		if (box.height + box.width >= strHeight * 0.7 && box.height > strHeight * 0.3) {
			break;
		}
	}
	if (lastBlobIdx == -1) {
		return;
	}
	cv::Rect lastBlobRect = blobs[lastBlobIdx]->boundingRect();
	int x = lastBlobRect.x + lastBlobRect.width;
	for (int i = blobs.size() - 1; i > lastBlobIdx; --i) {
		cv::Rect box = blobs[i]->boundingRect();
		if (box.x - x > strHeight / 2) {
			blobs.erase(i);
		}
	}
	if (lastBlobRect.width / (float) lastBlobRect.height > 1) {
		return;
	}
	int second = lastBlobIdx - 1;
	for (; second >= 0; --second) {
		cv::Rect box = blobs[second]->boundingRect();
		//(box.y <= middleLine && box.y + box.height > middleLine)
		if (box.height + box.width >= strHeight * 0.7 && box.height > strHeight * 0.3) {
			break;
		}
	}
	if (second == -1) {
		return;
	}
	cv::Rect secRect = blobs[second]->boundingRect();
	if (lastBlobRect.x - (secRect.x + secRect.width) > strHeight * 1.5) {
		for (int i = blobs.size() - 1; i > second; --i) {
			blobs.erase(i);
		}
	}
}

void cleanNoises(Blobs& blobs) {
	Blobs temp(blobs);
	std::vector<bool> del(temp.size(), false);
	double perimaterSum = 0;
	double baseLineSum = 0;
	for (int i = 0; i < blobs.size(); ++i) {
		cv::Rect bound = blobs[i]->boundingRect();
		perimaterSum += (bound.width + bound.height);
		baseLineSum += bound.y + bound.height;
	}
	double perimaterAverg = perimaterSum / blobs.size();
	double baseLineAverg = baseLineSum / blobs.size();
	double rate = 0;
	for (int i = blobs.size() - 1; i >= 0; --i) {
		cv::Rect bound = blobs[i]->boundingRect();
		rate = (bound.width + bound.height) / perimaterAverg;
		if (bound.width < 3 || bound.height <= 3 || rate < 0.2 || (rate < 0.5 && (bound.y + bound.height) < baseLineAverg * 0.9)) {
			blobs.erase(i);
			del[i] = true;
		}
	}
	if (blobs.size() == 0) {
		return;
	}
	//add period
	float strHeight = 0;
	float middleLine = 0;
	estHeightVertCenter(blobs, strHeight, middleLine);
	cv::Rect searchArea = blobs.boundingRect();
	for (int i = blobs.size() - 1, counter = 0; i > 0; --i) {
		cv::Rect box = blobs[i]->boundingRect();
		if (box.y <= middleLine && box.y + box.height > middleLine) {
			++counter;
		}
		if (counter == 4) {
			searchArea = blobs.boundingRect(i, blobs.size());
			break;
		}
	}
	for (int i = temp.size() - 1; i >= 0 ; --i) {
		if (!del[i]) {
			continue;
		}
		cv::Rect box = temp[i]->boundingRect();
		if (intersect(box, searchArea)) {
			Blob* b = temp.detach(i);
			blobs.add(b);
		}
	}
	blobs.sort(sortByVertical);
	removeTailNoise(blobs, strHeight, middleLine);
}

void groupVertical(Blobs& blobs) {
	std::vector<int> labels;
	blobs.partition(labels, OverlapRejoin());
	//join group
	for (int label = 0; label < labels.size(); ++label) {
		//find first blob of label
		Blob *blob = NULL;
		int i = 0;
		for (; i < labels.size(); ++i) {
			if (labels[i] == label) {
				blob = blobs[i];
				break;
			}
		}
		if (blob == NULL) { //no blob (aka no label)
			break;
		}
		//filter 2 blob small blobs which overlap
		int numSameLabel = numLabel(labels, label);
		if (numSameLabel == 2 && isSmallBlob(*blob)) {
			blobs.erase(i);
			labels.erase(labels.begin() + i);
			continue;
		}
		++i;
		while (i < labels.size()) {
			if (labels[i] == label) {
				if (numSameLabel != 2 || !isSmallBlob(*blobs[i])) {
					blob->add(*blobs[i]);
				}
				blobs.erase(i);
				labels.erase(labels.begin() + i);
				continue;
			}
			++i;
		}
	}
	//filter low area
	//removeSmallBlobs(blobs);
	cleanNoises(blobs);
}

bool sortByVertical(Blob* blob1, Blob* blob2) {
	cv::Rect b1 = blob1->boundingRect();
	cv::Rect b2 = blob2->boundingRect();
	int center1 = b1.x + b1.width / 2;
	int center2 = b2.x + b2.width / 2;
	return (center1 < center2);
}

void sortBlobsByVertical(Blobs &blobs) {
	//sort
	blobs.sort(sortByVertical);
}


#define T_BROKEN_1 3
#define T_BROKEN_2 0.5
#define T_GROUPING 0.2

bool isFragment(int strHeight, int median, cv::Rect &bound) {
	if (bound.y + bound.height < median || bound.y > median) { //if not intersect median
		return true;
	}
	float aboveH = median - bound.y;
	float belowH = bound.y + bound.height - median;
	if (std::max(aboveH, belowH) / std::min(aboveH, belowH) > T_BROKEN_1) {
		return true;
	}
	return (bound.height / (float) strHeight) < T_BROKEN_2;
}

void groupFragment(Blobs &blobs, int idx1, int idx2, int idx3 = -1);

void groupFragment(Blobs &blobs, int idx1, int idx2, int idx3) {
	Blob* blob1 = blobs[idx1];
	Blob* blob2 = blobs[idx2];
	blob1->add(*blob2);
	blobs.erase(idx2);

	if (idx3 != -1) {
		if (idx3 > idx2) {
			--idx3;
		}
		Blob* blob3 = blobs[idx3];
		blob1->add(*blob3);
		blobs.erase(idx3);
	}
}

void defragment(cv::Mat& strImg, Blobs &blobs) {
	sortBlobsByVertical(blobs);
	bool hasBroken = false;
	int strH = strImg.rows;
	int median = strH / 2;
	do {
		hasBroken = false;
		for (int i = 0; i < blobs.size(); ++i) {
			auto bound = blobs[i]->boundingRect();
			if (!isFragment(strH, median, bound)) {
				continue;
			}

			int ccLeft = (i > 0) ? distanceBlobs(*blobs[i], *blobs[i - 1]) : INT_MAX;
			int ccRight = (i < blobs.size() - 1) ? distanceBlobs(*blobs[i], *blobs[i + 1]) : INT_MAX;
//			if (std::abs(ccLeft - ccRight) / (float) strH < T_GROUPING) {
//				continue;
////				groupFragment(blobs, i - 1, i, i + 1);
//			} else
				if (ccLeft < ccRight) {
				groupFragment(blobs, i - 1, i);
			} else {
				groupFragment(blobs, i, i + 1);
			}
			hasBroken = true;
		}
	} while (hasBroken);
}

inline double _distance(cv::Point2i p1, cv::Point2i p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

double distanceBlobs(const std::vector<cv::Point2i >& blob1, const std::vector<cv::Point2i >& blob2) {
	double min = INT_MAX;
	double temp = 0;
	for (auto &p1 : blob1) {
		for (auto &p2 : blob2) {
			if ((temp = _distance(p1, p2)) < min) {
				min = temp;
			}
		}
	}
	return min;
}

double distanceBlobs(const Blob& blob1, const Blob& blob2) {
	return distanceBlobs(blob1.points, blob2.points);
}
