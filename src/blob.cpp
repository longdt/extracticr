#include "preprocessor.h"
#include <opencv2/imgproc/imgproc.hpp>

Blob::Blob() : needNewRect(false) {
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
	points.insert(points.end(), other.points.begin(), other.points.end());
}
void Blob::add(const cv::Point2i& point) {
	needNewRect = true;
	points.push_back(point);
}

void Blob::setModify(bool flag) {
	needNewRect = flag;
}

Blobs::Blobs() {}

size_t Blobs::size() const {
	return blobs.size();
}

void Blobs::erase(int index) {
	Blob* b = blobs[index];
	delete b;
	blobs.erase(blobs.begin() + index);
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

cv::Rect boundingRect(const Blobs &blobs) {
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

cv::Mat drawBlob(const Blobs& blobs) {
	cv::Rect rect = boundingRect(blobs);
	cv::Mat output = cv::Mat::zeros(rect.y + rect.height, rect.x + rect.width, CV_8UC3);
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
	return output;
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

		int min = 0;
		int max = 0;
		if (b1.x < maxX2 || b2.x < maxX1) {
			min = std::max(b1.x, b2.x);
			max = std::min(maxX1, maxX2);
		}
		float overLap = (max - min);
		return (overLap / b1.width + overLap / b2.width) > 0.7 ? true : (b1.y + b1.height <= b2.y || b2.y + b2.height <= b1.y);
	}
};



void groupVertical(Blobs& blobs, std::vector<int> &labels) {
	blobs.partition(labels, DisjointDigit());
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
		++i;
		while (i < labels.size()) {
			if (labels[i] == label) {
				blob->add(*blobs[i]);
				blobs.erase(i);
				labels.erase(labels.begin() + i);
				continue;
			}
			++i;
		}
	}
	//filter low area
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
		if (rate < 0.2 || (rate < 0.5 && (bound.y + bound.height) < baseLineAverg * 0.9)) {
			blobs.erase(i);
			labels.erase(labels.begin() + i);
			continue;
		}
		++i;
	}
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

double distanceBlobs(std::vector<cv::Point2i >& blob1, std::vector<cv::Point2i >& blob2) {
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

double distanceBlobs(Blob& blob1, Blob& blob2) {
	return distanceBlobs(blob1.points, blob2.points);
}
