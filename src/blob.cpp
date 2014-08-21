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


Blobs findBlobs(const cv::Mat &binary) {
	Blobs blobs;
	findBlobs(binary, blobs);
	return blobs;
}

void findBlobs(const cv::Mat &binary, Blobs &blobs) {
	clearBlobs(blobs);
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
			blobs.push_back(blob);
			label_count++;
		}
	}
}

void clearBlobs(Blobs &blobs) {
	for (Blob* blob : blobs) {
		delete blob;
	}
	blobs.clear();
}

cv::Mat drawBlob(const cv::Mat& src, const Blobs& blobs) {
	cv::Mat output = cv::Mat::zeros(src.size(), CV_8UC3);
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
	cv::partition(blobs, labels, DisjointDigit());
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
				blobs.erase(blobs.begin() + i);
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
			blobs.erase(blobs.begin() + i);
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
	sort(blobs.begin(), blobs.end(), sortByVertical);
}
