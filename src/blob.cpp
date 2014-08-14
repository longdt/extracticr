#include "preprocessor.h"
#include <opencv2/imgproc/imgproc.hpp>

void findBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs, std::vector<cv::Rect> *bounds)
{
	blobs.clear();
	if (bounds != NULL) {
		bounds->clear();
	}
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

			std::vector <cv::Point2i> blob;
			int maxX = 0, minX = binary.cols, maxY = 0, minY = binary.rows;
			for (int i = rect.y; i < (rect.y + rect.height); i++) {
				int *row2 = (int*)label_image.ptr(i);
				for (int j = rect.x; j < (rect.x + rect.width); j++) {
					if (row2[j] != label_count) {
						continue;
					}
					else if (bounds != NULL) {
						maxX = std::max(maxX, j);
						minX = std::min(minX, j);

						maxY = std::max(maxY, i);
						minY = std::min(minY, i);
					}
					blob.push_back(cv::Point2i(j, i));
				}
			}
			blobs.push_back(blob);
			if (bounds != NULL) {
				rect.x = minX;
				rect.y = minY;
				rect.width = maxX - minX + 1;
				rect.height = maxY - minY + 1;
				bounds->push_back(rect);
			}
			label_count++;
		}
	}
}


cv::Mat drawBlob(const cv::Mat& src, const std::vector < std::vector<cv::Point2i > >& blobs) {
	cv::Mat output = cv::Mat::zeros(src.size(), CV_8UC3);
	// Randomy color the blobs
	for (size_t i = 0; i < blobs.size(); i++) {
		unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));

		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			output.at<cv::Vec3b>(y, x)[0] = b;
			output.at<cv::Vec3b>(y, x)[1] = g;
			output.at<cv::Vec3b>(y, x)[2] = r;
		}
	}
	return output;
}

class DisjointDigit {
public:
	bool operator() (cv::Rect b1, cv::Rect b2) {
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



void groupVertical(std::vector < std::vector<cv::Point2i> > &blobs, std::vector<cv::Rect> &bounds, std::vector<int> &labels) {
	cv::partition(bounds, labels, DisjointDigit());
	//join group
	for (int label = 0; label < labels.size(); ++label) {
		//find first blob of label
		std::vector<cv::Point2i> *blob = NULL;
		cv::Rect *bound = NULL;
		int i = 0;
		for (; i < labels.size(); ++i) {
			if (labels[i] == label) {
				blob = &blobs[i];
				bound = &bounds[i];
				break;
			}
		}
		if (blob == NULL) { //no blob (aka no label)
			break;
		}
		++i;
		bool joint = false;
		while (i < labels.size()) {
			if (labels[i] == label) {
				blob->insert(blob->end(), blobs[i].begin(), blobs[i].end());
				blobs.erase(blobs.begin() + i);
				bounds.erase(bounds.begin() + i);
				labels.erase(labels.begin() + i);
				joint = true;
				continue;
			}
			++i;
		}
		if (joint) {
			*bound = cv::boundingRect(*blob);
		}
	}
	//filter low area
	double perimaterSum = 0;
	double baseLineSum = 0;
	int i = 0;
	for (; i < bounds.size(); ++i) {
		perimaterSum += (bounds[i].width + bounds[i].height);
		baseLineSum += bounds[i].y + bounds[i].height;
	}
	double perimaterAverg = perimaterSum / bounds.size();
	double baseLineAverg = baseLineSum / bounds.size();
	i = 0;
	double rate = 0;
	while (i < bounds.size()) {
		rate = (bounds[i].width + bounds[i].height) / perimaterAverg;
		if (rate < 0.2 || (rate < 0.5 && (bounds[i].y + bounds[i].height) < baseLineAverg * 0.9)) {
			blobs.erase(blobs.begin() + i);
			bounds.erase(bounds.begin() + i);
			labels.erase(labels.begin() + i);
			continue;
		}
		++i;
	}
}

class VerticalSort {
private:
	std::vector<cv::Rect> m_bounds;
public:
	VerticalSort(std::vector<cv::Rect> &bounds) : m_bounds(bounds) {}
	bool operator() (int i, int j) {
		int centeri = m_bounds[i].x + m_bounds[i].width / 2;
		int centerj = m_bounds[j].x + m_bounds[j].width / 2;
		return (centeri < centerj);
	}
};

void sortBlobsByVertical(std::vector<cv::Rect> &bounds, std::vector<int> &order) {
	order.resize(bounds.size());
	for (int i = 0; i < bounds.size(); ++i) {
		order[i] = i;
	}
	//sort
	sort(order.begin(), order.end(), VerticalSort(bounds));
}