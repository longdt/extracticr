
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include "recognizer.h"
#include "util/misc.h"
#include "preprocessor.h"

using namespace tiny_cnn;
using namespace cv;


extern digit_recognizer recognizer;

void cv::doNothing() {}

cv::Mat cropBlob(Blob& blob, int pad) {
	cv::Rect bound = blob.boundingRect();
	cv::Mat rs = cv::Mat::zeros(bound.height + 2 * pad, bound.width + 2 * pad, CV_8UC1);
	for (size_t j = 0; j < blob.points.size(); j++) {
		int x = blob.points[j].x - bound.x + pad;
		int y = blob.points[j].y - bound.y + pad;

		rs.at<uchar>(y, x) = 255;
	}
	return rs;
}

bool cropMat(cv::Mat& src, cv::Mat& dst, int pad) {
	int minX = src.cols - 1;
	int maxX = 0;
	int minY = src.rows - 1;
	int maxY = 0;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (src.at<uchar>(i,j) > 0) {
				if (minX > j) {
					minX = j;
				}
				if (maxX < j) {
					maxX = j;
				}
				if (minY > i) {
					minY = i;
				}
				if (maxY < i) {
					maxY = i;
				}
			}
		}
	}
	//handle black iamge
	if (maxX < minX) {
		return false;
	}
	auto rect = cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
	if (pad > 0) {
		dst = cv::Mat::zeros(rect.height + 2 * pad, rect.width + 2 * pad, src.type());
		src(rect).copyTo(dst(cv::Rect(pad, pad, rect.width, rect.height)));
	} else {
		dst = src(rect);
	}
	return true;
}



void projectHorizontal(cv::Mat &input, std::vector<int> &output) {
	output.resize(input.rows);
	for (int i = 0; i < input.rows; ++i) {
		int sum = 0;
		for (int j = 0; j < input.cols; ++j) {
			if (input.at<uchar>(i, j) > 0) {
				sum += 1;
			}
		}
		output[i] = sum;
	}
}

void projectVeritcal(cv::Mat &input, std::vector<int> &output) {
	output.resize(input.cols);
	for (int i = 0; i < input.cols; ++i) {
		int sum = 0;
		for (int j = 0; j < input.rows; ++j) {
			if (input.at<uchar>(j, i) > 0) {
				sum += 1;
			}
		}
		output[i] = sum;
	}
}

void projectVeritcal(std::vector < std::vector<cv::Point2i> > &blobPoints, std::vector<int> &output) {
	std::fill(output.begin(), output.end(), 0);
	for (auto blob : blobPoints) {
		for (auto p : blob) {
			output[p.x] += 1;
		}
	}
}

void updateVerticalProjection(std::vector<cv::Point2i>& blobPoints, std::vector<int>& output) {
	for (auto p : blobPoints) {
		output[p.x] += 1;
	}
}

int findLastMin(std::vector<int>& vec, int from, int last) {
	int minVal = 99999999;
	int minIdx = last;
	for (int i = from; i < last; ++i) {
		if (vec[i] == 0) {
			return i;
		} else if (vec[i] <= minVal) {
			minVal = vec[i];
			minIdx = i;
		}
	}
	return minIdx;
}

std::vector<int> genVerticalCuts(std::vector<int>& projectV) {
	std::vector<int> rs;
	int start = 0;
	int i = 0;
	int height = *std::max_element(projectV.begin(), projectV.end());
	while (i < projectV.size()) {
		//find start
		for (; i < projectV.size() && projectV[i] == 0; ++i) {}
		start = i;
		//find end = arg(min([start-> start + maxSgmentLen)
		bool inscrea = false;
		int maxSegmentLen = std::min(start + height, (int)projectV.size());
		int end = findLastMin(projectV, start + 1, maxSegmentLen);
		rs.push_back(start);
		rs.push_back(end);
		i = end;
	}
	return rs;
}


int projectWidth(cv::Mat& input) {
	int width = 0;
	for (int i = 0; i < input.cols; ++i) {
		for (int j = 0; j < input.rows; ++j) {
			if (input.at<uchar>(j, i) > 0) {
				width += 1;
				break;
			}
		}
	}
	return width;
}


int costSlant(cv::Mat& input) {
	cv::Mat temp = input / 255;
	Blobs blobs = findBlobs(temp);
	int cost = 0;
	for (auto b : blobs) {
		cost += b->boundingRect().width;
	}
	return cost;
}

#define PI 3.14159265


cv::Mat slant(cv::Mat& src, float degree) {
	Point2f srcTri[3];
	Point2f dstTri[3];
	/// Set the dst image the same type and size as src
	Mat warp_dst = Mat::zeros(src.rows, src.cols + src.rows, src.type());

	/// Set your 3 points to calculate the  Affine Transform
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1.f, 0);
	srcTri[2] = Point2f(0, src.rows - 1.f);
	double tag = tan(abs(degree) * PI / 180.0);
	if (degree > 0) {
		dstTri[0] = Point2f(0, 0);
		dstTri[1] = Point2f(src.cols - 1, 0);
		dstTri[2] = Point2f(src.rows * tag, src.rows - 1.f);
	}
	else {
		dstTri[0] = Point2f(src.rows * tag, src.rows*0.0f);
		dstTri[1] = Point2f(src.cols - 1.f + src.rows * tag, src.rows*0.0f);
		dstTri[2] = Point2f(0, src.rows - 1.f);
	}
	/// Get the Affine Transform
	Mat warp_mat = getAffineTransform(srcTri, dstTri);

	/// Apply the Affine Transform just found to the src image
	warpAffine(src, warp_dst, warp_mat, warp_dst.size());
	return warp_dst;
}

cv::Mat deslant(cv::Mat& input) {
	int width = projectWidth(input);
	int minWidth = width;
	float degree = 0;
	float stepDegree = degree;
	cv::Mat rotated;
	//try rotate +/-5degree
	int step = 16;
	while (step != 0 && (degree <= 45 || degree <= -45)) {
		rotated = slant(input, degree + step);
		width = projectWidth(rotated);
		if (width < minWidth) {
			minWidth = width;
			degree += step;
			continue;
		}
		else if (degree != stepDegree) {
			step = step / 2;
			stepDegree = degree;
			continue;
		}
		step = -step;
		rotated = slant(input, degree + step);
		width = projectWidth(rotated);
		if (width < minWidth) {
			minWidth = width;
			degree += step;
			continue;
		}
		step = step / 2;
		stepDegree = degree;
		continue;
	}
	return slant(input, degree);
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
			hasBroken = true;
			int ccLeft = (i > 0) ? distanceBlobs(*blobs[i], *blobs[i - 1]) : INT_MAX;
			int ccRight = (i < blobs.size() - 1) ? distanceBlobs(*blobs[i], *blobs[i + 1]) : INT_MAX;
			if (std::abs(ccLeft - ccRight) / (float) strH < T_GROUPING) {
				groupFragment(blobs, i - 1, i, i + 1);
			} else if (ccLeft < ccRight) {
				groupFragment(blobs, i - 1, i);
			} else {
				groupFragment(blobs, i, i + 1);
			}
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

cv::Mat cropDigitString(cv::Mat& src) {
	vector<int> horizontal;
	projectHorizontal(src, horizontal);
	auto it = std::max_element(horizontal.begin(), horizontal.end());
	int ymin = it - horizontal.begin();
	for (; ymin > 0 && horizontal[ymin] > 0; --ymin) {}
	int ymax = it - horizontal.begin();
	for (; ymax < horizontal.size() - 1 && horizontal[ymax] > 0; ++ymax) {}
	cv::Mat dst = src(cv::Rect(0, ymin, src.cols, ymax - ymin)).clone();
	return dst;
}
