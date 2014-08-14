
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

cv::Mat cropBlob(std::vector<cv::Point2i >& blob, cv::Rect& bound, int pad) {
	cv::Mat rs = cv::Mat::zeros(bound.height + 2 * pad, bound.width + 2 * pad, CV_8UC1);
	for (size_t j = 0; j < blob.size(); j++) {
		int x = blob[j].x - bound.x + pad;
		int y = blob[j].y - bound.y + pad;

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
	dst = cv::Mat::zeros(rect.height + 2 * pad, rect.width + 2 * pad, src.type());
	src(rect).copyTo(dst(cv::Rect(pad, pad, rect.width, rect.height)));
	return true;
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

void projectVeritcal(std::vector < std::vector<cv::Point2i> > &blobs, std::vector<int> &output) {
	std::fill(output.begin(), output.end(), 0);
	for (auto blob : blobs) {
		for (auto p : blob) {
			output[p.x] += 1;
		}
	}
}

void updateVerticalProjection(std::vector<cv::Point2i>& blob, std::vector<int>& output) {
	for (auto p : blob) {
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
	std::vector < std::vector<cv::Point2i> > blobs;
	std::vector<cv::Rect> bounds;
	cv::Mat temp = input / 255;
	findBlobs(temp, blobs, &bounds);
	int cost = 0;
	for (auto b : bounds) {
		cost += b.width;
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
