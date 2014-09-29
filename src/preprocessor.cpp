
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
	auto rect = getROI(src);
	if (rect.width == 0) {
		return false;
	}
	if (pad > 0) {
		dst = cv::Mat::zeros(rect.height + 2 * pad, rect.width + 2 * pad, src.type());
		src(rect).copyTo(dst(cv::Rect(pad, pad, rect.width, rect.height)));
	} else {
		dst = src(rect);
	}
	return true;
}

cv::Rect getROI(cv::Mat& src) {
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
		return cv::Rect();
	}
	return cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
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


float projectWidth(cv::Mat& input) {
	float width = 0;
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


float blobsWidth(cv::Mat& input) {
	Blobs blobs = findBlobs(input);
	float cost = 0;
	for (auto b : blobs) {
		cost += b->boundingRect().width;
	}
	cost = projectWidth(input) * 0.8 + cost * 0.2;
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
	double tag = tan(std::abs(degree) * PI / 180.0);
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

float resolveBlobAngle(Blob& blob, int imgHeight, float imgSlantAngle) {
	double tag = tan(std::abs(imgSlantAngle) * PI / 180.0);
	float blobMove = 0;
	auto rect = blob.boundingRect();
	if (imgSlantAngle > 0) {
		blobMove = tag * (rect.y + rect.height);
	} else {
		blobMove = tag * (rect.y - imgHeight);
	}
	float result = atan(blobMove / rect.height) * 180 / PI;
	return result;
}
/* input[0,1] output[0,1] */
float deslant(cv::Mat& input, cv::Mat *dst, float (*fntSlantCost)(cv::Mat&)) {
	float width = fntSlantCost(input);
	float minWidth = width;
	float degree = 0;
	float stepDegree = degree;
	cv::Mat rotated;
	//try rotate +/-5degree
	int step = 16;
	while (step != 0 && (degree <= 48 && degree >= -48)) {
		rotated = slant(input, degree + step);
		width = fntSlantCost(rotated);
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
		width = fntSlantCost(rotated);
		if (width < minWidth) {
			minWidth = width;
			degree += step;
			continue;
		}
		step = step / 2;
		stepDegree = degree;
		continue;
	}
	if (dst != NULL && degree != 0) {
		*dst = slant(input, degree);
	}
	return degree;
}

/*implement new deslant function*/

void genSlantShiftX(float angle, int imgHeight, std::vector<int>& moveX) {
	moveX.resize(imgHeight);
	double tag = tan(std::abs(angle) * PI / 180.0);
	int x = 0;
	double error = 0;
	int increasement = angle >= 0 ? 1 : -1;
	for (int y = 0; y < imgHeight; ++y) {
		moveX[y] = x;
		error += tag;
		if (error >= 0.5) {
			x += increasement;
			--error;
		}
	}
}

void slant(int imgHeight, Blobs& blobs, float angle) {
	std::vector<int> moveX;
	genSlantShiftX(angle, imgHeight, moveX);
	int padding = angle >= 0 ? 0 : -moveX[imgHeight - 1];
	for (Blob* b : blobs) {
		for (auto &p : b->points) {
			p.x += moveX[p.y] + padding;
		}
		b->setModify(true);
	}
}

float slantCost(Size imgSize, Blobs& blobs, float angle) {
	float bcost = 0;
	float widCost = 0;
	std::vector<int> moveX;
	genSlantShiftX(angle, imgSize.height, moveX);
	int padding = angle >= 0 ? 0 : -moveX[imgSize.height - 1];
	int newWidth = imgSize.width + std::abs(moveX[imgSize.height - 1]);
	std::vector<bool> pwImg(newWidth, false);
	int newX = 0;
	for (Blob* b : blobs) {
		std::vector<bool> pwBlob(newWidth, false);
		for (auto &p : b->points) {
			newX = p.x + moveX[p.y] + padding;
			if (!pwImg[newX]) {
				pwImg[newX] = true;
				pwBlob[newX] = true;
				++widCost;
				++bcost;
			} else if (!pwBlob[newX]) {
				pwBlob[newX] = true;
				++bcost;
			}
		}
	}
	return widCost * 0.8 + bcost * 0.2;
}

float deslant(Size imgSize, Blobs& blobs) {
	int cost = slantCost(imgSize, blobs, 0);
	int minCost = cost;
	float degree = 0;
	float stepDegree = degree;
	//try rotate +/-5degree
	int step = 16;
	while (step != 0 && (degree <= 48 && degree >= -48)) {
		cost = slantCost(imgSize, blobs, degree + step);
		if (cost < minCost) {
			minCost = cost;
			degree += step;
			continue;
		}
		else if (degree != stepDegree) {
			step = step / 2;
			stepDegree = degree;
			continue;
		}
		step = -step;
		cost = slantCost(imgSize, blobs, degree + step);
		if (cost < minCost) {
			minCost = cost;
			degree += step;
			continue;
		}
		step = step / 2;
		stepDegree = degree;
		continue;
	}
	if (degree != 0) {
		slant(imgSize.height, blobs, degree);
	}
	return degree;
}
/*end implement new deslant function*/

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
