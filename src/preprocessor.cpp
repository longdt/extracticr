
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "recognizer.h"
#include "util/misc.h"
#include "preprocessor.h"

using namespace tiny_cnn;
using namespace cv;


extern digit_recognizer recognizer;

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

cv::Mat makeDigitMat(cv::Mat& crop) {
	int width = 0;
	int height = 0;
	int paddingX = 0;
	int paddingY = 0;
	if (crop.rows > crop.cols) {
		//scale to height
		height = 20;
		width = (height * crop.cols) / crop.rows;
	}
	else {
		width = 20;
		//scale to width
		height = (width * crop.rows) / crop.cols;
	}
	cv::Size size(width, height);
	cv::Mat resize;
	cv::resize(crop, resize, size);
	cv::Mat padded(28, 28, CV_8UC1);
	padded.setTo(cv::Scalar::all(0));
	paddingX = (28 - resize.cols) / 2;
	paddingY = (28 - resize.rows) / 2;
	resize.copyTo(padded(cv::Rect(paddingX, paddingY, resize.cols, resize.rows)));
	//		cv::copyMakeBorder(resize, pad, 4, 4, 4, 4, cv::BORDER_CONSTANT, cv::Scalar(0));
	return padded;
}

cv::Mat makeDigitMat(std::vector<cv::Point2i >& blob, cv::Rect* bound) {
	cv::Mat crop = cropBlob(blob, *bound);
	cropMat(deslant(crop), crop);
	int width = 0;
	int height = 0;
	int paddingX = 0;
	int paddingY = 0;
	if (crop.rows > crop.cols) {
		//scale to height
		height = 20;
		width = (height * crop.cols) / crop.rows;
	}
	else {
		width = 20;
		//scale to width
		height = (width * crop.rows) / crop.cols;
	}
	cv::Size size(width, height);
	cv::Mat resize;
	cv::resize(crop, resize, size);
	cv::Mat padded(28, 28, CV_8UC1);
	padded.setTo(cv::Scalar::all(0));
	paddingX = (28 - resize.cols) / 2;
	paddingY = (28 - resize.rows) / 2;
	resize.copyTo(padded(cv::Rect(paddingX, paddingY, resize.cols, resize.rows)));
	//		cv::copyMakeBorder(resize, pad, 4, 4, 4, 4, cv::BORDER_CONSTANT, cv::Scalar(0));
	return padded;
}

bool isSingleDigit(double conf, cv::Rect& bound) {
	float aspect = bound.width / (float)(bound.height);
	if (aspect > 1) {
		return conf > 0.8;
	}
	else if (aspect >= 0.6) {
		return conf > 0.6;
	}
	else if (aspect >= 0.5) {
		return conf > 0.1;
	}
	else {
		return conf > 0;
	}
}

void extractDigit(cv::Mat &binary, std::vector < std::vector<cv::Point2i > >& blobs, std::vector<cv::Rect> &bounds) {
	std::vector<int> projectV;
	projectV.resize(binary.cols, 0);
	vector<int> order;
	sortBlobsByVertical(bounds, order);
	for (auto i : order) {
		vec_t in;
		double conf;
		cv::Mat digit = makeDigitMat(blobs[i], &bounds[i]);
		mat_to_vect(digit, in);
		int label = recognizer.predict(in, &conf);
		cv::imshow(std::to_string(label) + "pad" + std::to_string(conf), digit);
		if (isSingleDigit(conf, bounds[i])) {
			std::cout << label;
		}
		else if (!recognizeDigits(blobs[i], bounds[i], label, conf)){
			std::cout << label;
			updateVerticalProjection(blobs[i], projectV);
		}
	}
	std::cout << std::endl;
	auto cuts = genVerticalCuts(projectV);
	//draw and show 
	cv::Scalar color = cv::Scalar(255, 255, 255);
	for (int c : cuts) {
		cv::line(binary, cv::Point(c, 0), cv::Point(c, binary.cols), color, 1, 8);
	}
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
	double sum = 0;
	int i = 0;
	for (; i < bounds.size(); ++i) {
		sum += bounds[i].area();
	}
	sum = sum / bounds.size();
	i = 0;
	while (i < bounds.size()) {
		if (bounds[i].area() / sum < 0.2) {
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