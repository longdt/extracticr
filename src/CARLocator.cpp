/*
 * CARLocator.cpp
 *
 *  Created on: Oct 8, 2014
 *      Author: thienlong
 */

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "CARLocator.h"

#include "preprocessor.h"

#include "blob.h"
using cv::GaussianBlur;
using cv::line;

namespace icr {
using namespace cv;
CARLocator::CARLocator(cv::Mat& cheqImg) : cheqImg(cheqImg) {
}

#define INSIDE(y0, y1, p) (y0 <= p && y1 >= p)

bool intersect(int y0, int y1, Rect rect) {
	if (y0 > y1) {
		std::swap(y0, y1);
	}
	return INSIDE(y0, y1, rect.y) || INSIDE(y0, y1, rect.y + rect.height)
			|| (y0 > rect.y && y1 < rect.y + rect.height);
}

/*alpha implementation*/
cv::Rect CARLocator::getCARLocation() {
	int x = 0.63 * cheqImg.cols;
	int y = 0.35 * cheqImg.rows;
	Rect roi(x, y, cheqImg.cols * 0.35, cheqImg.rows * 0.3);
	Mat img = this->cheqImg(roi);
	Mat edges;
	Canny(img, edges, 80, 120);
	std::vector<cv::Vec4i> lines;
	HoughLinesP(edges, lines, 1, CV_PI/ 2, 50, 50, 10 );
	Rect mpr = getMPRLocation();
	int minX = mpr.x;
	int maxX = mpr.x + mpr.width - 1;
	int minY = mpr.y;
	int maxY = mpr.y + mpr.height - 1;
	//find vertical line
	Mat cdst;
	cvtColor(this->cheqImg, cdst, CV_GRAY2BGR);
	for( size_t i = 0; i < lines.size(); i++ ) {
	  Vec4i l = lines[i];
	  l[0] += roi.x;
	  l[2] += roi.x;
	  l[1] += roi.y;
	  l[3] += roi.y;
	  line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0, 255), 1, CV_AA);
	  if (l[0] == l[2] && std::abs(l[1] - l[3]) > mpr.height * 0.5 && intersect(l[1], l[3], mpr)) {
		  if (l[0] < mpr.x + mpr.width / 2) {
			  minX = std::max(std::max(l[0], mpr.x), minX);
		  } else {
			  maxX = std::min(std::min(l[0], mpr.x + mpr.width), maxX);
		  }
	  } else if (l[1] == l[3] && std::abs(l[0] - l[2]) > mpr.width * 0.3) {
		  if (l[1] < mpr.y + mpr.height / 2) {
			  minY = std::max(std::max(l[1], mpr.y), minY);
		  } else {
			  maxY = std::min(std::min(l[1], mpr.y + mpr.height), maxY);
		  }
	  }
	}

	roi.x = minX;
	roi.y = minY;
	roi.width = maxX - minX + 1;
	roi.height = maxY - minY + 1;
	cv::rectangle(cdst, roi, CV_RGB(0,255,0));
	imshow("car", cdst);
	return roi;
}

cv::Rect CARLocator::getMPRLocation() {
	int x = 0.645 * cheqImg.cols;
	int y = 0.4 * cheqImg.rows;
	return Rect(x, y, 0.5 * x, 0.37 * y);
}

inline bool isInside(const Rect& box, Point p) {
	return (box.x <= p.x && box.y <= p.y) && (box.x + box.width) > p.x && (box.y + box.height) > p.y;
}

//cv::Rect CARLocator::getRMLocation() {
//	Mat bin;
//	doThreshold(cheqImg, bin, BhThresholdMethod::OTSU);
//	bin = bin / 255;
//	Blobs blobs;
//	findBlobs(bin, blobs);
//	return getRMLocation(blobs);
//}

cv::Rect CARLocator::getRMLocation(Blobs& blobs, cv::Rect& carLoc) {
	Rect box = getMPRLocation();
	int blobWidthThres = box.height / 2;
	if (blobWidthThres < 70) { //blobWidthThres must at least 70
		blobWidthThres = 70;
	}
#define PADDING_HEIGHT 0.2
	box.y += box.height * PADDING_HEIGHT;
	box.height *= 1 - PADDING_HEIGHT * 2;
	box.width = box.height;

	//select blob within box
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
		if (!isInside(box, center) || rect.width > blobWidthThres || rect.height >= blobWidthThres) {
			blobs.erase(i);
			--i;
		}
	}
	Rect rs = blobs.boundingRect();
	if (rs.height >= rs.width) {
		//select blob within box
		for (size_t i = 0; i < blobs.size(); ++i) {
			Blob* b = blobs[i];
			Rect rect = b->boundingRect();
			if (rect.y <= carLoc.y || rect.y + rect.height >= carLoc.y + carLoc.height) {
				blobs.erase(i);
				--i;
			}
		}
		rs = blobs.boundingRect();
	} else if (rs.height + rs.y > carLoc.y + carLoc.height) {
		carLoc.height += 5;
	}
	return rs;
}




void trimNoises(Blobs& blobs) {
	Rect rect = blobs.boundingRect();
	vector<int> projectV(rect.x + rect.width, 0);
	projectVertical(blobs, projectV);
	//find big space
	vector<int> spaces;
	spaces.push_back(rect.x);
	int start = rect.x;
	int end = 0;
	int minSpace = rect.height * 1.5;
	while (true) {
		//find start
		for (; start < projectV.size() && projectV[start] > 0; ++start) {}
		if (start == projectV.size()) {
			break;
		}
		//find end
		for (end = start + 1; end < projectV.size() && projectV[end] == 0; ++end){}
		if (end - start >= minSpace) {
			spaces.push_back(start);
			spaces.push_back(end);
		}
		start = end;
	}
	if (spaces.size() == 1) {
		return;
	}
	spaces.push_back(projectV.size());
	//select best region
	int bestStart = 0;
	int bestEnd = 0;
	int bestScore = 0;
	for (size_t i = 0; i < spaces.size(); i += 2) {
		start = spaces[i];
		end = spaces[i + 1];
		int score = 0;
		for (size_t j = start; j < end; ++j) {
			score += projectV[j];
		}
		if (score > bestScore) {
			bestScore = score;
			bestStart = start;
			bestEnd = end;
		}
	}
	//remove noisy blobs
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		if (!(rect.x >= bestStart && rect.x + rect.width <= bestEnd)) {
			blobs.erase(i);
			--i;
		}
	}
}

void CARLocator::getHandwrittenBlobs(Blobs& blobs) {
	Mat bin;
	doThreshold(cheqImg, bin, BhThresholdMethod::OTSU);
	bin = bin / 255;
	findBlobs(bin, blobs);
	Blobs rmBlobs;
	for (size_t i = 0; i < blobs.size(); ++i) {
		blobs[i]->boundingRect();
	}
	blobs.clone(rmBlobs);
	Rect box = getCARLocation();
	Rect rm = getRMLocation(rmBlobs, box);
//	Mat cdst;
//	cvtColor(this->cheqImg, cdst, CV_GRAY2BGR);
//	cv::rectangle(cdst, rm, CV_RGB(0,0,255));
//	cv::rectangle(cdst, box, CV_RGB(0,255,0));
//	imshow("cdst", cdst);
	if (rm.height == 0 || rm.width / rm.height > 5) {		//can't detect RM symbol
		blobs.clear();
		return;
	}

	int startX = rm.x + rm.width;
	int endX = box.x + box.width;
	//get only blob inside box
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
		if (!isInside(box, center) || rect.x <= startX || rect.width > box.width / 2 || rect.x + rect.width > endX) {
			blobs.erase(i);
			--i;
		} else if (rect.y + rect.height + 2 >= box.y + box.height && rect.width / (float) rect.height > 2) { //TODO more condition
			blobs.erase(i);
			--i;
		}
	}
	blobs.sort(sortByVertical);
	int y0 = rm.y;
	int y1 = rm.y + rm.height;
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		if (!intersect(y0, y1, rect)) {
			blobs.erase(i);
			--i;
		} else {
			y0 = std::min(y0, rect.y);
			y1 = std::max(y1, rect.y + rect.height);
		}
	}
	trimNoises(blobs);
}

CARLocator::~CARLocator() {
}

PhCARLocator::PhCARLocator(cv::Mat& cheqImg) : CARLocator(cheqImg), mprImg(), boundingBox(false) {
	Rect mpr = getMPRLocation();
	mprImg = cheqImg(mpr);
	threshold(mprImg, mprImg, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
}

cv::Rect PhCARLocator::getRMLocation(Blobs& blobs, cv::Rect& carLoc) {
	if (!boundingBox) {
		return Rect(carLoc.x, carLoc.y + carLoc.height / 4, 50, carLoc.height / 2);
	}
	Rect box = carLoc;
	int blobWidthThres = box.height / 2;
	if (blobWidthThres < 70) { //blobWidthThres must at least 70
		blobWidthThres = 70;
	}
#define PADDING_HEIGHT 0.2
	box.y += box.height * PADDING_HEIGHT;
	box.height *= 1 - PADDING_HEIGHT * 2;
	box.width = box.height;

	//select blob within box
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
		if (!isInside(box, center) || rect.width > blobWidthThres || rect.height >= blobWidthThres) {
			blobs.erase(i);
			--i;
		}
	}
	Rect rs = blobs.boundingRect();
	if (rs.height >= rs.width) {
		//select blob within box
		for (size_t i = 0; i < blobs.size(); ++i) {
			Blob* b = blobs[i];
			Rect rect = b->boundingRect();
			if (rect.y <= carLoc.y || rect.y + rect.height >= carLoc.y + carLoc.height) {
				blobs.erase(i);
				--i;
			}
		}
		rs = blobs.boundingRect();
	} else if (rs.height + rs.y > carLoc.y + carLoc.height) {
		carLoc.height += 5;
	}
	return rs;
}

cv::Rect PhCARLocator::getMPRLocation() {
	int x = 0.7 * cheqImg.cols;
	int y = 0.2 * cheqImg.rows;
	return Rect(x, y, cheqImg.cols - x, cheqImg.rows * 0.2);
}

void findStrokes(Mat& src, int row, std::vector<int>& strokes) {
	int c = 0;
	while (c < src.cols) {
		//find start
		for (; c < src.cols && src.at<uchar>(row, c) == 0; ++c) {}
		if (c == src.cols) {
			break;
		}
		strokes.push_back(c);
		//find end
		for (; c < src.cols && src.at<uchar>(row, c) != 0; ++c) {}
		strokes.push_back(c);
	}
}


int distanceK(Mat& src, int row, int left, int right) {
	--row;
	if (src.at<uchar>(row, left) == 0) {
		for (; left > 0 && src.at<uchar>(row, left) == 0; --left) {}
	} else {
		for (; left < src.cols && src.at<uchar>(row, left) != 0; ++left) {}
	}
	--right;
	if (src.at<uchar>(row, right) == 0) {
		for (; right < src.cols && src.at<uchar>(row, right) == 0; ++right) {}
	} else {
		for (; right > 0 && src.at<uchar>(row, right) != 0; --right) {}
	}
	return right - left - 1;
}

void fillUStrokes(Mat& src, int lt, int row, std::vector<int>& strokes) {
	if (strokes.size() < 4) {
		return;
	}
	for (int i = 1; i < strokes.size() - 1; i += 2) {
		int x = strokes[i + 1] - strokes[i];
		int y = floor((2 * lt + 2.0) * x / (2 * lt + 1.0));
		int k = distanceK(src, row, strokes[i], strokes[i + 1]);
		std::cout << k << std::endl;
		if (y <= k + 2 && y >= k -2) {
			line(src, Point(strokes[i] - 2, row), Point(strokes[i + 1] + 2, row), Scalar(255, 255, 255), 1, CV_AA);
			line(src, Point(strokes[i] - 2, row + 1), Point(strokes[i + 1] + 2, row + 1), Scalar(255, 255, 255), 1, CV_AA);
		}
	}
}

void removeBaseline(Mat& mprImg, std::vector<cv::Vec4i>& lines, int maxY) {
	--maxY;
	Rect r(0, maxY, mprImg.cols, mprImg.rows - maxY);
	mprImg(r) = Scalar::all(0);
	removeNoise(mprImg);
//	removeNoise(mprImg);
	std::set<int> lt;
	lt.insert(maxY);
	for( size_t i = 0; i < lines.size(); i++ ) {
		Vec4i l = lines[i];
		if (l[1] == l[3] && std::abs(l[0] - l[2]) > mprImg.cols * 0.3 && l[1] > mprImg.rows / 2) {
			lt.insert(l[1]);
		}
	}
	int l1 = *std::min_element(lt.begin(), lt.end()) - 1;
	std::vector<int> strokes;
	findStrokes(mprImg, l1, strokes);
	fillUStrokes(mprImg, lt.size(), l1, strokes);
}

cv::Rect PhCARLocator::getCARLocation() {
	boundingBox = false;
	std::vector<cv::Vec4i> lines;
	HoughLinesP(mprImg, lines, 1, CV_PI/ 2, 40, 40, 10 );
	int minX = 0;
	int maxX = mprImg.cols - 1;
	int minY = 0;
	int maxY = mprImg.rows - 1;
	//find vertical line
	Mat cdst;
	cvtColor(this->mprImg, cdst, CV_GRAY2BGR);
	for( size_t i = 0; i < lines.size(); i++ ) {
	  Vec4i l = lines[i];
	  line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0, 255), 1, CV_AA);
	  if (l[0] == l[2] && std::abs(l[1] - l[3]) > mprImg.rows * 0.3) {
		  if (l[0] < mprImg.cols / 2) {
			  minX = std::max(l[0], minX);
			  boundingBox = true;
		  } else {
			  maxX = std::min(l[0], maxX);
			  boundingBox = true;
		  }
	  } else if (l[1] == l[3] && std::abs(l[0] - l[2]) > mprImg.cols * 0.3) {
		  if (l[1] < mprImg.rows / 2) {
			  minY = std::max(l[1], minY);
		  } else {
			  maxY = std::min(l[1], maxY);
		  }
	  }
	}
	Rect car(minX, minY, maxX - minX + 1, maxY - minY + 1);
	//remove guideline
	removeBaseline(mprImg, lines, maxY);
	cv::rectangle(cdst, car, CV_RGB(0,255,0));
	imshow("car", cdst);
	imshow("mprImg", mprImg);
	return car;
}

void PhCARLocator::getHandwrittenBlobs(Blobs& blobs) {
	Rect box = getCARLocation();
	box.height += 4;
	Mat bin;
	bin = mprImg / 255;
	findBlobs(bin, blobs);
	Blobs rmBlobs;
	for (size_t i = 0; i < blobs.size(); ++i) {
		blobs[i]->boundingRect();
	}
	blobs.clone(rmBlobs);
	Rect rm = getRMLocation(rmBlobs, box);
	Mat cdst;
	cvtColor(this->mprImg, cdst, CV_GRAY2BGR);
	cv::rectangle(cdst, rm, CV_RGB(0,0,255));
	cv::rectangle(cdst, box, CV_RGB(0,255,0));
	imshow("cdst", cdst);
	if (rm.height == 0 || rm.width / rm.height > 5) {		//can't detect RM symbol
		blobs.clear();
		return;
	}

	int startX = rm.x + rm.width;
	int endX = box.x + box.width;
	//get only blob inside box
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
		if (!isInside(box, center) || rect.x <= startX || rect.width > box.width / 2 || rect.x + rect.width > endX) {
			blobs.erase(i);
			--i;
		} else if (rect.y + rect.height + 2 >= box.y + box.height && rect.width / (float) rect.height > 2) { //TODO more condition
			blobs.erase(i);
			--i;
		}
	}
	blobs.sort(sortByVertical);
	int y0 = rm.y;
	int y1 = rm.y + rm.height;
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		if (!intersect(y0, y1, rect)) {
			blobs.erase(i);
			--i;
		} else {
			y0 = std::min(y0, rect.y);
			y1 = std::max(y1, rect.y + rect.height);
		}
	}
	trimNoises(blobs);
}

} /* namespace icr */
