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
#ifdef DEBUG
	Mat cdst;
	cvtColor(this->cheqImg, cdst, CV_GRAY2BGR);
#endif
	for( size_t i = 0; i < lines.size(); i++ ) {
	  Vec4i l = lines[i];
	  l[0] += roi.x;
	  l[2] += roi.x;
	  l[1] += roi.y;
	  l[3] += roi.y;
#ifdef DEBUG
	  line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0, 255), 1, CV_AA);
#endif
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
#ifdef DEBUG
	cv::rectangle(cdst, roi, CV_RGB(0,255,0));
	imshow("car", cdst);
#endif
	return roi;
}

cv::Rect CARLocator::getMPRLocation() {
	int x = 0.645 * cheqImg.cols;
	int y = 0.4 * cheqImg.rows;
	return Rect(x, y, 0.5 * x, 0.37 * y);
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
		if (!box.contains(center) || rect.width > blobWidthThres || rect.height >= blobWidthThres) {
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
		if (!box.contains(center) || rect.x <= startX || rect.width > box.width / 2 || rect.x + rect.width > endX) {
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

PrintedCARLocator::PrintedCARLocator(cv::Mat& cheqImg) : CARLocator(cheqImg), mprImg(), boundingBox(false) {
	Rect mpr = getMPRLocation();
	mprImg = cheqImg(mpr);
	threshold(mprImg, mprImg, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
}

void PrintedCARLocator::getHandwrittenBlobs(Blobs& blobs) {
	Rect box = getCARLocation();
	Mat bin;
	bin = mprImg / 255;
	findBlobs(bin, blobs);
	Blobs rmBlobs;
	for (size_t i = 0; i < blobs.size(); ++i) {
		blobs[i]->boundingRect();
	}
	blobs.clone(rmBlobs);
	Rect rm = getRMLocation(rmBlobs, box);
	mprImg = drawBinaryBlobs(blobs);
#ifdef DEBUG
	Mat cdst;
	cvtColor(this->mprImg, cdst, CV_GRAY2BGR);
	cv::rectangle(cdst, rm, CV_RGB(0,0,255));
	cv::rectangle(cdst, box, CV_RGB(0,255,0));
	imshow("cdst", cdst);
#endif
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
		if (!box.contains(center) || rect.x <= startX || rect.width > box.width / 2 || rect.x + rect.width > endX) {
			blobs.erase(i);
			--i;
		} else if (rect.y + rect.height + 2 >= box.y + box.height && rect.width / (float) rect.height > 2) { //TODO more condition
			blobs.erase(i);
			--i;
		}
	}
	blobs.sort(sortByVertical);
	removeNoise(blobs);
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

cv::Rect PrintedCARLocator::getCARLocation() {
	boundingBox = false;
	std::vector<cv::Vec4i> lines;
	HoughLinesP(mprImg, lines, 1, CV_PI/ 2, 40, 40, 10 );
	int minX = 0;
	int maxX = mprImg.cols - 1;
	int minY = 0;
	int maxY = mprImg.rows - 1;
	//find vertical line
#ifdef DEBUG
	Mat cdst;
	cvtColor(this->mprImg, cdst, CV_GRAY2BGR);
#endif
	for( size_t i = 0; i < lines.size(); i++ ) {
	  Vec4i l = lines[i];
#ifdef DEBUG
	  line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0, 255), 1, CV_AA);
#endif
	  if (l[0] == l[2] && std::abs(l[1] - l[3]) > mprImg.rows * 0.3) {
		  if (l[0] < mprImg.cols / 10) {
			  minX = std::max(l[0], minX);
			  boundingBox = true;
		  } else if (l[0] >= mprImg.cols * 9 / 10) {
			  maxX = std::min(l[0], maxX);
			  boundingBox = true;
		  }
	  } else if (l[1] == l[3] && std::abs(l[0] - l[2]) > mprImg.cols * 0.3) {
		  if (l[1] < mprImg.rows / 4) {
			  minY = std::max(l[1], minY);
		  } else if (l[1] >= mprImg.rows * 3 / 4) {
			  maxY = std::min(l[1], maxY);
		  }
	  }
	}
	Rect car(minX, minY, maxX - minX + 1, maxY - minY + 1);
	//remove guideline
//	removeBaseline(mprImg, lines, maxY);
#ifdef DEBUG
	cv::rectangle(cdst, car, CV_RGB(0,255,0));
	imshow("car", cdst);
	imshow("mprImg", mprImg);
#endif
	return car;
}

cv::Rect PrintedCARLocator::getRMLocation(Blobs& blobs, cv::Rect& carLoc) {
	Rect box = carLoc;
	int blobWidthThres = box.height / 2;
	if (blobWidthThres < 70) { //blobWidthThres must at least 70
		blobWidthThres = 70;
	}
#define PADDING_HEIGHT 0.2
	box.y += box.height * PADDING_HEIGHT;
	box.height *= 1 - PADDING_HEIGHT * 2;
	box.width = box.height;
#ifdef DEBUG
	Mat rmImg(mprImg.size(), CV_8UC3);
	drawBlobs(blobs, rmImg);
	cv::rectangle(rmImg, box, CV_RGB(0,255,0));
	imshow("rmImg", rmImg);
#endif
	//select blob within box
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		Rect rect = b->boundingRect();
		Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
		if (!box.contains(center) || rect.width > blobWidthThres || rect.height >= blobWidthThres) {
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
	int biggestIdx = blobs.findBiggestBlob();
	int secIdx = blobs.findBiggestBlob([=](int i) -> bool {return i != biggestIdx;});
	if (secIdx == -1 || blobs[secIdx]->points.size() / ((float) blobs[biggestIdx]->points.size()) < 0.5 ) {
		if (biggestIdx == -1) {
			std::cout << "missing M" << std::endl;
			return rs;
		}
		auto brect = blobs[biggestIdx]->boundingRect();
		float aspect = brect.height /(float) brect.width;
		if (aspect > 0.45 && aspect < 0.8) {
			rs = brect;
		} else {
			std::cout << "missing M" << std::endl;
		}
	} else {
		rs = boundingRect(blobs[biggestIdx]->boundingRect(), blobs[secIdx]->boundingRect());
	}
	return rs;
}

} /* namespace icr */
