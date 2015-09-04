/*
 * PhCARLocator.cpp
 *
 *  Created on: Jun 10, 2015
 *      Author: thienlong
 */

#include <opencv2/opencv.hpp>
#include "CARLocator.h"

#include "preprocessor.h"

namespace icr {
using namespace cv;
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

bool hasAboveStroke(Mat& src, int row, int col, int boxHeight) {
	for (int rowEnd = row - 0.8 * boxHeight; row > rowEnd; --row) {
		if (src.at<uchar>(row, col) > 0) {
			return true;
		}
	}
	return false;
}

bool hasAboveStroke(Mat& src, int row, int startCol, int endCol, int boxHeight) {
	for (int col = startCol; col <= endCol; ++col) {
		if (!hasAboveStroke(src, row, col, boxHeight)) {
			return false;
		}
	}
	return true;
}

void fillUStrokes(Mat& src, int lt, int row, std::vector<int>& strokes, int boxHeight) {
	if (strokes.size() < 4) {
		return;
	}
	for (int i = 1; i < strokes.size() - 1; i += 2) {
		int x = strokes[i + 1] - strokes[i];
		if (x > boxHeight / 3) {
			continue;
		}
		int y = floor((2 * lt + 2.0) * x / (2 * lt + 1.0));
		int k = distanceK(src, row, strokes[i], strokes[i + 1]);
//		std::cout << k << std::endl;
		int center = (strokes[i] + strokes[i + 1]) / 2;
		if (y <= k + 2 && y >= k -2 && hasAboveStroke(src, row, strokes[i], strokes[i + 1], boxHeight)) {
			line(src, Point(strokes[i] - 2, row), Point(strokes[i + 1] + 2, row), Scalar(255, 255, 255), 1, CV_AA);
			line(src, Point(strokes[i] - 2, row + 1), Point(strokes[i + 1] + 2, row + 1), Scalar(255, 255, 255), 1, CV_AA);
		}
	}
}

void removeBaseline(Mat& mprImg, std::vector<cv::Vec4i>& lines, int maxY, int minY) {
	--maxY;
	Rect r(0, maxY, mprImg.cols, mprImg.rows - maxY);
	mprImg(r) = Scalar::all(0);
//	removeNoise(mprImg);
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
	fillUStrokes(mprImg, lt.size(), l1, strokes, maxY - minY);
}

void correctCAR(Mat& mprImg, Rect& car) {
	Mat roi = mprImg(car);
	vector<int> projectV;
	projectVertical1(roi, projectV);
	int startX = 0;
	int endX = car.width / 4;
	for (int thres = car.height / 2; startX < endX && projectV[startX] < thres; ++startX) {}
	if (startX != endX) {
		car.x += startX;
		car.width = car.width - startX;
	}
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
		  if (l[0] < mprImg.cols / 4) {
			  minX = std::max(l[0], minX);
			  boundingBox = true;
		  } else if (l[0] > mprImg.cols * 3 / 4) {
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
	if (!boundingBox && car.height < mprImg.rows / 1.8f) {
		boundingBox = true;
	}
	if (boundingBox && minX == 0) {
		correctCAR(mprImg, car);
	}
	//remove guideline
	removeBaseline(mprImg, lines, maxY, minY);
#ifdef DEBUG
	cv::rectangle(cdst, car, CV_RGB(0,255,0));
	imshow("car", cdst);
	imshow("mprImg", mprImg);
#endif
	return car;
}

void PhCARLocator::getHandwrittenBlobs(Blobs& blobs) {
	Rect box = getCARLocation();
	box.height += 4;
	Mat bin;
	bin = mprImg / 255;
	findBlobs(bin, blobs);
	removeNoise(blobs);
	Blobs rmBlobs;
	for (size_t i = 0; i < blobs.size(); ++i) {
		blobs[i]->boundingRect();
	}
	blobs.clone(rmBlobs);
	Rect rm = getRMLocation(rmBlobs, box);
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
		} else if (rect.width / (float) rect.height > 20 && rect.height < 8) { //blob too long
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
}
