/*
 * BFSegment.cpp
 *
 *  Created on: Sep 30, 2014
 *      Author: thienlong
 */

#include <BFSegment.h>
#include <opencv2/highgui/highgui.hpp>

#include "preprocessor.h"

void findSEPoint(const Mat& thinned, Point& start, Point& end);
void foregroundTFP(Mat& thinned, Point start, Point end, vector<Point>& output);
void foregroundBFP(Mat& thinned, Point start, Point end, vector<Point>& output);
void backgroundTFP(Mat& src, vector<Point>& output);
void backgroundBFP(Mat& src, vector<Point>& output);

BFSegment::BFSegment(Mat& binImg) : binImg(binImg) {
	Mat temp = binImg / 255;
	Blobs blobs = findBlobs(temp);
	broken = blobs.size() != 1;
}

void BFSegment::genFPLayers() {
	if (broken) {
		return;
	}
	Mat fgSkeleton;
	thinning(binImg, fgSkeleton);
	imshow("skeleton", fgSkeleton);
	Point start, end;
	findSEPoint(fgSkeleton, start, end);
	backgroundTFP(binImg, fpLayers[0]);
	foregroundTFP(fgSkeleton, start, end, fpLayers[1]);
	foregroundBFP(fgSkeleton, start, end, fpLayers[2]);
	backgroundBFP(binImg, fpLayers[3]);
}


#define ALPHA 0.25
int findMatch(Point p, vector<Point> fps, int index, int maxDigitWidth, int blobHeight) {
	for (; index < fps.size(); ++index) {
		if (fps[index].x > maxDigitWidth) {
			break;
		} else if (std::abs(fps[index].x - p.x) <= ALPHA * blobHeight) {
			return index;
		}
	}
	return -1;
}

void genVertCut(Point p, int row, vector<Point>& cut) {
	int minY = std::min(p.y, row);
	int maxY = std::max(p.y, row);
	for (int i = minY; i <= maxY; ++i) {
		cut.push_back(Point(p.x, i));
	}
}

void genVertCut(int x, int blobHeight, vector<Point>& cut) {
	for (int i = 0; i < blobHeight; ++i) {
		cut.push_back(Point(x, i));
	}
}

void genP2PCut(Point pStart, Point pEnd, vector<Point>& cut) {
	int x0 = pStart.x;
	int y0 = pStart.y;
	int x1 = pEnd.x;
	int y1 = pEnd.y;
	int dx = abs(x1-x0), sx = x0<x1 ? 1 : -1;
	  int dy = abs(y1-y0), sy = y0<y1 ? 1 : -1;
	  int err = (dx>dy ? dx : -dy)/2, e2;

	  for(;;){
	    cut.push_back(Point(x0,y0));
	    if (x0==x1 && y0==y1) break;
	    e2 = err;
	    if (e2 >-dx) { err -= dy; x0 += sx; }
	    if (e2 < dy) { err += dx; y0 += sy; }
	  }
}

void BFSegment::genCutPath(Point p, vector<Point>& cut, int layerIdx, int maxDigitWidth, vector<vector<Point> >& cutPaths, bool down) {
	int layerIdxEnd = down ? 4 : -1;
	int incrLayerIdx = down ? 1 : -1;
	int index = 0;
	bool noMore = (layerIdx == layerIdxEnd) || (index = findMatch(p, fpLayers[layerIdx], 0, maxDigitWidth, binImg.rows)) == -1;
	if (noMore) {
		int rowIdx = down ? binImg.rows - 1 : 0;
		vector<Point> newCut(cut.begin(), cut.end());
		genVertCut(p, rowIdx, newCut);
		cutPaths.push_back(newCut);
		return;
	}
	auto& fps = fpLayers[layerIdx];
	do {
		vector<Point> newCut(cut.begin(), cut.end());
		genP2PCut(p, fps[index], newCut);
		genCutPath(fps[index], newCut, layerIdx + incrLayerIdx, maxDigitWidth, cutPaths, down);
		index = findMatch(p, fps, index + 1, maxDigitWidth, binImg.rows);
	} while (index != -1);
}

void BFSegment::genCutPath(int maxDigitWidth, vector<vector<Point> >& cutPaths, bool down) {
	//downward path searching
	int layerIdx = down ? 0 : 3;
	int incrLayerIdx = down ? 1 : -1;
	int rowIdx = down ? 0 : binImg.rows - 1;
	for (Point p : fpLayers[layerIdx]) {
		if (p.x > maxDigitWidth) {
			break;
		}
		vector<Point> cut;
		genVertCut(p, rowIdx, cut);
		genCutPath(p, cut, layerIdx + incrLayerIdx, maxDigitWidth, cutPaths, down);
	}
}

void BFSegment::genCutPath(int maxDigitWidth, vector<vector<Point>>& cutPaths) {
	genCutPath(maxDigitWidth, cutPaths, true);
	genCutPath(maxDigitWidth, cutPaths, false);
}

BFSegment::~BFSegment() {
}


void findSEPoint(const Mat& src, Point& start, Point& end) {
	//find start point
	for (int c = 0; c < src.cols; ++c) {
		for (int r = 0; r < src.rows; ++r) {
			if (src.at<uchar>(r, c) > 0) {
				start.x = c;
				start.y = r;
				goto end;
			}
		}
	}
	end: for (int c = src.cols - 1; c >= 0; --c) {
		for (int r = 0; r < src.rows; ++r) {
			if (src.at<uchar>(r, c) > 0) {
				end.x = c;
				end.y = r;
				return;
			}
		}
	}
}

inline bool moveTopNext(const Mat& thinned, Point& blackMove, Point& whiteMove) {
	int relative = (whiteMove.y - blackMove.y) * 3 + (whiteMove.x - blackMove.x);
	switch(relative) {
	case 1: {
		if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			whiteMove.y -= 1;
		} else if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x)) {
			whiteMove.y -= 1;
			blackMove.y -= 1;
		} else if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x + 1)) {
			whiteMove.x += 1;
			whiteMove.y -= 1;
			blackMove.x += 1;
			blackMove.y -= 1;
		} else {
			blackMove.x += 1;
			blackMove.y -= 1;
			return false;
		}
		break;
	}
	case 3: {
		if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x + 1)) {
			//i think it never occur
			whiteMove.x += 1;
			whiteMove.y -= 1;
		} else if (thinned.at<uchar>(whiteMove.y, whiteMove.x + 1)) {
			whiteMove.x += 1;
			blackMove.x += 1;
		} else if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x + 1)) {
			whiteMove.x += 1;
			whiteMove.y += 1;
			blackMove.x += 1;
			blackMove.y += 1;
		} else {
			blackMove.x += 1;
			blackMove.y += 1;
			return false;
		}
		break;
	}
	case -1: {
		if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x + 1)) {
			//i think it never occur
			whiteMove.x += 1;
			whiteMove.y += 1;
		} else if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x)) {
			whiteMove.y += 1;
			blackMove.y += 1;
		} else if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			whiteMove.y += 1;
			blackMove.x -= 1;
			blackMove.y += 1;
		} else {
			blackMove.x -= 1;
			blackMove.y += 1;
			return false;
		}
		break;
	}
	case -3: {
		if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x - 1)) {
			//i think it never occur
			whiteMove.x -= 1;
			whiteMove.y += 1;
		} else if (thinned.at<uchar>(whiteMove.y, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			blackMove.x -= 1;
		} else if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			whiteMove.y -= 1;
			blackMove.x -= 1;
			blackMove.y -= 1;
		} else {
			blackMove.x -= 1;
			blackMove.y -= 1;
			return false;
		}
		break;
	}
	}
	return true;
}


inline bool moveBotNext(const Mat& thinned, Point& blackMove, Point& whiteMove) {
	int relative = (whiteMove.y - blackMove.y) * 3 + (whiteMove.x - blackMove.x);
	switch(relative) {
	case 1: {
		if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			whiteMove.y += 1;
		} else if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x)) {
			whiteMove.y += 1;
			blackMove.y += 1;
		} else if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x + 1)) {
			whiteMove.x += 1;
			whiteMove.y += 1;
			blackMove.x += 1;
			blackMove.y += 1;
		} else {
			blackMove.x += 1;
			blackMove.y += 1;
			return false;
		}
		break;
	}
	case -3: {
		if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x + 1)) {
			//i think it never occur
			whiteMove.x += 1;
			whiteMove.y += 1;
		} else if (thinned.at<uchar>(whiteMove.y, whiteMove.x + 1)) {
			whiteMove.x += 1;
			blackMove.x += 1;
		} else if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x + 1)) {
			whiteMove.x += 1;
			whiteMove.y -= 1;
			blackMove.x += 1;
			blackMove.y -= 1;
		} else {
			blackMove.x += 1;
			blackMove.y -= 1;
			return false;
		}
		break;
	}
	case -1: {
		if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x + 1)) {
			//i think it never occur
			whiteMove.x += 1;
			whiteMove.y -= 1;
		} else if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x)) {
			whiteMove.y -= 1;
			blackMove.y -= 1;
		} else if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			whiteMove.y -= 1;
			blackMove.x -= 1;
			blackMove.y -= 1;
		} else {
			blackMove.x -= 1;
			blackMove.y -= 1;
			return false;
		}
		break;
	}
	case 3: {
		if (thinned.at<uchar>(whiteMove.y - 1, whiteMove.x - 1)) {
			//i think it never occur
			whiteMove.x -= 1;
			whiteMove.y -= 1;
		} else if (thinned.at<uchar>(whiteMove.y, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			blackMove.x -= 1;
		} else if (thinned.at<uchar>(whiteMove.y + 1, whiteMove.x - 1)) {
			whiteMove.x -= 1;
			whiteMove.y += 1;
			blackMove.x -= 1;
			blackMove.y += 1;
		} else {
			blackMove.x -= 1;
			blackMove.y += 1;
			return false;
		}
		break;
	}


	}
	return true;
}

inline bool isIP(const Mat& thinned, const Point& p) {
	int counter = 0;
	int m[8];
	m[0] = thinned.at<uchar>(p.y - 1, p.x - 1);
	m[1] = thinned.at<uchar>(p.y -1, p.x);
	m[2] = thinned.at<uchar>(p.y -1, p.x + 1);
	m[3] = thinned.at<uchar>(p.y, p.x + 1);
	m[4] = thinned.at<uchar>(p.y + 1, p.x + 1);
	m[5] = thinned.at<uchar>(p.y + 1, p.x);
	m[6] = thinned.at<uchar>(p.y + 1, p.x - 1);
	m[7] = thinned.at<uchar>(p.y, p.x - 1);
	if (m[7] > 0 && m[0] == 0) ++counter;
	for (int i = 0; i < 7; ++i) {
		if (m[i] > 0 && m[i + 1] == 0) ++counter;
	}
	if (counter >= 3) {
		return true;
	}
	counter = 0;
	for (int i = 0; i < 8; ++i)
		counter += m[i] / 255;
	if (counter < 4) {
		return false;
	}
	for (int i = 0; i < 6; ++i) {
		if (m[i] > 0 && m[i + 1] > 0 && m[i + 2] > 0) {
			return true;
		}
	}
	return (m[6] > 0 && m[7] > 0 && m[0] > 0) || (m[1] > 0 && m[7] > 0 && m[0] > 0);
}
#define ANGLE_EDGE_LENGTH 5
Point findFP(vector<Point> whiteTrack, int ipIdx) {
	Point end = whiteTrack[whiteTrack.size() - 1];
	Point ip = whiteTrack[ipIdx];
	int startIdx = ipIdx - ANGLE_EDGE_LENGTH;
	Point start = startIdx >= 0 ? whiteTrack[startIdx] : whiteTrack[0];
	int x = (start.x + end.x) / 2;
	int y = (start.y + end.y) / 2;
	return Point(x, y);
}

void foregroundTFP(Mat& thinned, Point start, Point end, vector<Point>& output) {
	Point whiteMove = start;
	Point blackMove(start.x -1, start.y);
	vector<Point> blackTrack;
	blackTrack.push_back(blackMove);
	int counter = 0;
	int ipIdx = 0;
	while (whiteMove != end) {
		bool move = moveTopNext(thinned, blackMove, whiteMove);
		blackTrack.push_back(blackMove);
		if (counter > 0) {
			counter = (counter + 1) % ANGLE_EDGE_LENGTH;
			if (counter == 0) {
				auto p = findFP(blackTrack, ipIdx);
				output.push_back(p);
			}
		} else if (move && isIP(thinned, whiteMove)) {
			ipIdx = blackTrack.size() - 1;
			++counter;
		}
	}
}

void foregroundBFP(Mat& thinned, Point start, Point end, vector<Point>& output) {
	Point whiteMove = start;
	Point blackMove(start.x -1, start.y);
	vector<Point> blackTrack;
	blackTrack.push_back(blackMove);
	int counter = 0;
	int ipIdx = 0;
	while (whiteMove != end) {
		bool move = moveBotNext(thinned, blackMove, whiteMove);
		blackTrack.push_back(blackMove);
		if (counter > 0) {
			counter = (counter + 1) % ANGLE_EDGE_LENGTH;
			if (counter == 0) {
				auto p = findFP(blackTrack, ipIdx);
				output.push_back(p);
			}
		} else if (move && isIP(thinned, whiteMove)) {
			ipIdx = blackTrack.size() - 1;
			++counter;
		}
	}
}

bool isEndPoint(Mat& src, int r, int c) {
	if (src.at<uchar>(r, c) == 0) {
		return false;
	}
	int sum = src.at<uchar>(r - 1, c - 1)
			+ src.at<uchar>(r - 1, c)
			+ src.at<uchar>(r - 1, c + 1)
			+ src.at<uchar>(r, c + 1)
			+ src.at<uchar>(r + 1, c + 1)
			+ src.at<uchar>(r + 1, c)
			+ src.at<uchar>(r + 1, c - 1)
			+ src.at<uchar>(r, c - 1);
	return sum == 255;
}

void findEndPoint(Mat& thinned, vector<Point>& output) {
	for (int c = 1; c < thinned.cols - 1; ++c) {
		for (int r = 1; r < thinned.rows - 1; ++r) {
			if (isEndPoint(thinned, r, c)) {
				output.push_back(Point(c, r));
			}
		}
	}
}

void backgroundTFP(Mat& src, vector<Point>& output) {
	Mat bgTop(src.rows, src.cols, CV_8UC1, Scalar(255));
	for (int c = 0; c < src.cols; ++c) {
		bool setPixel = false;
		for (int r = 0; r < src.rows; ++r) {
			if (setPixel) {
				bgTop.at<uchar>(r, c) = 0;
			} else if (src.at<uchar>(r, c) > 0) {
				setPixel = true;
				bgTop.at<uchar>(r, c) = 0;
			}
		}
	}
	thinning(bgTop, bgTop);
	imshow("bgtop", bgTop);
	findEndPoint(bgTop, output);
}

void backgroundBFP(Mat& src, vector<Point>& output) {
	Mat bgBot(src.rows, src.cols, CV_8UC1, Scalar(255));
	for (int c = 0; c < src.cols; ++c) {
		bool setPixel = false;
		for (int r = src.rows - 1; r >= 0 ; --r) {
			if (setPixel) {
				bgBot.at<uchar>(r, c) = 0;
			} else if (src.at<uchar>(r, c) > 0) {
				setPixel = true;
				bgBot.at<uchar>(r, c) = 0;
			}
		}
	}
//	bgBot.col(0).setTo(Scalar(0));
//	bgBot.col(src.cols - 1).setTo(Scalar(0));
	thinning(bgBot, bgBot);
	imshow("bgbot", bgBot);
	findEndPoint(bgBot, output);
//	output.erase(output.begin());
//	output.pop_back();
}

#include <opencv2/opencv.hpp>
int bfmain()
{
	cv::Mat src = cv::imread("/media/thienlong/linux/CAR/cvl-strings/train/25000-0164-08.png");
	if (!src.data)
		return -1;

	cv::Mat bw, binary;
	cv::cvtColor(src, bw, CV_BGR2GRAY);
//	src = cv::Scalar::all(255) -src;
	cv::threshold(bw, binary, 10, 255, CV_THRESH_OTSU | CV_THRESH_BINARY_INV);
	cv::Rect roi = getROI(binary);
	Mat cropB = binary(roi);
	int padding = 1;
	cv::copyMakeBorder(cropB, binary, padding, padding, padding , padding, BORDER_CONSTANT, cv::Scalar(0));
	cv::imshow("binary", binary);
	BFSegment segmenter(binary);
	segmenter.genFPLayers();
	vector<vector<Point>> cuts;
	segmenter.genCutPath(binary.rows, cuts);
	for (auto cut : cuts) {
		Mat srcClone = binary.clone();
		drawCut(srcClone, cut);
		//	cv::destroyAllWindows();
		imshow("cut-source" + std::to_string(clock()), srcClone);
	}
	cv::imshow("src", binary);
	cv::waitKey();
	return 0;
}
