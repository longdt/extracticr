/*
 * bfground-segment.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: thienlong
 */

#include <opencv2/opencv.hpp>
#include <vector>

#include "opencv2/core/core.hpp"

#include "preprocessor.h"

using namespace cv;

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

void foregroundTFP(Mat& cc, Mat& thinned, Point start, Point end, vector<Point>& output) {
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

void foregroundBFP(Mat& cc, Mat& thinned, Point start, Point end, vector<Point>& output) {
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
	for (int c = 0; c < thinned.cols; ++c) {
		for (int r = 0; r < thinned.rows; ++r) {
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
	thinning(bgBot, bgBot);
	findEndPoint(bgBot, output);
}


