/*
 * bfground-segment.cpp
 *
 *  Created on: Sep 6, 2014
 *      Author: thienlong
 */

#include <opencv2/opencv.hpp>
#include <vector>

#include "opencv2/core/core.hpp"

using namespace cv;

void findSEPoint(const Mat& thinned, Point& start, Point& end) {
	//find start point
	for (int c = 0; c < thinned.cols; ++c) {
		for (int r = 0; r < thinned.rows; ++r) {
			if (thinned.at<uchar>(r, c) > 0) {
				start.x = c;
				start.y = r;
				goto end;
			}
		}
	}
	end: for (int c = thinned.cols - 1; c >= 0; --c) {
		for (int r = 0; r < thinned.rows; ++r) {
			if (thinned.at<uchar>(r, c) > 0) {
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

void findTopFeaturePoint(Mat& cc, Mat& thinned, Point start, Point end, vector<Point>& output) {
	Point whiteMove = start;
	Point blackMove(start.x -1, start.y);
	Mat view = Mat::zeros(thinned.size(), thinned.type());
	view.at<uchar>(blackMove) = 255;
	vector<Point> whiteTrack;
	whiteTrack.push_back(whiteMove);
	while (whiteMove != end) {
		imshow("findTFP", view);
		char c = waitKey(10);
		if (c == 's') {
			std::cout << "white move: "<< whiteMove.x <<"x" <<whiteMove.y<<std::endl;
		}
		if(moveTopNext(thinned, blackMove, whiteMove)) {
			whiteTrack.push_back(whiteMove);
			if (isIP(thinned, whiteMove)) {
				output.push_back(whiteMove);
				std::cout << whiteMove.x <<"x" <<whiteMove.y<<std::endl;
			}
		}
		view.at<uchar>(blackMove) = 255;
	}
}

void findBottomFeaturePoint(Mat& cc, Mat& thinned, Point start, Point end, vector<Point>& output) {
	Point whiteMove = start;
	Point blackMove(start.x -1, start.y);
	Mat view = Mat::zeros(thinned.size(), thinned.type());
	view.at<uchar>(blackMove) = 255;
	while (whiteMove != end) {
		imshow("findBFP", view);
		char c = waitKey(10);
		if (c == 's') {
			std::cout << "white move: "<< whiteMove.x <<"x" <<whiteMove.y<<std::endl;
		}
		if(moveBotNext(thinned, blackMove, whiteMove)) {
			if (isIP(thinned, whiteMove)) {
				output.push_back(whiteMove);
				std::cout << whiteMove.x <<"x" <<whiteMove.y<<std::endl;
			}
		}
		view.at<uchar>(blackMove) = 255;
	}
}


