#include <opencv2/core/core.hpp>
#include <algorithm>

using namespace cv;

void dropfallLeft(const Mat& src, std::vector<cv::Point2i >& cut, int start, bool top) {
	cut.clear();
	int i = top ? 0 : src.rows - 1;
	cut.push_back(Point2i(start, i));
	int step = top ? 1 : -1;
	bool mvLeft = false;
	bool mvRight = false;
	while (true) {
		if (src.at<uchar>(i + step, start) == 0) { //if down is white, then move down
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		else if (src.at<uchar>(i + step, start + 1) == 0) { //if down-right, move down-right
			if (mvRight) {
				mvRight = false;
			}
			cut.push_back(Point2i(start + 1, i + step));
			i += step;
			++start;
		}
		else if (src.at<uchar>(i + step, start - 1) == 0) { //if down-left, move down-left
			if (mvLeft) {
				mvLeft = false;
			}
			cut.push_back(Point2i(start - 1, i + step));
			i += step;
			--start;
		}
		else if (src.at<uchar>(i, start + 1) == 0 && !mvLeft) { //if right, move right
			mvRight = true;
			++start;
		}
		else if (src.at<uchar>(i, start - 1) == 0) { //if left, move left
			mvLeft = true;
			--start;
		}
		else {	//cut
			mvLeft = false;
			mvRight = false;
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		if ((top && i >= src.rows - 1) || (!top && i <= 0)) {
			cut.push_back(Point2i(start, i));
			break;
		}
		else if (start == 0 || start == src.cols - 1) {
			return;
		}
	}

}



void dropfallRight(const Mat& src, std::vector<cv::Point2i >& cut, int start, bool top) {
	cut.clear();
	int i = top ? 0 : src.rows - 1;
	int step = top ? 1 : -1;
	bool mvLeft = false;
	bool mvRight = false;
	while (true) {
		if (src.at<uchar>(i + step, start) == 0) { //if down is white, then move down
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		else if (src.at<uchar>(i + step, start - 1) == 0) { //if down-left, move down-left
			if (mvLeft) {
				mvLeft = false;
			}
			cut.push_back(Point2i(start - 1, i + step));
			i += step;
			--start;
		}
		else if (src.at<uchar>(i + step, start + 1) == 0) { //if down-right, move down-right
			if (mvRight) {
				mvRight = false;
			}
			cut.push_back(Point2i(start + 1, i + step));
			i += step;
			++start;
		}
		else if (src.at<uchar>(i, start - 1) == 0 && !mvRight) { //if left, move left
			mvLeft = true;
			--start;
		}
		else if (src.at<uchar>(i, start + 1) == 0) { //if right, move right
			mvRight = true;
			++start;
		}
		else {	//cut
			mvLeft = false;
			mvRight = false;
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		if ((top && i >= src.rows - 1) || (!top && i <= 0)) {
			cut.push_back(Point2i(start, i));
			break;
		}
		else if (start == 0 || start == src.cols - 1) {
			return;
		}
	}
}



void dropfallExtLeft(const Mat& src, std::vector<cv::Point2i >& cut, int start, bool top) {
	cut.clear();
	int i = top ? 0 : src.rows - 1;
	cut.push_back(Point2i(start, i));
	int step = top ? 1 : -1;
	bool mvLeft = false;
	bool mvRight = false;
	while (true) {
		if (src.at<uchar>(i, start) > 0) { //if is cut mode
			if (src.at<uchar>(i + step, start) > 0) {
				cut.push_back(Point2i(start, i + step));
				i += step;
			}
			else if (src.at<uchar>(i + step, start + 1) > 0) {
				cut.push_back(Point2i(start + 1, i + step));
				i += step;
				++start;
			}
			else if (src.at<uchar>(i + step, start - 1) > 0) {
				cut.push_back(Point2i(start - 1, i + step));
				i + step;
				--start;
			}
			else {
				cut.push_back(Point2i(start, i + step));
				i += step;
			}
		}
		else if (src.at<uchar>(i + step, start) == 0) { //if down is white, then move down
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		else if (src.at<uchar>(i + step, start + 1) == 0) { //if down-right, move down-right
			if (mvRight) {
				mvRight = false;
			}
			cut.push_back(Point2i(start + 1, i + step));
			i += step;
			++start;
		}
		else if (src.at<uchar>(i + step, start - 1) == 0) { //if down-left, move down-left
			if (mvLeft) {
				mvLeft = false;
			}
			cut.push_back(Point2i(start - 1, i + step));
			i += step;
			--start;
		}
		else if (src.at<uchar>(i, start + 1) == 0 && !mvLeft) { //if right, move right
			mvRight = true;
			++start;
		}
		else if (src.at<uchar>(i, start - 1) == 0) { //if left, move left
			mvLeft = true;
			--start;
		}
		else {	//cut
			mvLeft = false;
			mvRight = false;
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		if ((top && i >= src.rows - 1) || (!top && i <= 0)) {
			cut.push_back(Point2i(start, i));
			break;
		}
		else if (start == 0 || start == src.cols - 1) {
			return;
		}
	}

}



void dropfallExtRight(const Mat& src, std::vector<cv::Point2i >& cut, int start, bool top) {
	cut.clear();
	int i = top ? 0 : src.rows - 1;
	cut.push_back(Point2i(start, i));
	int step = top ? 1 : -1;
	bool mvLeft = false;
	bool mvRight = false;
	while (true) {
		if (src.at<uchar>(i, start) > 0) { //if is cut mode
			if (src.at<uchar>(i + step, start) > 0) {
				cut.push_back(Point2i(start, i + step));
				i += step;
			}
			else if (src.at<uchar>(i + step, start - 1) > 0) {
				cut.push_back(Point2i(start - 1, i + step));
				i + step;
				--start;
			}
			else if (src.at<uchar>(i + step, start + 1) > 0) {
				cut.push_back(Point2i(start + 1, i + step));
				i += step;
				++start;
			}
			else {
				cut.push_back(Point2i(start, i + step));
				i += step;
			}
		}
		else if (src.at<uchar>(i + step, start) == 0) { //if down is white, then move down
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		else if (src.at<uchar>(i + step, start - 1) == 0) { //if down-left, move down-left
			if (mvLeft) {
				mvLeft = false;
			}
			cut.push_back(Point2i(start - 1, i + step));
			i += step;
			--start;
		}
		else if (src.at<uchar>(i + step, start + 1) == 0) { //if down-right, move down-right
			if (mvRight) {
				mvRight = false;
			}
			cut.push_back(Point2i(start + 1, i + step));
			i += step;
			++start;
		}
		else if (src.at<uchar>(i, start - 1) == 0 && !mvRight) { //if left, move left
			mvLeft = true;
			--start;
		}
		else if (src.at<uchar>(i, start + 1) == 0) { //if right, move right
			mvRight = true;
			++start;
		}
		else {	//cut
			mvLeft = false;
			mvRight = false;
			cut.push_back(Point2i(start, i + step));
			i += step;
		}
		if ((top && i >= src.rows - 1) || (!top && i <= 0)) {
			cut.push_back(Point2i(start, i));
			break;
		}
		else if (start == 0 || start == src.cols - 1) {
			return;
		}
	}

}