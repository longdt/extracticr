
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "recognizer.h"
#include "util/misc.h"
#include "preprocessor.h"


using namespace tiny_cnn;
using namespace cv;


extern digit_recognizer recognizer;

std::string recognizeND(Mat& src, double& srcConf);

void drawCut(Mat& src, vector<Point> &cut) {
	for (auto p : cut) {
		src.at<uchar>(p) = 100;
	}
}

Mat makeCut(Mat& src, vector<Point> &cut) {
	Mat other = Mat::zeros(src.rows, src.cols, src.type());
	for (auto p : cut) {
		for (int j = 0; j < p.x; ++j) {
			other.at<uchar>(p.y, j) = src.at<uchar>(p.y, j);
			src.at<uchar>(p.y, j) = 0;
		}
	}
	return other;
}

int recognize1D(Mat& src, double& conf) {
	auto digitMat = makeDigitMat(src);
	vec_t in;
	mat_to_vect(digitMat, in);
	int label = recognizer.predict(in, &conf);
	return label;
}

std::string tryGuest(const Mat& src, vector<Point>& cut, double& conf) {
	Mat part2 = src.clone();
	Mat other = makeCut(part2, cut);
	Mat temp;
	if (!cropMat(other, temp) || temp.rows <= src.rows * 0.6) {
		return "";
	}
	other = makeDigitMat(temp);
	Mat temp2;
	if (!cropMat(part2, temp2) || temp2.rows <= src.rows * 0.6) {
		return "";
	}
	Mat cutSrc = src.clone();
	drawCut(cutSrc, cut);
	imshow("cut source**", cutSrc);
	imshow("temp", temp2);
	part2 = makeDigitMat(temp2);
	vec_t in;
	double otherConf;
	mat_to_vect(other, in);
	int label = recognizer.predict(in, &otherConf);
	double conf2;
	mat_to_vect(part2, in);
	int label2 = recognizer.predict(in, &conf2);
	/*imshow(std::to_string(label) + "part1." + std::to_string(otherConf), other);
	imshow(std::to_string(label2) + "part2." + std::to_string(conf2), part2);
	imshow("source***", src);
	waitKey(0);*/
	if ((otherConf < 0 || conf2 < 0) || (otherConf * conf2 < conf)) {
		return "";
	}
	conf = otherConf * conf2;
	return std::to_string(label) + std::to_string(label2);
}

std::string recognize2D(Mat& src, double& srcConf) {
	int start = (int) (src.cols * 0.375);
	vector<Point> cut;
	dropfallLeft(src, cut, start, true);
	double conf[4];
	for (int i = 0; i < 4; ++i) {
		conf[i] = srcConf;
	}
	std::string val[4];
	val[0] = tryGuest(src, cut, conf[0]);
	if (val[0].empty()) {
		conf[0] = -1;
	}

	dropfallLeft(src, cut, start, false);
	val[1] = tryGuest(src, cut, conf[1]);
	if (val[1].empty()) {
		conf[1] = -1;
	}

	start = src.cols - 1 - start;
	dropfallRight(src, cut, start, true);
	val[2] = tryGuest(src, cut, conf[2]);
	if (val[2].empty()) {
		conf[2] = -1;
	}

	dropfallRight(src, cut, start, false);
	val[3] = tryGuest(src, cut, conf[3]);
	if (val[3].empty()) {
		conf[3] = -1;
	}
	auto it = std::max_element(conf, conf + 4);
	srcConf = *it;
	int index = it - conf;
	return val[index];
}

std::string tryGuestND(const Mat& src, vector<Point>& cut, double& conf) {
	Mat part2 = src.clone();
	Mat part1 = makeCut(part2, cut);
	Mat cropedPart1;
	if (!cropMat(part1, cropedPart1) || cropedPart1.rows <= src.rows * 0.6) {
		return "";
	}
	double conf1;
	int label = recognize1D(cropedPart1, conf1);
	Mat cropedPart2;
	if (!cropMat(part2, cropedPart2, 1) || cropedPart2.rows <= src.rows * 0.6) {
		return "";
	}
	
	double confPart2;
	int labelPart2 = recognize1D(cropedPart2, confPart2);
	//cv::destroyAllWindows();
	//imshow(std::to_string(label) + "part1." + std::to_string(conf1), cropedPart1);
	//imshow(std::to_string(labelPart2) + "part2." + std::to_string(confPart2), cropedPart2);
	//imshow("source", src);
	//waitKey(0);
	double confND2 = confPart2;
	std::string label2 = recognizeND(cropedPart2, confND2);
	if (confND2 < confPart2) {
		confND2 = confPart2;
		label2 = std::to_string(labelPart2);
	}

	if ((conf1 < 0 || confND2 < 0) || (conf1 * confND2 < conf)) {
		return "";
	}
	conf = conf1 * confND2;
	return std::to_string(label) + label2;
}

std::string recognizeND(Mat& src, double& srcConf) {
	if (src.rows / src.cols > 2) {
		return "";
	}
	int start = (int)(src.cols * 0.375);
	double conf[4];
	for (int i = 0; i < 4; ++i) {
		conf[i] = srcConf;
	}
	std::string val[4];

	vector<Point> cut;
	dropfallLeft(src, cut, start, true);
	val[0] = tryGuestND(src, cut, conf[0]);
	if (val[0].empty()) {
		conf[0] = -1;
	}

	dropfallLeft(src, cut, start, false);
	val[1] = tryGuestND(src, cut, conf[1]);
	if (val[1].empty()) {
		conf[1] = -1;
	}

	start = src.cols - 1 - start;
	dropfallRight(src, cut, start, true);
	val[2] = tryGuestND(src, cut, conf[2]);
	if (val[2].empty()) {
		conf[2] = -1;
	}

	dropfallRight(src, cut, start, false);
	val[3] = tryGuestND(src, cut, conf[3]);
	if (val[3].empty()) {
		conf[3] = -1;
	}
	auto it = std::max_element(conf, conf + 4);
	srcConf = *it;
	int index = it - conf;
	return val[index];
}

bool recognizeDigits(std::vector<cv::Point2i >& blob, cv::Rect& bound, int& label, double conf) {
	cv::Mat temp = cropBlob(blob, bound);
	double tempConf = 0;
	int tempLabel = recognize1D(temp, tempConf);
	if (tempConf > conf) {
		label = tempLabel;
		conf = tempConf;
	}
	cv::Mat crop;
	cropMat(temp, crop, 1);
//	cropMat(deslant(temp), crop, 1);
	//guest number digit
	int numDigit = 0;
	int width = crop.cols;
	int height = crop.rows;
	float aspect = width /(float) height;
	if (aspect <= 0.5) {
		numDigit = 1;
	} else
	if (aspect <= 1.8) {
		std::string val = recognizeND(crop, conf);
		if (val.empty()) {
			return false;
		}

		std::cout << "\"" << val << "\"";
		numDigit = 2;
		return true;
	}
	else if (aspect > 1.1) {
		std::string val = recognizeND(crop, conf);
		if (val.empty()) {
			return false;
		}

		std::cout << "\"" << val << "\"";
		numDigit = 3;
		return true;
	}
	return false;
	//try cut
	//vector<int> cut;
	//dropfallLeft();
}