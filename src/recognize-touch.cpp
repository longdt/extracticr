
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include "recognizer.h"
#include "util/misc.h"
#include "preprocessor.h"

#define CONFIDENCE_THRESHOLD 0.01
using namespace tiny_cnn;
using namespace cv;


extern digit_recognizer recognizer;

std::string recognizeND(Mat& src, double& srcConf);


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

cv::Mat makeDigitMat(Blob& blob) {
	cv::Mat crop = cropBlob(blob);
	crop = deslant(crop);
	cropMat(crop, crop);
	return makeDigitMat(crop);
}

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
	if (confND2 < confPart2 || label2.empty()) {
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
	if (start > src.rows) {
		start = src.rows;
	}
	double conf[8];
	for (int i = 0; i < 8; ++i) {
		conf[i] = srcConf;
	}
	std::string val[8];

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

	dropfallExtLeft(src, cut, start, true);
	val[2] = tryGuestND(src, cut, conf[2]);
	if (val[2].empty()) {
		conf[2] = -1;
	}

	dropfallExtLeft(src, cut, start, false);
	val[3] = tryGuestND(src, cut, conf[3]);
	if (val[3].empty()) {
		conf[3] = -1;
	}

	start = src.cols - 1 - start;
	dropfallRight(src, cut, start, true);
	val[4] = tryGuestND(src, cut, conf[4]);
	if (val[4].empty()) {
		conf[4] = -1;
	}

	dropfallRight(src, cut, start, false);
	val[5] = tryGuestND(src, cut, conf[5]);
	if (val[5].empty()) {
		conf[5] = -1;
	}

	dropfallExtRight(src, cut, start, true);
	val[6] = tryGuestND(src, cut, conf[6]);
	if (val[6].empty()) {
		conf[6] = -1;
	}

	dropfallExtRight(src, cut, start, false);
	val[7] = tryGuestND(src, cut, conf[7]);
	if (val[7].empty()) {
		conf[7] = -1;
	}
	auto it = std::max_element(conf, conf + 8);
	srcConf = *it;
	int index = it - conf;
	return val[index];
}

std::string recognizeDigits(Blob& blob, int estDigitWidth, digit_recognizer::result& r) {
	cv::Mat temp = cropBlob(blob);
	cv::Mat digit = makeDigitMat(temp);
	digit_recognizer::result otherResult = recognizer.predict(digit);
	if (otherResult.conf > r.conf || otherResult.softmaxScore > r.softmaxScore) {
		r = otherResult;
	}
	cv::Mat crop;
	cropMat(temp, crop, 1);
//	cropMat(deslant(temp), crop, 1);
	//guest number digit
	auto bound = blob.boundingRect();
	int numDigit = (int) (bound.width /(float) estDigitWidth + 0.4);
	int width = crop.cols;
	int height = crop.rows;
	float aspect = width /(float) height;
	if (numDigit >= 2) {
		r.conf = -1;
	}
	if (aspect <= 0.5 || numDigit == 1) {
		numDigit = 1;
	} else
	if (aspect <= 1.8) {
		std::string val = recognizeND(crop, r.conf);
		numDigit = 2;
		return val;
	}
	else if (aspect > 1.1) {
		std::string val = recognizeND(crop, r.conf);
		numDigit = 3;
		return val;
	}
	return "";
}


bool isSingleDigit(double conf, cv::Rect& bound, average<int>& estDigitWidth) {
	if (estDigitWidth.size() > 0) {
		return bound.width <= estDigitWidth.mean() + estDigitWidth.deviation();
	}
	float aspect = bound.width / (float)(bound.height);
	if (aspect > 1) {
		return conf > 0.1;
	}
	else if (aspect >= 0.6) {
		return conf > 0.08;
	}
	else if (aspect >= 0.5) {
		return conf > 0.05;
	}
	else {
		return conf > 0.01;
	}
}

std::string concate(vector<string> strs) {
	stringstream ss;
	for (string& str : strs) {
		ss << str;
	}
	return ss.str();
}

std::string extractDigit(cv::Mat &binary, Blobs& blobs) {
	sortBlobsByVertical(blobs);
	vector<string> labels;
	labels.resize(blobs.size());
	vector<digit_recognizer::result> predRs;
	average<int> widthDigit;

	//try recognize sing digit first
	for (int blobIdx = 0; blobIdx < blobs.size(); ++blobIdx) {
		cv::Mat digit = makeDigitMat(*blobs[blobIdx]);
		digit_recognizer::result r = recognizer.predict(digit);
		predRs.push_back(r);
		//debug show info
		imshow(std::to_string(r.label) + "pad" + std::to_string(r.conf) + "*" + std::to_string(r.softmaxScore), digit);
		if (r.softmaxScore > 0.09) {
			labels[blobIdx] = to_string(r.label);
			if (r.label != 1) {
				widthDigit.update(blobs[blobIdx]->boundingRect().width);
			}
			continue;
		}
	}
	int dw = widthDigit.size() > 0 ? widthDigit.mean() : -1;
	//try recognize touching-digit 
	for (int i = 0; i < labels.size(); ++i) {
		if (!labels[i].empty()) {
			continue;
		}
		digit_recognizer::result r = predRs[i];
		auto bound = blobs[i]->boundingRect();
		if (isSingleDigit(r.softmaxScore, bound, widthDigit)) {
			if (r.conf > CONFIDENCE_THRESHOLD)
				labels[i] = to_string(r.label);
		}
		else if ((labels[i] = recognizeDigits(*blobs[i], dw, r)).empty()){
			//debug print miss recognize
			//std::cout << r.label;
		}
	}
	bool reject = false;
	for (int i = 0; i < labels.size(); ++i) {
		if (!labels[i].empty()) {
			continue;
		}
		else if (predRs[i].softmaxScore > CONFIDENCE_THRESHOLD) {
			labels[i] = to_string(predRs[i].label);
		}
		else {
			reject = true;
			labels[i] = "-" + to_string(predRs[i].label);
		}
	}
	if (reject) {
		std::cout << concate(labels) << std::endl;
		return "";
	}
	return concate(labels);
}
