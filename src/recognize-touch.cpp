
#include <BFSegment.h>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <cmath>
#include "recognizer.h"
#include "util/misc.h"
#include "preprocessor.h"

#include "digit-statistic.h"

#define CONFIDENCE_THRESHOLD  0.03
#define PROBABLY_SINGLE_DIGIT 0.09
using namespace tiny_cnn;
using namespace cv;


extern digit_recognizer recognizer;
extern vector<DigitWidthStatistic> digitStatistics;

std::string recognizeND(Mat& src, average<int>& estDigitWidth, double& srcConf);
bool isSingleDigit(digit_recognizer::result& predRs, cv::Rect& bound, average<int>& estDigitWidth);

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

cv::Mat blobToMat(Blob& blob) {
	cv::Rect bound = blob.boundingRect();
	cv::Mat rs = cv::Mat::zeros(bound.y + bound.height, bound.width, CV_8UC1);
	for (size_t j = 0; j < blob.points.size(); j++) {
		int x = blob.points[j].x - bound.x;
		int y = blob.points[j].y;
		rs.at<uchar>(y, x) = 255;
	}
	return rs;
}

cv::Mat makeDigitMat(Blob& blob, float slantAngle) {
	cv::Mat crop = blobToMat(blob);
	float angle = deslant(crop, NULL, projectWidth);
	if (slantAngle * angle > 0) {
		angle = abs(angle + slantAngle) > 48 ? 0 : angle;
	}
	if (angle != 0) {
		crop = slant(crop, angle);
	}
	cropMat(crop, crop);
	return makeDigitMat(crop);
}

void drawCut(Mat& src, vector<Point> &cut) {
	for (auto p : cut) {
		src.at<uchar>(p) = 100;
	}
}

void makeCut(const Mat& src, vector<Point> &cut, Mat& part1, Mat& part2) {
	part1 = Mat::zeros(src.rows, src.cols, src.type());
	part2 = Mat::zeros(src.rows, src.cols, src.type());
	for (auto p : cut) {
		for (int j = 0; j < src.cols; ++j) {
			if (j < p.x) {
				part1.at<uchar>(p.y, j) = src.at<uchar>(p.y, j);
			} else {
				part2.at<uchar>(p.y, j) = src.at<uchar>(p.y, j);
			}
		}
	}
}

digit_recognizer::result recognize1D(Mat& src) {
	auto digitMat = makeDigitMat(src);
	vec_t in;
	mat_to_vect(digitMat, in);
	return recognizer.predict(in);
}

std::string tryGuestND(const Mat& src, vector<Point>& cut, double& conf, average<int>& estDW) {
	Mat part2;
	Mat part1;
	makeCut(src, cut, part1, part2);
//	Mat srcClone = src.clone();
//	drawCut(srcClone, cut);
//	cv::destroyAllWindows();
//	imshow("cut-source", srcClone);
//	waitKey(0);
	Mat croppedPart1;
	if (!cropMat(part1, croppedPart1) || croppedPart1.rows <= src.rows * 0.6) {
		return "";
	}
	auto rs1 = recognize1D(croppedPart1);
//	imwrite("temp/" + std::to_string(rs1.label()) + std::to_string(clock()) + "-part1.png", 255 - makeDigitMat(croppedPart1));
	if (rs1.label() == 10 || rs1.softmaxScore() <= CONFIDENCE_THRESHOLD) {
		return "";
	}
	Mat croppedPart2;
	if (!cropMat(part2, croppedPart2, 1) || croppedPart2.rows <= src.rows * 0.57) {
		return "";
	}
	auto rs2 = recognize1D(croppedPart2);
//	imwrite("temp/" + std::to_string(rs2.label()) + std::to_string(clock()) + "-part2.png", 255 - makeDigitMat(croppedPart2));
//	cv::destroyAllWindows();
//	imshow(std::to_string(rs1.label()) + "part1." + std::to_string(rs1.confidence()), croppedPart1);
//	imshow(std::to_string(rs2.label()) + "part2." + std::to_string(rs2.confidence()), croppedPart2);
//	imshow("source", src);
//	waitKey(0);
	cv::Rect bound2 = cv::Rect(0, 0, croppedPart2.cols -1, croppedPart2.rows -1);
	double confidence2 = isSingleDigit(rs2, bound2, estDW) ? rs2.confidence() : -1;
	double confND2 = confidence2;
	std::string label2;
	if (rs2.label() == 10) {
		label2 = recognizeND(croppedPart2, estDW, confND2);
	}
	if (confND2 < confidence2 || label2.empty()) {
		if (rs2.label() == 10 || rs2.softmaxScore() <= CONFIDENCE_THRESHOLD) {
			return "";
		}
		confND2 = confidence2;
		label2 = std::to_string(rs2.label());
	}

	if (confND2 < 0 || (rs1.confidence() * confND2 < conf)) {
		return "";
	}
	conf = rs1.confidence() * confND2;
	return std::to_string(rs1.label()) + label2;
}

std::string recognizeND(Mat& src, average<int>& estDigitWidth, double& srcConf) {
	if (src.rows / src.cols > 2) {
		return "";
	}
	BFSegment segmenter(src);
	segmenter.genFPLayers();
	vector<vector<Point>> cuts;
	segmenter.genCutPath(src.rows, cuts);
	vector<string> labels(cuts.size());
	vector<double> confidens(cuts.size(), -1);
	for (int i = 0; i < cuts.size(); ++i) {
		labels[i] = tryGuestND(src, cuts[i], confidens[i], estDigitWidth);
	}
	int bestIdx = std::max_element(confidens.begin(), confidens.end()) - confidens.begin();
	if (!cuts.empty() && confidens[bestIdx] > srcConf && !labels[bestIdx].empty()) {
		srcConf = confidens[bestIdx];
		return labels[bestIdx];
	}
//	else {
//		return "";
//	}
	int start = estDigitWidth.size() == 0 ? (int)(src.cols * 0.375) : 0.8 * estDigitWidth.mean() - estDigitWidth.deviation();
	if (start > src.rows) {
		start = src.rows * 0.80;
	}
	if (start >= src.cols) {
		return "";
	}
	double conf[8];
	for (int i = 0; i < 8; ++i) {
		conf[i] = srcConf;
	}
	std::string val[8];

	vector<Point> cut;
	dropfallLeft(src, cut, start, true);
	val[0] = tryGuestND(src, cut, conf[0], estDigitWidth);
	if (val[0].empty()) {
		conf[0] = -1;
	}

	dropfallLeft(src, cut, start, false);
	val[1] = tryGuestND(src, cut, conf[1], estDigitWidth);
	if (val[1].empty()) {
		conf[1] = -1;
	}

	dropfallExtLeft(src, cut, start, true);
	val[2] = tryGuestND(src, cut, conf[2], estDigitWidth);
	if (val[2].empty()) {
		conf[2] = -1;
	}

	dropfallExtLeft(src, cut, start, false);
	val[3] = tryGuestND(src, cut, conf[3], estDigitWidth);
	if (val[3].empty()) {
		conf[3] = -1;
	}

	start = estDigitWidth.size() > 0 ? estDigitWidth.mean() + estDigitWidth.deviation() : src.cols - 1 - start;
	if (start > src.rows) {
		start = src.rows;
	}
	if (start >= src.cols) {
		return "";
	}
	dropfallRight(src, cut, start, true);
	val[4] = tryGuestND(src, cut, conf[4], estDigitWidth);
	if (val[4].empty()) {
		conf[4] = -1;
	}

	dropfallRight(src, cut, start, false);
	val[5] = tryGuestND(src, cut, conf[5], estDigitWidth);
	if (val[5].empty()) {
		conf[5] = -1;
	}

	dropfallExtRight(src, cut, start, true);
	val[6] = tryGuestND(src, cut, conf[6], estDigitWidth);
	if (val[6].empty()) {
		conf[6] = -1;
	}

	dropfallExtRight(src, cut, start, false);
	val[7] = tryGuestND(src, cut, conf[7], estDigitWidth);
	if (val[7].empty()) {
		conf[7] = -1;
	}
	auto it = std::max_element(conf, conf + 8);
	srcConf = *it;
	int index = it - conf;
	return val[index];
}

std::string recognizeDigits(Blob& blob, average<int>& estDigitWidth, digit_recognizer::result& r) {
	cv::Mat temp = cropBlob(blob);
	cv::Mat digit = makeDigitMat(temp);
	digit_recognizer::result otherResult = recognizer.predict(digit);
	if (otherResult.confidence() > r.confidence() || otherResult.softmaxScore() > r.softmaxScore()) {
		r = otherResult;
	}
	cv::Mat crop;
	cropMat(temp, crop, 1);
//	cropMat(deslant(temp), crop, 1);
	//guest number digit
	auto bound = blob.boundingRect();
	int numDigit = estDigitWidth.size() != 0 ? (int) (bound.width /(float) estDigitWidth.mean() + 0.5) : 0;
	int width = crop.cols;
	int height = crop.rows;
	float aspect = width /(float) height;
	double confidence = r.label() == 10 ? -1 : r.confidence();
	if (numDigit >= 2 || (numDigit == 0 && aspect > 1.1)) {
		confidence = -1;
	}
//	if (aspect <= 0.5 || numDigit == 1) {
//		numDigit = 1;
//		if (r.softmaxScore() > CONFIDENCE_THRESHOLD) {
//			return std::to_string(r.label());
//		}
//	} else
	if (aspect <= 1.8) {
		std::string val = recognizeND(crop, estDigitWidth, confidence);
		numDigit = 2;
		return val;
	}
	else if (aspect > 1.1) {
		std::string val = recognizeND(crop, estDigitWidth, confidence);
		numDigit = 3;
		return val;
	}
	return "";
}


bool isSingleDigit(digit_recognizer::result& predRs, cv::Rect& bound, average<int>& estDigitWidth) {
//	if (estDigitWidth.size() > 0) {
//		return bound.width <= estDigitWidth.mean() + estDigitWidth.deviation();
//	}
//	float aspect = bound.width / (float)(bound.height);
//	if (aspect > 1) {
//		return predRs.confidence() > 0.1;
//	}
//	else if (aspect >= 0.6) {
//		return predRs.confidence() > 0.08;
//	}
//	else if (aspect >= 0.5) {
//		return predRs.confidence() > 0.05;
//	}
//	else {
//		return predRs.confidence() > 0.01;
//	}
	//other implement
//	if (predRs.label() == 10) {
//		return false;
//	} else {
//		return true;
//	}
	double normalWidth = GET_NORMAL_DIGIT_WIDTH(bound.width, bound.height);
	double score = 0;
	for (int i = 0; i < 10; ++i) {
		double tkPI = std::erfc(std::abs(normalWidth - digitStatistics[i].mean) / digitStatistics[i].deviation);
		score += tkPI * predRs.softmaxScore(i) * 10;
	}
	score = score / 10;
	return score > 0.0025;
}

std::string concate(vector<string> strs) {
	stringstream ss;
	for (string& str : strs) {
		ss << str;
	}
	return ss.str();
}
bool isPeriod(cv::Rect carBox, int middleLine, Blob& blob);
std::string extractDigit(Blobs& blobs, float slantAngle) {
	sortBlobsByVertical(blobs);
	vector<string> labels;
	labels.resize(blobs.size());
	vector<digit_recognizer::result> predRs(blobs.size(), digit_recognizer::result(0));
	average<int> widthDigit;
	cv::Rect blobsBox = blobs.boundingRect();
	int middleLine = blobsBox.y + blobsBox.height / 2;
	//try recognize sing digit first
	for (int blobIdx = 0; blobIdx < blobs.size(); ++blobIdx) {
		cv::Mat digit = makeDigitMat(*blobs[blobIdx], slantAngle);
		Blob* blob = blobs[blobIdx];
		if (isPeriod(blobsBox, middleLine, *blob)) {
			labels[blobIdx] = ".";
			continue;
		}
		digit_recognizer::result r = recognizer.predict(digit);
		//debug show info
		imshow(std::to_string(r.label()) + "pad" + std::to_string(r.confidence()) + "*" + std::to_string(r.softmaxScore()), digit);
//		digit = 255 - digit;
//		imwrite("temp/" + std::to_string(r.label()) + "-" + std::to_string(clock()) + ".png", digit);
		auto bound = blobs[blobIdx]->boundingRect();
		if (r.softmaxScore() > PROBABLY_SINGLE_DIGIT && isSingleDigit(r, bound, widthDigit)) {
			labels[blobIdx] = to_string(r.label());
			if (r.label() != 1) {
				widthDigit.update(blobs[blobIdx]->boundingRect().width);
			}
		}
		predRs[blobIdx] = r;
	}
	//try recognize touching-digit 
	for (int i = 0; i < labels.size(); ++i) {
		if (!labels[i].empty()) {
			continue;
		}
		digit_recognizer::result r = predRs[i];
		auto bound = blobs[i]->boundingRect();
		if (isSingleDigit(r, bound, widthDigit)) {
			if (r.confidence() > CONFIDENCE_THRESHOLD)
				labels[i] = to_string(r.label());
		}
		else {
			labels[i] = recognizeDigits(*blobs[i], widthDigit, r);
		}
	}
	bool reject = false;
	for (int i = 0; i < labels.size(); ++i) {
		if (!labels[i].empty()) {
			continue;
		}
//		else if (predRs[i].softmaxScore() > CONFIDENCE_THRESHOLD) {
//			labels[i] = to_string(predRs[i].label());
//		}
		else {
			reject = true;
			labels[i] = "-" + to_string(predRs[i].label());
		}
	}
	if (reject) {
		std::cout << concate(labels) << std::endl;
		return "";
	}
	return concate(labels);
}
