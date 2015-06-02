#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "recognizer.h"
#include "blob.h"

namespace cv {
	void doNothing();
}
#define DEBUG
#ifndef DEBUG
#define imshow(str, img) doNothing()
#endif
//threshold
enum class BhThresholdMethod{OTSU, NIBLACK, SAUVOLA, WOLFJOLION};
void doThreshold(cv::InputArray src, cv::OutputArray dst, const BhThresholdMethod &method);
void removeNoise(cv::Mat& binary);

//dropfall
//@deprecated
void dropfallLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallExtLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallExtRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);

//preprocessing
cv::Mat cropBlob(Blob& blob, int pad = 0);
bool cropMat(cv::Mat& src, cv::Mat& dst, int pad = 0);
cv::Mat slant(cv::Mat& src, float degree);
float objectWidth(cv::Mat& input);
float blobsWidth(cv::Mat& input);
/* input[0,1] output[0,1] */
float deslant(cv::Mat& input, cv::Mat *dst = NULL, float (*fntSlantCost)(cv::Mat&) = blobsWidth);
/* need implementing*/
float deslant(cv::Size imgSize, Blobs& blobs);
float resolveBlobAngle(Blob& blob, int imgHeight, float imgSlantAngle);
cv::Rect getROI(cv::Mat& src);

void projectHorizontal(cv::Mat& input, std::vector<int>& output);
cv::Mat cropDigitString(cv::Mat& src);
void drawCut(cv::Mat& src, std::vector<cv::Point> &cut);

//@deprecated
void projectVertical(cv::Mat &input, std::vector<int> &output);
void projectVertical(Blobs &blobs, std::vector<int> &output);
void updateVerticalProjection(std::vector<cv::Point2i>& blobPoints, std::vector<int>& output);
std::vector<int> genVerticalCuts(std::vector<int>& projectV);

//thinning
void thinning(const cv::Mat& src, cv::Mat& dst);

//recognize-touch
//@deprecated
cv::Mat makeDigitMat(const cv::Mat& crop);
cv::Mat makeDigitMat(Blob& blob, float slantAngle = 0);
digit_recognizer::result recognize1D(const cv::Mat& src);
std::string extractDigit(Blobs& blobs, float slantAngle);
std::string recognizeDigits(Blob& blob, int estDigitWidth, digit_recognizer::result& r);

#endif
