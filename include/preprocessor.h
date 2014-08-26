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


//dropfall
void dropfallLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallExtLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallExtRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);

//preprocessing
cv::Mat cropBlob(Blob& blob, int pad = 0);
bool cropMat(cv::Mat& src, cv::Mat& dst, int pad = 0);
cv::Mat deslant(cv::Mat& input);

void projectHorizontal(cv::Mat& input, std::vector<int>& output);
cv::Mat cropDigitString(cv::Mat& src);

//@deprecate
void projectVeritcal(cv::Mat &input, std::vector<int> &output);
void projectVeritcal(std::vector < std::vector<cv::Point2i> > &blobPoints, std::vector<int> &output);
void updateVerticalProjection(std::vector<cv::Point2i>& blobPoints, std::vector<int>& output);
std::vector<int> genVerticalCuts(std::vector<int>& projectV);

//thinning
void thinning(const cv::Mat& src, cv::Mat& dst);

//recognize-touch
cv::Mat makeDigitMat(cv::Mat& crop);
cv::Mat makeDigitMat(Blob& blob);
std::string extractDigit(cv::Mat &binary, Blobs& blobs);
std::string recognizeDigits(Blob& blob, int estDigitWidth, digit_recognizer::result& r);

#endif
