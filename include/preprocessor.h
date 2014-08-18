#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <vector>
#include <opencv2/core/core.hpp>
#include "recognizer.h"
namespace cv {
	void doNothing();
}
//#define imshow(str, img) doNothing()
//blob
void findBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs, std::vector<cv::Rect> *bounds = NULL);
cv::Mat drawBlob(const cv::Mat& src, const std::vector < std::vector<cv::Point2i > >& blobs);
void groupVertical(std::vector < std::vector<cv::Point2i> > &blobs, std::vector<cv::Rect> &bounds, std::vector<int> &labels);
void sortBlobsByVertical(std::vector<cv::Rect> &bounds, std::vector<int> &order);


//dropfall
void dropfallLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallExtLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallExtRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);

//preprocessing
cv::Mat cropBlob(std::vector<cv::Point2i >& blob, cv::Rect& bound, int pad = 0);
bool cropMat(cv::Mat& src, cv::Mat& dst, int pad = 0);
cv::Mat deslant(cv::Mat& input);

//@deprecate
void projectVeritcal(cv::Mat &input, std::vector<int> &output);
void projectVeritcal(std::vector < std::vector<cv::Point2i> > &blobs, std::vector<int> &output);
void updateVerticalProjection(std::vector<cv::Point2i>& blob, std::vector<int>& output);
std::vector<int> genVerticalCuts(std::vector<int>& projectV);

//thinning
void thinning(const cv::Mat& src, cv::Mat& dst);

//recognize-touch
cv::Mat makeDigitMat(cv::Mat& crop);
cv::Mat makeDigitMat(std::vector<cv::Point2i >& blob, cv::Rect* bound);
std::string extractDigit(cv::Mat &binary, std::vector < std::vector<cv::Point2i > >& blobs, std::vector<cv::Rect> &bounds);
std::string recognizeDigits(std::vector<cv::Point2i >& blob, cv::Rect& bound, int estDigitWidth, digit_recognizer::result& r);

#endif