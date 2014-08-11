#include <vector>
#include <opencv2/core/core.hpp>

void findBlobs(const cv::Mat &binary, std::vector < std::vector<cv::Point2i> > &blobs, std::vector<cv::Rect> *bounds = NULL);

void extractDigit(cv::Mat &binary, std::vector < std::vector<cv::Point2i > >& blobs, std::vector<cv::Rect> &bounds);

cv::Mat cropBlob(std::vector<cv::Point2i >& blob, cv::Rect& bound, int pad = 0);

bool cropMat(cv::Mat& src, cv::Mat& dst, int pad = 0);

void groupVertical(std::vector < std::vector<cv::Point2i> > &blobs, std::vector<cv::Rect> &bounds, std::vector<int> &labels);

void sortBlobsByVertical(std::vector<cv::Rect> &bounds, std::vector<int> &order);

void projectVeritcal(cv::Mat &input, std::vector<int> &output);

void projectVeritcal(std::vector < std::vector<cv::Point2i> > &blobs, std::vector<int> &output);

void updateVerticalProjection(std::vector<cv::Point2i>& blob, std::vector<int>& output);

std::vector<int> genVerticalCuts(std::vector<int>& projectV);

cv::Mat makeDigitMat(cv::Mat& crop);

cv::Mat makeDigitMat(std::vector<cv::Point2i >& blob, cv::Rect* bound);

cv::Mat deslant(cv::Mat& input);

bool recognizeDigits(std::vector<cv::Point2i >& blob, cv::Rect& bound, int& label, double conf);

void dropfallLeft(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
void dropfallRight(const cv::Mat& src, std::vector<cv::Point2i >& cut, int start, bool top);
