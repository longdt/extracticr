#include "recognizer.h"
#include "preprocessor.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <random>       // std::default_random_engine
#include <chrono>
using namespace std;

using namespace boost::filesystem;

digit_recognizer recognizer;

cv::Mat removeNoise(cv::Mat& src) {
	cv::Mat temp, dst;
	int morph_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	cv::morphologyEx(src, temp, cv::MORPH_CLOSE, element);
//	cv::morphologyEx(temp, dst, cv::MORPH_OPEN, element);
	return temp;
}

cv::Mat drawBlob(const cv::Mat& src, const std::vector < std::vector<cv::Point2i > >& blobs) {
	cv::Mat output = cv::Mat::zeros(src.size(), CV_8UC3);
	// Randomy color the blobs
	for (size_t i = 0; i < blobs.size(); i++) {
		unsigned char r = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char g = 255 * (rand() / (1.0 + RAND_MAX));
		unsigned char b = 255 * (rand() / (1.0 + RAND_MAX));

		for (size_t j = 0; j < blobs[i].size(); j++) {
			int x = blobs[i][j].x;
			int y = blobs[i][j].y;

			output.at<cv::Vec3b>(y, x)[0] = b;
			output.at<cv::Vec3b>(y, x)[1] = g;
			output.at<cv::Vec3b>(y, x)[2] = r;
		}
	}
	return output;
}

int main(int argc, char **argv)
{
	char c = 0;
	path p("D:\\workspace\\data\\train");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	typedef vector<path> vec;             // store paths,
	vec v;                                // so we can sort them later
	copy(directory_iterator(p), directory_iterator(), back_inserter(v));
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	shuffle(v.begin(), v.end(), std::default_random_engine(seed));
	for (vec::const_iterator it(v.begin()), it_end(v.end()); it != it_end && c != 'q'; ++it)
	{
		//cv::Mat img = cv::imread(it->string(), 0); // force greyscale
		cv::Mat img = cv::imread("D:\\workspace\\data\\train\\136075-0179-03.png", 0); // force greyscale
		if (!img.data) {
			std::cout << "File not found" << std::endl;
			return -1;
		}
		img = 255 - img;
		cv::namedWindow("binary");
		cv::namedWindow("labelled");

		cv::Mat binary;
		std::vector < std::vector<cv::Point2i > > blobs;
		std::vector<cv::Rect> bounds;
		cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY | CV_THRESH_OTSU);
		//	binary = removeNoise(binary);
		binary = deslant(binary);
		findBlobs(binary, blobs, &bounds);
		binary = binary * 255;
		cv::imshow("binary", binary);
		cv::Mat output = drawBlob(binary, blobs);
		cv::imshow("labelled", output);
		std::vector<int> labels;
		groupVertical(blobs, bounds, labels);
		extractDigit(binary, blobs, bounds);
		cv::imshow(it->filename().string(), img);
		c = cv::waitKey(0);
		cv::destroyAllWindows();
	}
	return 0;
}