#include "recognizer.h"
#include "preprocessor.h"
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <random>       // std::default_random_engine
#include <chrono>
#include "util/misc.h"

#include "digit-statistic.h"
using namespace std;

using namespace boost::filesystem;

digit_recognizer recognizer;
vector<DigitWidthStatistic> digitStatistics;

cv::Mat removeNoise(const cv::Mat& src) {
	cv::Mat temp(src.size(), src.type()), dst;
	int morph_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	cv::morphologyEx(src, temp, cv::MORPH_CLOSE, element);
//	cv::morphologyEx(temp, dst, cv::MORPH_OPEN, element);
	return temp;
}


int main(int argc, char **argv)
{
	computeDigitWidth("/media/thienlong/linux/CAR/cvl-digits/train", digitStatistics);
	char c = 0;
	path p("/media/thienlong/linux/CAR/cvl-strings/train");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	typedef vector<path> vec;             // store paths,
	vec v;                                // so we can sort them later
	copy(directory_iterator(p), directory_iterator(), back_inserter(v));
//	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

//	shuffle(v.begin(), v.end(), std::default_random_engine(seed));
	int reject = 0;
	int correct = 0;
	for (vec::const_iterator it(v.begin()), it_end(v.end()); it != it_end && c != 'q'; ++it)
	{
//		cv::Mat img = cv::imread(it->string(), 0); // force greyscale
		cv::Mat img = cv::imread("/media/thienlong/linux/CAR/cvl-strings/train/25000-0003-08.png", 0); // force greyscale
		if (!img.data) {
			std::cout << "File not found" << std::endl;
			return -1;
		}
		cout << it->filename().string() << ": ";
		img = 255 - img;

		cv::Mat binary;
		Blobs blobs;
		cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY | CV_THRESH_OTSU);
//		binary = removeNoise(binary);
		binary = cropDigitString(binary);
		binary = deslant(binary);
		findBlobs(binary, blobs);
		binary = binary * 255;
		cv::imshow("binary", binary);
		cv::Mat thined;
		thinning(binary, thined);
		cv::imshow("thinning", thined);
		std::vector<int> labels;
		groupVertical(blobs, labels);
		defragment(binary, blobs);
		cv::Mat output = drawBlob(binary, blobs);
		cv::imshow("labelled", output);
		cv::imshow(it->filename().string(), img);

		string actual = extractDigit(binary, blobs);
		string desire = parse_label(it->filename().string());
		if (actual.empty()) {
			++reject;
#ifdef DEBUG
			c = cv::waitKey(0);
#endif
		}
		else if (actual.compare(desire) == 0) {
			++correct;
			cout << endl;
		}
		else {
			cout << actual << endl;
#ifdef DEBUG
			c = cv::waitKey(0);
#endif
		}
		
		cv::destroyAllWindows();
	}
	cout << "correct: " << correct << endl << "reject: " << reject << endl << "total: " << v.size() <<endl;
	cin.get();
	return 0;
}
