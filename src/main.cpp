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
#include <iosfwd>
using namespace std;

using namespace boost::filesystem;



cv::Mat removeNoise(const cv::Mat& src) {
	cv::Mat temp(src.size(), src.type()), dst;
	int morph_size = 1;
	cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));
	cv::morphologyEx(src, temp, cv::MORPH_CLOSE, element);
//	cv::morphologyEx(temp, dst, cv::MORPH_OPEN, element);
	return temp;
}

bool exportSegment = false;
string filename;
int mmain(int argc, char **argv)
{
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
	ofstream ofs("newSegment.txt");
	for (vec::const_iterator it(v.begin()), it_end(v.end()); it != it_end && c != 'q'; ++it)
	{
//		cv::Mat img = cv::imread(it->string(), 0); // force greyscale
		cv::Mat img = cv::imread("/media/thienlong/linux/CAR/cvl-strings/car 6.png", 0); // force greyscale
		if (!img.data) {
			std::cout << "File not found" << std::endl;
			return -1;
		}
		filename = it->filename().string();
		cout << filename << ": ";
		cout.flush();
		img = 255 - img;

		cv::Mat binary;
		Blobs blobs;
		cv::threshold(img, binary, 0.0, 1.0, cv::THRESH_BINARY | CV_THRESH_OTSU);
//		binary = removeNoise(binary);
		binary = cropDigitString(binary);
		findBlobs(binary, blobs);
		float angle = deslant(binary.size(), blobs); //deslant(binary, &binary); //
		cv::Mat output = drawBlob(blobs);
		cv::imshow("labelled", output);
//		cv::waitKey(0);
		binary = binary * 255;
		cv::imshow("binary", binary);
		groupVertical(blobs);
		defragment(output, blobs);

		cv::imshow(it->filename().string(), img);

		string actual = extractDigit(blobs, angle);
		string desire = parse_label(it->filename().string());
		if (actual.empty()) {
			++reject;
#ifdef DEBUG
 			c = cv::waitKey(0);
#endif
		}
		else if (actual.compare(desire) == 0) {
			ofs << filename << endl;
			++correct;
			cout << endl;
			c = cv::waitKey(0);
		}
		else {
			cout << actual << endl;
//			exportSegment = true;
//			extractDigit(binary, blobs);
//			exportSegment = false;
#ifdef DEBUG
			c = cv::waitKey(0);
#endif
		}
		cv::destroyAllWindows();
	}
	ofs.close();
	cout << "correct: " << correct << endl << "reject: " << reject << endl <<"error: " << (v.size() - correct - reject) << endl << "total: " << v.size() <<endl;
	cin.get();
	return 0;
}
