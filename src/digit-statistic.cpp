/*
 * digit-statistic.cpp
 *
 *  Created on: Aug 26, 2014
 *      Author: thienlong
 */



#include "digit-statistic.h"
#include "util/misc.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;

using namespace boost::filesystem;



void computeDigitWidth(const std::string& folder, std::vector<DigitWidthStatistic>& output) {
	output.clear();
	path p(folder);
	if (!exists(p) || !is_directory(p)) {
		return;
	}
	vector<average<int>> accumulates;
	accumulates.resize(10);
	output.resize(10);
	for (auto it = directory_iterator(p), end = directory_iterator(); it != end ; ++it) {
		cv::Mat img = cv::imread(it->path().string(), 0);
		string lableStr = parse_label(it->path().filename().string());
		int label = std::stoi(lableStr);
		accumulates[label].update(GET_NORMAL_DIGIT_WIDTH(img.cols, img.rows));
	}
	for (int i = 0; i < accumulates.size(); ++i) {
		output[i].mean = accumulates[i].mean();
		output[i].deviation = accumulates[i].sdeviation();
	}

}
