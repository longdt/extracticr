#include "util/misc.h"

void mat_to_vect(const cv::Mat& input, vec_t &dst) {
	int x_padding = 2;
	int y_padding = 2;
	const int width = input.cols + 2 * x_padding;
	const int height = input.rows + 2 * y_padding;
	float scale_min = -1.0;
	float scale_max = 1.0;
	dst.resize(width * height, scale_min);
	for (size_t y = 0; y < input.rows; y++)
		for (size_t x = 0; x < input.cols; x++)
			dst[width * (y + y_padding) + x + x_padding]
			= (input.at<uchar>(y, x) / 255.0) * (scale_max - scale_min) + scale_min;
}


string parse_label(string filename) {
	string label;
	std::size_t found = filename.find("-");
	if (found != std::string::npos) {
		label = filename.substr(0, found);
	}
	return label;
}

//
//average::average() : sum(0), m_size(0){}
//
//void average::update(int value) {
//	sum += value;
//	++m_size;
//}
//
//int average::mean() {
//	return sum / m_size;
//}
//
//int average::size() {
//	return m_size;
//}


