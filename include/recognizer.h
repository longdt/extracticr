#ifndef RECOGNIZER_H
#define RECOGNIZER_H
#include "cnn/tiny_cnn.h"
#include <opencv2/core/core.hpp>

using namespace tiny_cnn;
// construct LeNet-5 architecture
typedef network<mse, gradient_descent_levenberg_marquardt> CNN;

class digit_recognizer {
private :
	CNN nn;
	convolutional_layer<CNN, tanh_activation> C1;
	average_pooling_layer<CNN, tanh_activation> S2;
	convolutional_layer<CNN, tanh_activation> C3;
	average_pooling_layer<CNN, tanh_activation> S4;
	convolutional_layer<CNN, tanh_activation> C5;
	fully_connected_layer<CNN, tanh_activation> F6;
	void test();
public:
	digit_recognizer(bool load_weight = true);
	~digit_recognizer();
	void predict(const vec_t& in, vec_t& out);
	class result {
	private:
		label_t index;
		vec_t softmaxScores;
		double softmaxScoreSum;
		double getSoftmaxScoreSum();
	public:
		explicit result(int numLabels);
		vec_t out;
		label_t label(bool force = false);
		double confidence();
		double softmaxScore();
		double softmaxScore(label_t lb);
	};
	result predict(const vec_t& in);
	result predict(const cv::Mat& in);
};
#endif
