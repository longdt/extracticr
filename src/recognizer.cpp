/*
Copyright (c) 2013, Taiga Nomi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
* Neither the name of the <organization> nor the
names of its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/
#include <iostream>
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>
#include <math.h>
#include "util/cvl_parser.h"
#include "util/misc.h"
#include "recognizer.h"

#include "cnn/util.h"
//#define NOMINMAX
//#include "imdebug.h"

using namespace cv;

using namespace std;

#define O true
#define X false
static const bool connection[] = {
	O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
	O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
	O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
	X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
	X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
	X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
};
#undef O
#undef X

digit_recognizer::digit_recognizer(bool load_weight) : C1(32, 32, 5, 1, 6), S2(28, 28, 6, 2),
		C3(14, 14, 5, 6, 16, connection_table(connection, 6, 16)), 
		S4(10, 10, 16, 2), C5(5, 5, 5, 16, 120), F6(120, 12) {
	assert(C1.param_size() == 156 && C1.connection_size() == 122304);
	assert(S2.param_size() == 12 && S2.connection_size() == 5880);
	assert(C3.param_size() == 1516 && C3.connection_size() == 151600);
	assert(S4.param_size() == 32 && S4.connection_size() == 2000);
	assert(C5.param_size() == 48120 && C5.connection_size() == 48120);

	nn.add(&C1);
	nn.add(&S2);
	nn.add(&C3);
	nn.add(&S4);
	nn.add(&C5);
	nn.add(&F6);

	std::cout << "load models..." << std::endl;
	if (load_weight) {
		std::ifstream ifs("LeNet-weights");
		if (ifs.is_open()) {
			ifs >> C1;
			ifs >> S2;
			ifs >> C3;
			ifs >> S4;
			ifs >> C5;
			ifs >> F6;
			ifs.close();
			std::cout << "loaded weights" << std::endl;
		}
	}
}

digit_recognizer::~digit_recognizer() {
}


///////////////////////////////////////////////////////////////////////////////
// learning convolutional neural networks (LeNet-5 like architecture)
void digit_recognizer::test() {


	// load MNIST dataset
	std::vector<label_t> train_labels, test_labels;
	std::vector<vec_t> train_images, test_images;

	//parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
	//parse_mnist_images("train-images.idx3-ubyte", &train_images);
	//parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
	//parse_mnist_images("t10k-images.idx3-ubyte", &test_images);
	//parse_cvl("D:\\workspace\\visual\\tiny-cnn-master\\vc\\cvl\\train", train_images, train_labels);
	//parse_cvl("D:\\workspace\\visual\\tiny-cnn-master\\vc\\cvl\\valid", test_images, test_labels);

	std::cout << "start learning" << std::endl;

	boost::progress_display disp(train_images.size());
	boost::timer t;
	int minibatch_size = 10;

	nn.optimizer().alpha *= std::sqrt(minibatch_size);

	// create callback
	auto on_enumerate_epoch = [&](){
		std::cout << t.elapsed() << "s elapsed." << std::endl;

		tiny_cnn::result res = nn.test(test_images, test_labels);

		std::cout << nn.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;

		nn.optimizer().alpha *= 0.85; // decay learning rate
		nn.optimizer().alpha = std::max(0.00001, nn.optimizer().alpha);

		disp.restart(train_images.size());
		t.restart();
	};

	auto on_enumerate_minibatch = [&](){
		disp += minibatch_size;

		// weight visualization in imdebug
		/*static int n = 0;
		n+=minibatch_size;
		if (n >= 1000) {
		image img;
		C3.weight_to_image(img);
		imdebug("lum b=8 w=%d h=%d %p", img.width(), img.height(), &img.data()[0]);
		n = 0;
		}*/
	};

	// training
	nn.train(train_images, train_labels, minibatch_size, 100, on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;
	// save networks
	std::cin.get();
	std::ofstream ofs("LeNet-weights");
	ofs << C1 << S2 << C3 << S4 << C5 << F6;
	// test and show results
	//nn.test(test_images, test_labels).print_detail(std::cout);
	for (int i = 0; i < 6; ++i) {
		vec_t in;
		string file = "pad" + to_string(i) + ".png";
		load_image(in, file, false);
		cout << "predict image test" << file << ": " << predict(in).label() << endl;
	}
	cin.get();

}

void digit_recognizer::predict(const vec_t& in, vec_t& out) {
	nn.predict(in, &out);
}

digit_recognizer::result digit_recognizer::predict(const vec_t& in) {
	result r(nn.out_dim());
	nn.predict(in, &r.out);
	return r;
}

digit_recognizer::result digit_recognizer::predict(const cv::Mat& digit) {
	assert(digit.cols == 28 && digit.rows == 28);
	vec_t in;
	matToVect(digit, in);
	return predict(in);
}

digit_recognizer::result::result(int numLabels) : index(-1), softmaxScoreSum(-1) {
	softmaxScores.resize(numLabels, -1);
}

label_t digit_recognizer::result::label(bool force) {
	if (index == -1 || force) {
		index = tiny_cnn::max_index(out);
	}
	return index;
}

double digit_recognizer::result::confidence() {
	return out[label()];
}

double digit_recognizer::result::getSoftmaxScoreSum() {
	if (softmaxScoreSum == -1) {
		double sum = 0;
		for (size_t i = 0; i < out.size(); i++) {
			sum += exp(3.75 * out[i]);
		}
		softmaxScoreSum = sum + exp(5);
	}
	return softmaxScoreSum;
}

double digit_recognizer::result::softmaxScore() {
	return softmaxScore(label());
}

double digit_recognizer::result::softmaxScore(label_t idx) {
	if (softmaxScores[idx] == -1) {
		double smScoreSum = getSoftmaxScoreSum();
		softmaxScores[idx] = exp(3.75 * out[idx]) / smScoreSum;
	}
	return softmaxScores[idx];
}
