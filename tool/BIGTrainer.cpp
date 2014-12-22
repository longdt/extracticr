/*
 * BIGTrainer.cpp
 *
 *  Created on: Nov 26, 2014
 *      Author: thienlong
 */




#include <boost/progress.hpp>
#include <cnn/tiny_cnn.h>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <unordered_map>

#include "CARLocator.h"

#include "NumberRecognizer.h"

#include "preprocessor.h"
using icr::CARLocator;
using icr::NumberRecognizer;
using icr::Segment;
using namespace tiny_cnn;

void loadSegmConfig(
		std::unordered_map<std::string, std::vector<Segment>>& segmStrs, std::string file);

void loadBIGTrainData(std::vector<vec_t>& inputs,
		std::vector<label_t>& labels) {
	std::unordered_map<std::string, std::vector<Segment>> segmStrs;
	loadSegmConfig(segmStrs, "train/segments.txt");
	std::unordered_map<std::string, std::vector<Segment>> ignoreSegms;
	loadSegmConfig(ignoreSegms, "train/big-ingnore.txt");
	for (auto it = segmStrs.begin(), end = segmStrs.end(); it != end; ++it) {
		std::cout << it->first << std::endl;
		cv::Mat cheque = cv::imread(
				"/home/thienlong/cheque/500 Cheques/ValidChq/" + it->first, 0);
		CARLocator locator(cheque);
		Blobs blobs;
		locator.getHandwrittenBlobs(blobs);
		if (blobs.size() == 0) {
			return;
		}
		cv::Rect loc = blobs.boundingRect();
		blobs.move(-loc.x, -loc.y);
		float angle = deslant(loc.size(), blobs);
		loc = blobs.boundingRect();
		groupVertical(blobs);
		NumberRecognizer nr(blobs);
		nr.genTrainData(it->second, ICR_BIG, inputs, labels);
		auto ignoreIt = ignoreSegms.find(it->first);
		if (ignoreIt != ignoreSegms.end()) {
			nr.genTrainData(ignoreIt->second, ICR_BIG_IGNORE, inputs, labels);
		}
	}
}

int bigmain(int argc, char **argv) {
	typedef network<mse, gradient_descent> MLP;
	MLP mynet;
	fully_connected_layer<MLP, sigmoid_activation> F1(13, 80);
	fully_connected_layer<MLP, sigmoid_activation> F2(80, 6);
	mynet.add(&F1);
	mynet.add(&F2);

	std::vector<label_t> train_labels;
	std::vector<vec_t> train_inputs;
	loadBIGTrainData(train_inputs, train_labels);
	std::ifstream ifs("big");
	if (ifs.is_open()) {
		ifs >> F1 >> F2;
		ifs.close();
	}
	else {
		mynet.init_weight();
	}
	int minibatch_size = 1;
	boost::progress_display disp(train_inputs.size());
	// create callback
	int counter = 0;
	auto on_enumerate_epoch =
			[&]() {
				tiny_cnn::result res = mynet.test(train_inputs, train_labels);

				std::cout << mynet.optimizer().alpha << "," << res.num_success << "/" << res.num_total << std::endl;
				counter = (counter + 1) % 2;
				if (counter == 0) {
//					mynet.optimizer().alpha *= 0.9; // decay learning rate
				}
				mynet.optimizer().alpha = std::max(0.0001, mynet.optimizer().alpha);

				disp.restart(train_inputs.size());
			};

	auto on_enumerate_minibatch = [&]() {
		disp += minibatch_size;
	};
	// training
	mynet.train(train_inputs, train_labels, minibatch_size, 1000,
			on_enumerate_minibatch, on_enumerate_epoch);

	std::cout << "end training." << std::endl;
	// save networks
	std::ofstream ofs("big");
	ofs << F1 << F2;
	// test and show results
	mynet.test(train_inputs, train_labels).print_detail(std::cout);
	std::cin.get();
}


