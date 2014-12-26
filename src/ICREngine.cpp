/*
 * ICREngine.cpp
 *
 *  Created on: Oct 24, 2014
 *      Author: thienlong
 */

#include <ICREngine.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "recognizer.h"
#include "blob.h"
#include "CARLocator.h"
#include "preprocessor.h"

#include "NumberRecognizer.h"
#include <unordered_map>

#include "util/misc.h"
#include <cstdlib>
#include <iosfwd>
using std::unordered_map;
digit_recognizer recognizer;


bool isPeriod(cv::Rect carBox, int middleLine, Blob& blob) {
	cv::Rect brect = blob.boundingRect();
	if (middleLine > brect.y + brect.height) {
		return false;
	}
	float score = blob.points.size() / (float) (brect.width * brect.height);
	score *= std::min(brect.width, brect.height) / (float) std::max(brect.width, brect.height);
	return score > 0.8;
}

void removeDelimiter(cv::Rect carBox, int middleLine, Blobs& blobs, float slantAngle) {
	for (int idx = blobs.size() - 1; idx >= 0; --idx) {
		Blob& blob = *(blobs[idx]);
		cv::Rect brect = blob.boundingRect();
		if (middleLine > brect.y + brect.height) {
			return;
		}
		float score = blob.points.size() / (float) (brect.width * brect.height);
		score *= std::min(brect.width, brect.height) / (float) std::max(brect.width, brect.height);
		if(score <= 0.8) {
			cv::Mat digit = makeDigitMat(blob, slantAngle);
			digit_recognizer::result rs = recognizer.predict(digit);
			if (rs.label() == 1) {
				blobs.erase(idx);
			}
		}
	}
}

namespace icr {

int detectTerminator(Blobs& blobs) {
	if (blobs.size() <= 1) {
		return -1;
	}
	int firstHeighest = 0;
	int secondHeighest = 0;
	int index = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		cv::Rect rect = blobs[i]->boundingRect();
		if (firstHeighest < rect.height) {
			secondHeighest = firstHeighest;
			firstHeighest = rect.height;
			index = i;
		}
	}
	if (firstHeighest / (float) secondHeighest > 2) {
		return index;
	}
	return -1;
}

void removeTerminator(Blobs& blobs) {
	int idx = detectTerminator(blobs);
	if (idx > 0) {
		for (int i = blobs.size() - 1; i >= idx; --i) {
			blobs.erase(i);
		}
	}
}

int detectPeriod(Blobs& blobs) {
	cv::Rect rect = blobs.boundingRect();
	int middleLine = rect.y + rect.height / 2;
	for (size_t i = 0; i < blobs.size(); ++i) {
		if (isPeriod(rect, middleLine, *blobs[i])) {
			return i;
		}
	}
	return -1;
}

ICREngine::ICREngine() {
}

ICREngine::~ICREngine() {
}

std::string ICREngine::recognite(cv::Mat& cheque) {
	CARLocator locator(cheque);
	Blobs blobs;
	locator.getHandwrittenBlobs(blobs);
	if (blobs.size() == 0) {
		return "";
	}
	cv::Rect loc = blobs.boundingRect();
	blobs.move(-loc.x, -loc.y);
	float angle = deslant(loc.size(), blobs);
	loc = blobs.boundingRect();
//	removeDelimiter(loc, loc.y + loc.height / 2, blobs, angle);
	groupVertical(blobs);
	removeTerminator(blobs);
//	defragment(car, blobs);
	NumberRecognizer nr(blobs);
	//return extractDigit(blobs, angle);
	return nr.predict();
}

void ICREngine::trainWeight() {
	float step = 0.01;
	std::vector<float> weights {0, 0, 0, 0, 0};
	float bestCorrect = 0;
	for (float& w : weights) {
		float bestWeight = 0;
		for (int j = 0; j < 100 && w <= 1; ++j) {
			w += step;
			int correct = trainWeight(weights);
			std::cout << "correct: " << correct << std::endl;
			if (correct > bestCorrect) {
				bestCorrect = correct;
				bestWeight = w;
			}
		}
		w = bestWeight;
	}
	std::cout << "Best Correct: " << bestCorrect << " with weights: ";
	for (float w : weights) {
		std::cout << w << ", ";
	}
	std::cout << std::endl;
}

void ICREngine::trainWeightV2() {
	srand(time(NULL));
	unsigned long seed = time(NULL);
	std::default_random_engine generator(seed);
	float maxRange = 100;
	std::uniform_int_distribution<int> distribution(0, (int)maxRange);
	int wSize = 5;
	std::vector<float> weights(wSize, 0);
	for (int i = 0; i < wSize; ++i) {
		weights[i] = distribution(generator) / maxRange;
	};
	std::vector<float> bestWeights = weights;
	float bestCorrect = trainWeight(bestWeights);
	int correct = bestCorrect;
	int patience = 500;
	float temp = 200;
	float decrease = 0.99;
	for (int i = 0; i < patience ; ++i) {
		std::vector<float> nextWeights = weights;
		int idx = distribution(generator) % wSize;
		nextWeights[idx] = distribution(generator) / maxRange;
		int nextCorrect = trainWeight(nextWeights);
		if (nextCorrect > correct) {
			weights = nextWeights;
			correct = nextCorrect;
		} else if((temp = decrease * temp) > 1) {
			float rate = exp((nextCorrect - correct) / temp);
			if (rate > (rand() / (float) RAND_MAX)) {
				weights = nextWeights;
				correct = nextCorrect;
			}
		}
		if (correct > bestCorrect) {
			bestWeights = weights;
			bestCorrect = correct;
			std::cout << "Best Correct: " << bestCorrect << " with weights: ";
			for (float w : bestWeights) {
				std::cout << w << ", ";
			}
			std::cout << std::endl;
		}
	}
	std::cout << "Best Correct: " << bestCorrect << " with weights: ";
	for (float w : bestWeights) {
		std::cout << w << ", ";
	}
	std::cout << std::endl;
}

int ICREngine::trainWeight(std::vector<float>& weights) {
	unordered_map<std::string, std::string> labels;
	loadChequeLabel("train/labels.txt", labels);
	int correct = 0;
	for (auto it = labels.begin(), end = labels.end(); it != end; ++it) {
		cv::Mat cheque = cv::imread("/home/thienlong/cheque/500 Cheques/ValidChq/" + it->first, 0);
		CARLocator locator(cheque);
		Blobs blobs;
		locator.getHandwrittenBlobs(blobs);
		if (blobs.size() == 0) {
			continue;
		}
		cv::Rect loc = blobs.boundingRect();
		blobs.move(-loc.x, -loc.y);
		float angle = deslant(loc.size(), blobs);
		loc = blobs.boundingRect();
	//	removeDelimiter(loc, loc.y + loc.height / 2, blobs, angle);
		groupVertical(blobs);
		removeTerminator(blobs);
	//	defragment(car, blobs);
		NumberRecognizer nr(blobs, weights);
		//return extractDigit(blobs, angle);
		std::string predLabel = nr.predict();
		if (predLabel.compare(it->second) == 0) {
			++correct;
		}
	}
	return correct;
}


} /* namespace icr */
