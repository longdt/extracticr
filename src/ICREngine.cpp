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
#include <random>
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

//@deprecated
int detectTerminator(Blobs& blobs) {
	if (blobs.size() <= 1) {
		return -1;
	}
	int firstHeighest = 0;
	int idx1 = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		cv::Rect rect = blobs[i]->boundingRect();
		if (firstHeighest < rect.height) {
			firstHeighest = rect.height;
			idx1 = i;
		}
	}
	int secondHeighest = 0;
	int idx2 = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		cv::Rect rect = blobs[i]->boundingRect();
		if (secondHeighest < rect.height && i != idx1) {
			secondHeighest = rect.height;
			idx2 = i;
		}
	}
	if (secondHeighest == 0) {
		return -1;
	}
	float rate = firstHeighest / (float) secondHeighest;
	if ((rate >= 1.8 && idx1 > 1) || (idx1 > idx2 + 2 && rate >= 1.5)) {
		return idx1;
	}
//	else if (idx2 < idx1) {
//		return -1;
//	}

	secondHeighest = 0;
	for (size_t i = 0; i < idx1; ++i) {
		cv::Rect rect = blobs[i]->boundingRect();
		if (secondHeighest < rect.height) {
			secondHeighest = rect.height;
			idx2 = i;
		}
	}
	if (secondHeighest == 0) {
		return -1;
	}
	rate = firstHeighest / (float) secondHeighest;
	return rate > 1.5 ? idx1 : -1;
}

int detectTerm(Blobs& blobs) {
	if (blobs.size() <= 1) {
		return -1;
	}
	int firstHeighest = 0;
	int idx1 = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		cv::Rect rect = blobs[i]->boundingRect();
		if (firstHeighest < rect.height) {
			firstHeighest = rect.height;
			idx1 = i;
		}
	}
	float strHeight = 0;
	if (idx1 == 0) {
		return -1;
	} else if (idx1 == 1) {
		strHeight = blobs[0]->boundingRect().height;
		return -1;
	} else {
		float heightScore = 0;
		int sumHeightW = 0;
		for (int i = 0; i < idx1; ++i) {
			cv::Rect rect = blobs[i]->boundingRect();
			heightScore += rect.height * blobs[i]->points.size();
			sumHeightW += blobs[i]->points.size();
		}
		strHeight = heightScore / sumHeightW;
	}
	if (strHeight < 20 && idx1 == 1) {
		return -1;
	}
	float rate = firstHeighest / strHeight;
	if (rate <= 1.5) {
		return -1;
	} else if (rate > 1.8) {
		return idx1;
	}
	cv::Mat src = blobs.cropBlobs(idx1, idx1 + 1);
	auto digitMat = makeDigitMat(src);
	vec_t in;
	matToVect(digitMat, in);
	auto rs = recognizer.predict(in);
	int label = rs.label();
	if (label == 1 || (rs.confidence() < 0.1)) {
		return idx1;
	}
	return -1;
}

bool removeTerminator(Blobs& blobs) {
	int idx = detectTerm(blobs);
	if (idx > 0) {
		for (int i = blobs.size() - 1; i >= idx; --i) {
			blobs.erase(i);
		}
		return true;
	}
	return false;
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

ICREngine::ICREngine(int type) : chequeType(type) {
}

ICREngine::~ICREngine() {
}

void ICREngine::loadModels(std::string mpath) {
	NumberRecognizer::loadModels(mpath);
}

std::string ICREngine::recognite(cv::Mat& cheque, float* confidence) {
	CARLocator* locator = NULL;
	if (chequeType == CHEQUE_PH) {
		locator = new PhCARLocator(cheque);
	} else {
		locator = new PrintedCARLocator(cheque);
	}
	Blobs blobs;
	locator->getHandwrittenBlobs(blobs);
	delete locator;
	if (blobs.size() == 0) {
		return "";
	}
	cv::Rect loc = blobs.boundingRect();
	blobs.move(-loc.x, -loc.y);
	float angle = deslant(loc.size(), blobs);
	loc = blobs.boundingRect();
//	removeDelimiter(loc, loc.y + loc.height / 2, blobs, angle);
	bool hasTerm = false;
	if (chequeType == CHEQUE_MY) {
		hasTerm = removeTerminator(blobs);
	}
	groupVertical(blobs);
//	defragment(car, blobs);
	if (blobs.size() == 0) {
		return "";
	}
	NumberRecognizer nr(blobs, !hasTerm);
	//return extractDigit(blobs, angle);
	return nr.predict(confidence);
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
	float maxRange = 1000;
	std::uniform_int_distribution<int> distribution(0, (int)maxRange * 2);
	int wSize = 5;
//	std::vector<float> weights(wSize, 0);
//	for (int i = 0; i < wSize; ++i) {
//		weights[i] = distribution(generator) / maxRange;
//	};
	std::vector<float> weights{0.958f, 0.306f, 1.765f, 0.506f, 1.008f};
	std::vector<float> bestWeights = weights;
	float bestCorrect = trainWeight(bestWeights);
	std::cout << "Best Correct: " << bestCorrect << " with weights: ";
	for (float w : bestWeights) {
		std::cout << w << ", ";
	}
	std::cout << std::endl;
	int correct = bestCorrect;
	int patience = 1000;
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
	loadChequeLabel("test-labels.txt", labels);
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
		removeTerminator(blobs);
		groupVertical(blobs);
	//	defragment(car, blobs);
		if (blobs.size() == 0) {
			continue;
		}
		NumberRecognizer nr(blobs, weights);
		//return extractDigit(blobs, angle);
		std::string predLabel = nr.predict();
		std::string target = it->second;
		predLabel = removeDelimiter(predLabel);
		target = removeDelimiter(target);
		if (predLabel.compare(target) == 0) {
			++correct;
		}
	}
	return correct;
}


} /* namespace icr */
