/*
 * NumberRecognizer.h
 *
 *  Created on: Oct 29, 2014
 *      Author: thienlong
 */

#ifndef INCLUDE_NUMBERRECOGNIZER_H_
#define INCLUDE_NUMBERRECOGNIZER_H_
#include <GeoContextModel.h>
#include "blob.h"
#include <vector>

#include "recognizer.h"
#include <utility>

namespace icr {
struct Segment {
	label_t label = -1;
	int start = 0;
	int end = 0;
};
#define ICR_UCG 0
#define ICR_BCG 1
#define ICR_UIG 2
#define ICR_BIG 3
#define ICR_BIG_IGNORE 4
#define ICR_UIG_IGNORE 5

class Path;
class Beam;
class NumberRecognizer {
private:
	static GeoContextModel gcm;
	NumberModel nm;
	std::vector<float> weights;
	float strHeight;
	float middleLine;
	Blobs segms;
	void genOverSegm(Blobs& blobs);
	void genVertCuts(Blob& blob, std::vector<int>& cuts);
	bool isTouching(Blob& blobs);
	void segment(Blobs& segms, Blob& blob);
	bool isCandidatePattern(int from, int end);
	void expandPath(Beam& beam, const std::vector<Path>& paths, int node);
	digit_recognizer::result recognizeBlob(Blobs& segms, int start, int end);
	std::pair<label_t, float> predictNode(std::vector<label_t>& labels, digit_recognizer::result rs, GeoContext& gc, bool lastNode, std::pair<label_t, float>* secChoice = NULL);
public:
	NumberRecognizer(Blobs &blobs, std::vector<float>& weights, bool decimal = true);
	NumberRecognizer(Blobs &blobs, bool decimal = true);
	static void loadModels(std::string& mpath);
	std::string predict(float* confidence = NULL);
	virtual ~NumberRecognizer();
	void genTrainData(std::vector<Segment>& segmsCnf, int dataType, std::vector<vec_t>& inputs, std::vector<label_t>& labels);
};
} /* namespace icr */
bool isPeriod(cv::Rect carBox, int middleLine, Blob& blob);

#endif /* SRC_NUMBERRECOGNIZER_H_ */
