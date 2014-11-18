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
namespace icr {
class Path;
class Beam;
class NumberRecognizer {
private:
	static NumberModel nm;
	static GeoContextModel gcm;
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
public:
	NumberRecognizer(Blobs &blobs);
	std::string predict();
	virtual ~NumberRecognizer();
	void genTrainData();
};
} /* namespace icr */
bool isPeriod(cv::Rect carBox, int middleLine, Blob& blob);

#endif /* SRC_NUMBERRECOGNIZER_H_ */
