/*
 * NumberRecognizer.h
 *
 *  Created on: Oct 29, 2014
 *      Author: thienlong
 */

#ifndef INCLUDE_NUMBERRECOGNIZER_H_
#define INCLUDE_NUMBERRECOGNIZER_H_
#include "blob.h"
#include <vector>

namespace icr {
class Path;
class Beam;
class NumberRecognizer {
private:
	int strHeight;
	int middleLine;
	Blobs segms;
	void estHeightVertCenter(Blobs& blobs);
	void genOverSegm(Blobs& blobs);
	void genVertCuts(Blob& blob, std::vector<int>& cuts);
	bool isTouching(Blob& blobs);
	void segment(Blobs& segms, Blob& blob);
	bool isCandidatePattern(int from, int end);
	void expandPath(Beam& beam, const std::vector<Path>& paths, int node);
public:
	NumberRecognizer(Blobs &blobs);
	std::string predict();
	virtual ~NumberRecognizer();
};

} /* namespace icr */

#endif /* SRC_NUMBERRECOGNIZER_H_ */
