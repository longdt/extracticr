/*
 * BFSegment.h
 *
 *  Created on: Sep 30, 2014
 *      Author: thienlong
 */

#ifndef BFSEGMENT_H_
#define BFSEGMENT_H_
#include <array>
#include <vector>

#include "opencv2/core/core.hpp"

using namespace std;
using namespace cv;
class BFSegment {
private:
	vector<Point> fpLayers[4];
	Mat binImg;
	bool broken;
	void genCutPath(int maxDigitWidth, vector<vector<Point>>& cutPaths, bool down);
	void genCutPath(Point p, vector<Point>& cut, int layerIdx, int maxDigitWidth, vector<vector<Point> >& cutPaths, bool down);
public:
	BFSegment(Mat& binImg);
	void genFPLayers();
	void genCutPath(int maxDigitWidth, vector<vector<Point>>& cutPaths);
	virtual ~BFSegment();
};

#endif /* BFSEGMENT_H_ */
