/*
 * ICREngine.h
 *
 *  Created on: Oct 24, 2014
 *      Author: thienlong
 */

#ifndef INCLUDE_ICRENGINE_H_
#define INCLUDE_ICRENGINE_H_
#include <GeoContextModel.h>
#include <string>

#include "opencv2/core/core.hpp"
#include <vector>

#include "digit-statistic.h"
namespace icr {
#define CHEQUE_MY 0
#define CHEQUE_PH 1
class ICREngine {
private:
	int chequeType;
	int trainWeight(std::vector<float>& weights);
public:
	ICREngine(int type = CHEQUE_MY);
	static void loadModels(std::string mpath);
	std::string recognite(cv::Mat& cheque, float* confidence = NULL);
	void trainWeight();
	void trainWeightV2();
	virtual ~ICREngine();
};

} /* namespace icr */

#endif /* INCLUDE_ICRENGINE_H_ */
