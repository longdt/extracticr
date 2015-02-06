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

class ICREngine {
private:
	int trainWeight(std::vector<float>& weights);
public:
	ICREngine();
	static void loadModels(std::string mpath);
	std::string recognite(cv::Mat& cheque);
	void trainWeight();
	void trainWeightV2();
	virtual ~ICREngine();
};

} /* namespace icr */

#endif /* INCLUDE_ICRENGINE_H_ */
