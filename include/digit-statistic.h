/*
 * digit-statistic.h
 *
 *  Created on: Aug 26, 2014
 *      Author: thienlong
 */

#ifndef DIGIT_STATISTIC_H_
#define DIGIT_STATISTIC_H_

#include <string>
#include <vector>

#define NORMAL_DIGIT_HEIGHT 40
#define GET_NORMAL_DIGIT_WIDTH(w, h) ((w) * 40.0f / (h))

class DigitWidthStatistic {
public:
	double mean;
	double deviation;
};

void computeDigitWidth(const std::string& path, std::vector<DigitWidthStatistic>& output);

#endif /* DIGIT_STATISTIC_H_ */
