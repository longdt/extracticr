/*
 * CombineWeightTrainer.cpp
 *
 *  Created on: Dec 15, 2014
 *      Author: thienlong
 */
#include <ICREngine.h>
#include <iosfwd>




int cwmain(int argc, char **argv) {
	icr::ICREngine engine;
	engine.trainWeightV2();
	std::cin.get();
}
