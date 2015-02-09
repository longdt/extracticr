#include <boost/filesystem.hpp>
#include <ICREngine.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iosfwd>
#include <string>
#include <vector>

#include "util/misc.h"
using boost::filesystem::directory_iterator;
using boost::filesystem::path;
std::string chqName;
int main()
{
	path p("/home/thienlong/cheque/500 Cheques/ValidChq");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	icr::ICREngine::loadModels("models");
	icr::ICREngine engine;
	int skip = 10 - 1; //*73 *169 313 330 398 *42 218
	int counter = 0;
	int end = 1000;
	for (directory_iterator iter(p), iterend; iter != iterend; ++iter) {
		++counter;
		if (counter <= skip) {
			continue;
		} else if (counter > end) {
			break;
		}
		cv::Mat cheque = cv::imread(iter->path().string(), 0);
//		cv::Mat cheque = cv::imread("/home/thienlong/cheque/test.jpeg", 0);
		if (cheque.empty())
			return -1;
		chqName = iter->path().filename().string();
		std::size_t pos = chqName.find(".");

		chqName = chqName.substr(0, pos);
		std::cout << counter << "\t" << iter->path().filename().string() << "\t";
		std::cout.flush();
		std::string amountStr = removeDelimiter(engine.recognite(cheque));
//		long double amount = std::stold(amountStr);
		std::cout << amountStr << std::endl;
		cv::waitKey(0);
//		cv::destroyAllWindows();
	}
    return 0;
}
