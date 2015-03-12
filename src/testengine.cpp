#include <boost/filesystem.hpp>
#include <ICREngine.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iosfwd>
#include <string>
#include <vector>

#include "util/misc.h"
#include <algorithm>
using boost::filesystem::directory_iterator;
using boost::filesystem::path;
std::string chqName;
int main()
{
	path p("/home/thienlong/cheque/500 Cheques/ValidChq");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	vector<path> files;
	std::copy(directory_iterator(p), directory_iterator(), std::back_inserter(files));
	std::sort(files.begin(), files.end());
	icr::ICREngine::loadModels("models");
	icr::ICREngine engine;
	int skip = 1 - 1; //*73 *169 313 330 398 *42 *78
	int counter = 0;
	int end = 1000;
	for (auto iter = files.begin(), iterend = files.end(); iter != iterend; ++iter) {
		++counter;
		if (counter <= skip) {
			continue;
		} else if (counter > end) {
			break;
		}
		cv::Mat cheque = cv::imread(iter->string(), 0);
//		cv::Mat cheque = cv::imread("/home/thienlong/cheque/500 Cheques/ValidChq/chq_00461_00.jpeg", 0);
		if (cheque.empty())
			return -1;
		chqName = iter->filename().string();
		std::size_t pos = chqName.find(".");

		chqName = chqName.substr(0, pos);
		std::cout << counter << "\t" << iter->filename().string() << "\t";
		std::cout.flush();
		std::string amountStr = removeDelimiter(engine.recognite(cheque));
//		long double amount = std::stold(amountStr);
		std::cout << amountStr << std::endl;
		cv::waitKey(0);
//		cv::destroyAllWindows();
	}
    return 0;
}
