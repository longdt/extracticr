#include <boost/filesystem.hpp>
#include <ICREngine.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iosfwd>
#include <vector>

using boost::filesystem::directory_iterator;
using boost::filesystem::path;
std::vector<DigitWidthStatistic> digitStatistics;

int main()
{
	computeDigitWidth("/media/thienlong/linux/CAR/cvl-digits/train", digitStatistics);
	path p("/home/thienlong/cheque/500 Cheques/ValidChq");
	if (!exists(p) || !is_directory(p)) {
		return 0;
	}
	icr::ICREngine engine;
	for (directory_iterator iter(p), iterend; iter != iterend; ++iter) {
		cv::Mat cheque = cv::imread(iter->path().string(), 0);
//		cv::Mat cheque = cv::imread("/home/thienlong/cheque/500 Cheques/ValidChq/chq_00458_00.jpeg", 0);
		if (cheque.empty())
			return -1;
		std::cout << iter->path().filename().string() << ": ";
		std::cout.flush();
		std::cout << engine.recognite(cheque) << std::endl;
		cv::waitKey(0);
		cv::destroyAllWindows();
	}
    return 0;
}
