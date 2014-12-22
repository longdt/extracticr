#include <GeoContextModel.h>
#include <opencv2/core/core.hpp>
#include "cnn/util.h"
#include <algorithm>
#include <vector>
#include <unordered_map>

using icr::GeoContext;
using namespace tiny_cnn;
using namespace std;
void matToVect(const cv::Mat& input, vec_t &dst);

string parse_label(string filename);
void loadChequeLabel(string filename, unordered_map<string, string>& dst);
cv::Mat projectTop(const cv::Mat& src);
cv::Mat projectBottom(const cv::Mat& src);
void toFile(GeoContext& gc, std::string file);
int commonArea(cv::Rect r1, cv::Rect r2);
bool intersect(cv::Rect r1, cv::Rect r2);
template<class T> class average {
private:
	vector<T> values;
	T sum;
public:
	average() : sum(0) {}
	void update(T value) {
		values.push_back(value);
		sum += value;
	}

	double mean() {
		return sum /(double) values.size();
	}

	double deviation() {
		double m = mean();
		double devSum(0);
		for (T v : values) {
			devSum += abs(m - v);
		}
		return devSum / values.size();
	}

	double sdeviation() {
		double m = mean();
		double devSum(0);
		for (T v : values) {
			devSum += (m - v) * (m - v);
		}
		return sqrt(devSum / (double) values.size());
	}

	int size() {
		return values.size();
	}
};
