#include <opencv2/core/core.hpp>
#include "cnn/util.h"
#include <algorithm>
#include <vector>
using namespace tiny_cnn;
using namespace std;
void mat_to_vect(const cv::Mat& input, vec_t &dst);

string parse_label(string filename);

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

	T mean() {
		return sum / values.size();
	}

	T deviation() {
		T m = mean();
		T devSum(0);
		for (T v : values) {
			devSum += abs(m - v);
		}
		return devSum / values.size();
	}

	T sdeviation() {
		T m = mean();
		T devSum(0);
		for (T v : values) {
			devSum += (m - v) * (m - v);
		}
		return sqrt(devSum / (double) values.size());
	}

	int size() {
		return values.size();
	}
};
