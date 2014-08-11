#include <string>
#include <vector>
#include "cnn/util.h"
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <random>       // std::default_random_engine
#include <chrono>
using namespace cv;
using namespace boost::filesystem;
using namespace std;
using namespace tiny_cnn;

void load_image(vec_t &dst, string file, bool preprocess = true);

void load_image(vec_t &dst, string file, bool preprocess) {
	Mat org = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//	imshow("org", org);
	Mat img = org;
	if (preprocess)
		threshold(org, img, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
//	imshow("binary", img);
//	waitKey(1000);
//	cout << img.cols << "x" << img.rows << endl;
	int x_padding = 2;
	int y_padding = 2;
	const int width = img.cols + 2 * x_padding;
	const int height = img.rows + 2 * y_padding;
	float scale_min = -1.0;
	float scale_max = 1.0;
	dst.resize(width * height, scale_min);
	for (size_t y = 0; y < img.rows; y++)
		for (size_t x = 0; x < img.cols; x++)
			dst[width * (y + y_padding) + x + x_padding]
			= (img.at<uchar>(y, x) / 255.0) * (scale_max - scale_min) + scale_min;
}

void parse_cvl(char* folder, vector<vec_t>& images, vector<label_t>& labels) {
	path p(folder);
	  try
  {
	  if (exists(p) && is_directory(p)) {
		  typedef vector<path> vec;             // store paths,
		  vec v;                                // so we can sort them later
		  copy(directory_iterator(p), directory_iterator(), back_inserter(v));
		  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

		  shuffle(v.begin(), v.end(), std::default_random_engine(seed));
		  for (vec::const_iterator it(v.begin()), it_end(v.end()); it != it_end; ++it)
		  {
				  string label = it->filename().string();
				  std::size_t found = label.find("-");
				  if (found != std::string::npos) {
					  label = label.substr(0, found);
					  labels.push_back(stoi(label));
					  vec_t in;
					  load_image(in, it->string());
					  images.push_back(in);
				  }
			  
		  }
      }
  } catch (const filesystem_error& ex)
  {
    cout << ex.what() << '\n';
  }

}