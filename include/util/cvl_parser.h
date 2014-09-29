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
Mat distort(Mat& src);
cv::Mat resizeAspect(cv::Mat& src, int size);


void mat_to_vect(const cv::Mat& input, vec_t &dst);

void vect_to_mat(const vec_t &src, cv::Mat& output) {
	output = Mat::zeros(28, 28, CV_8UC1);
	int x_padding = 2;
	int y_padding = 2;
	const int width = output.cols + 2 * x_padding;
	const int height = output.rows + 2 * y_padding;
	float scale_min = -1.0;
	float scale_max = 1.0;
	for (size_t y = 0; y < output.rows; y++)
		for (size_t x = 0; x < output.cols; x++)
			output.at<uchar>(y, x) = (src[width * (y + y_padding) + x + x_padding] - scale_min) * 255.0 / (scale_max - scale_min);
}


void load_image(vec_t &dst, string file, bool preprocess) {
	Mat org = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
//	imshow("org", org);
	Mat img = org;
	if (preprocess) {
		img = 255 - org;
		//threshold(org, img, 0, 255, THRESH_BINARY_INV | CV_THRESH_OTSU);
	}
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


#define UNIFORM_PLUS_MINUS_ONE ( (double)(2.0 * rand())/RAND_MAX - 1.0 )
#define MAX_ROTATION 20
#define MAX_HEIGHT_SCALE 0.2
#define MAX_WIDTH_SCALE 0.2
#define PI 3.14159265

Mat distort(Mat& src) {
	Point2f srcTri[3];
	Point2f dstTri[3];
	/// Set your 3 points to calculate the  Affine Transform
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1.f, 0);
	srcTri[2] = Point2f(0, src.rows - 1.f);
	double degree = MAX_ROTATION * UNIFORM_PLUS_MINUS_ONE;
	double tag = tan(std::abs(degree) * PI / 180.0);
	int newRows = src.rows + src.rows * MAX_HEIGHT_SCALE * UNIFORM_PLUS_MINUS_ONE;
	int newCols = src.cols + src.cols * MAX_WIDTH_SCALE * UNIFORM_PLUS_MINUS_ONE;
	/// Set the dst image the same type and size as src
	Mat warp_dst = Mat::zeros(newRows, newCols + newRows * tag, src.type());
	if (degree > 0) {
		dstTri[0] = Point2f(0, 0);
		dstTri[1] = Point2f(newCols - 1, 0);
		dstTri[2] = Point2f(newRows * tag, newRows - 1.f);
	}
	else {
		dstTri[0] = Point2f(newRows * tag, 0);
		dstTri[1] = Point2f(newCols - 1.f + newRows * tag, 0);
		dstTri[2] = Point2f(0, newRows - 1.f);
	}
	/// Get the Affine Transform
	Mat warp_mat = getAffineTransform(srcTri, dstTri);

	/// Apply the Affine Transform just found to the src image
	warpAffine(src, warp_dst, warp_mat, warp_dst.size());
	Rect roi(newRows * tag / 2, 0, newCols, newRows);
	Mat center = warp_dst(roi);
	return resizeAspect(center, src.rows);
}


cv::Mat resizeAspect(cv::Mat& src, int size) {
	int width = 0;
	int height = 0;
	int paddingX = 0;
	int paddingY = 0;
	if (src.rows > src.cols) {
		//scale to height
		height = size;
		width = (height * src.cols) / src.rows;
	}
	else {
		width = size;
		//scale to width
		height = (width * src.rows) / src.cols;
	}
	cv::Size sizeDst(width, height);
	cv::Mat resize;
	cv::resize(src, resize, sizeDst);
	cv::Mat padded(size, size, CV_8UC1);
	padded.setTo(cv::Scalar::all(0));
	paddingX = (size - resize.cols) / 2;
	paddingY = (size - resize.rows) / 2;
	resize.copyTo(padded(cv::Rect(paddingX, paddingY, resize.cols, resize.rows)));
	//		cv::copyMakeBorder(resize, pad, 4, 4, 4, 4, cv::BORDER_CONSTANT, cv::Scalar(0));
	return padded;
}
#define GAUSSIAN_FIELD_SIZE 21
void generateElasticMap(Mat& dispH, Mat& dispV, Size size, float sigma, float scale) {
	int rows = size.height;
	int cols = size.width;
	Mat uniformH(rows, cols, CV_32FC1);
	Mat uniformV(rows, cols, CV_32FC1);
	for (int r = 0; r < rows; ++r) {
		for (int c = 0; c < cols; ++c) {
			uniformH.at<float>(r, c) = UNIFORM_PLUS_MINUS_ONE;
			uniformV.at<float>(r, c) = UNIFORM_PLUS_MINUS_ONE;
		}
	}
	// filter with gaussian
	GaussianBlur(uniformH, dispH, Size(GAUSSIAN_FIELD_SIZE, GAUSSIAN_FIELD_SIZE), sigma);
	GaussianBlur(uniformV, dispV, Size(GAUSSIAN_FIELD_SIZE, GAUSSIAN_FIELD_SIZE), sigma);
	dispH = dispH  * scale;
	dispV = dispV * scale;
}

void distortElastic(const Mat& src, const Mat& dispH, const Mat& dispV, Mat& dst) {
	dst = Mat(src.size(), src.type());
	double sourceRow, sourceCol;
	double fracRow, fracCol;
	double w1, w2, w3, w4;
	double sourceValue;
	int sRow, sCol, sRowp1, sColp1;
	bool bSkipOutOfBounds;
	for (int row = 0; row < src.rows; ++row)
	{
		for (int col = 0; col < src.cols; ++col)
		{
			// the pixel at sourceRow, sourceCol is an "phantom" pixel that doesn't really exist, and
			// whose value must be manufactured from surrounding real pixels (i.e., since
			// sourceRow and sourceCol are floating point, not ints, there's not a real pixel there)
			// The idea is that if we can calculate the value of this phantom pixel, then its
			// displacement will exactly fit into the current pixel at row, col (which are both ints)

			sourceRow = (double)row - dispV.at<float>(row, col);
			sourceCol = (double)col - dispH.at<float>(row, col);

			// weights for bi-linear interpolation

			fracRow = sourceRow - (int)sourceRow;
			fracCol = sourceCol - (int)sourceCol;


			w1 = (1.0 - fracRow) * (1.0 - fracCol);
			w2 = (1.0 - fracRow) * fracCol;
			w3 = fracRow * (1 - fracCol);
			w4 = fracRow * fracCol;


			// limit indexes

			/*
			while (sourceRow >= m_cRows ) sourceRow -= m_cRows;
			while (sourceRow < 0 ) sourceRow += m_cRows;

			while (sourceCol >= m_cCols ) sourceCol -= m_cCols;
			while (sourceCol < 0 ) sourceCol += m_cCols;
			*/
			bSkipOutOfBounds = false;

			if ((sourceRow + 1.0) >= src.rows)	bSkipOutOfBounds = true;
			if (sourceRow < 0)				bSkipOutOfBounds = true;

			if ((sourceCol + 1.0) >= src.cols)	bSkipOutOfBounds = true;
			if (sourceCol < 0)				bSkipOutOfBounds = true;

			if (bSkipOutOfBounds == false)
			{
				// the supporting pixels for the "phantom" source pixel are all within the
				// bounds of the character grid.
				// Manufacture its value by bi-linear interpolation of surrounding pixels

				sRow = (int)sourceRow;
				sCol = (int)sourceCol;

				sRowp1 = sRow + 1;
				sColp1 = sCol + 1;

				while (sRowp1 >= src.rows) sRowp1 -= src.rows;
				while (sRowp1 < 0) sRowp1 += src.rows;

				while (sColp1 >= src.cols) sColp1 -= src.cols;
				while (sColp1 < 0) sColp1 += src.cols;

				// perform bi-linear interpolation

				sourceValue = w1 * src.at<uchar>(sRow, sCol) +
					w2 * src.at<uchar>(sRow, sColp1) +
					w3 * src.at<uchar>(sRowp1, sCol) +
					w4 * src.at<uchar>(sRowp1, sColp1);
			}
			else
			{
				// At least one supporting pixel for the "phantom" pixel is outside the
				// bounds of the character grid. Set its value to "background"

				sourceValue = 0;
			}

			dst.at<uchar>(row, col) = sourceValue;
		}
	}
}


#define USE_ELASTIC_DISTORTION
void applyDistortion(vector<vec_t>& images, vector<label_t>& labels) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine generator(seed);
	std::uniform_int_distribution<int> distribution(0, images.size() - 1);
	vector<bool> visited(images.size(), false);
	Mat img;
	vect_to_mat(images[0], img);
	Mat dispH, dispV;

	for (int i = 0; i < 3; ++i) {
		fill(visited.begin(), visited.end(), false);
		int counter = 0;
		generateElasticMap(dispH, dispV, img.size(), 4, 20);

		while (counter < visited.size()) {
			int idx = distribution(generator);
			if (visited[idx]) {
				continue;
			}
			vect_to_mat(images[idx], img);
			Mat distortion;
#ifdef USE_ELASTIC_DISTORTION
			distortElastic(img, dispH, dispV, distortion);
#else
			distortion = distort(img);
#endif
			vec_t vec;
			mat_to_vect(distortion, vec);
			images.push_back(vec);
			labels.push_back(labels[idx]);
			visited[idx] = true;
			++counter;
		}
	}
}
