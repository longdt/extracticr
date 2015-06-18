
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <sstream>
#include "recognizer.h"
#include "util/misc.h"
#include "preprocessor.h"

using namespace tiny_cnn;
using namespace cv;


extern digit_recognizer recognizer;

void cv::doNothing() {}

#define uget(x,y)    at<unsigned char>(y,x)
#define uset(x,y,v)  at<unsigned char>(y,x)=v;
#define fget(x,y)    at<float>(y,x)
#define fset(x,y,v)  at<float>(y,x)=v;

// *************************************************************
// glide a window across the image and
// create two maps: mean and standard deviation.
// *************************************************************
//#define BINARIZEWOLF_VERSION  "2.3 (February 26th, 2013)"
double calcLocalStats (Mat &im, Mat &map_m, Mat &map_s, int win_x, int win_y) {

    double m,s,max_s, sum, sum_sq, foo;
    int wxh = win_x / 2;
    int wyh = win_y / 2;
    int x_firstth = wxh;
    int y_lastth = im.rows-wyh-1;
    int y_firstth= wyh;
    double winarea = win_x*win_y;

    max_s = 0;
    for (int j = y_firstth ; j<=y_lastth; j++)
    {
        // Calculate the initial window at the beginning of the line
        sum = sum_sq = 0;
        for (int wy=0 ; wy<win_y; wy++)
            for (int wx=0 ; wx<win_x; wx++) {
                foo = im.uget(wx,j-wyh+wy);
                sum    += foo;
                sum_sq += foo*foo;
            }
        m  = sum / winarea;
        s  = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
        if (s > max_s)
            max_s = s;
        map_m.fset(x_firstth, j, m);
        map_s.fset(x_firstth, j, s);

        // Shift the window, add and remove new/old values to the histogram
        for (int i=1 ; i <= im.cols  -win_x; i++) {

            // Remove the left old column and add the right new column
            for (int wy=0; wy<win_y; ++wy) {
                foo = im.uget(i-1,j-wyh+wy);
                sum    -= foo;
                sum_sq -= foo*foo;
                foo = im.uget(i+win_x-1,j-wyh+wy);
                sum    += foo;
                sum_sq += foo*foo;
            }
            m  = sum / winarea;
            s  = sqrt ((sum_sq - (sum*sum)/winarea)/winarea);
            if (s > max_s)
                max_s = s;
            map_m.fset(i+wxh, j, m);
            map_s.fset(i+wxh, j, s);
        }
    }

    return max_s;
}




void NiblackSauvolaWolfJolion (InputArray _src, OutputArray _dst,const BhThresholdMethod &version,int winx, int winy, double k, double dR) {

    Mat src = _src.getMat();
    Mat dst = _dst.getMat();
    double m, s, max_s;
    double th=0;
    double min_I, max_I;
    int wxh = winx/2;
    int wyh = winy/2;
    int x_firstth= wxh;
    int x_lastth = src.cols-wxh-1;
    int y_lastth = src.rows-wyh-1;
    int y_firstth= wyh;
    int mx, my;

    // Create local statistics and store them in a double matrices
    Mat map_m = Mat::zeros (src.size(), CV_32FC1);
    Mat map_s = Mat::zeros (src.size(), CV_32FC1);
    max_s = calcLocalStats (src, map_m, map_s, winx, winy);

    minMaxLoc(src, &min_I, &max_I);

    Mat thsurf (src.size(), CV_32FC1);

    // Create the threshold surface, including border processing
    // ----------------------------------------------------

    for (int j = y_firstth ; j<=y_lastth; j++) {

        // NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
        for (int i=0 ; i <= src.cols-winx; i++) {

            m  = map_m.fget(i+wxh, j);
            s  = map_s.fget(i+wxh, j);

            // Calculate the threshold
            switch (version) {

            case BhThresholdMethod::NIBLACK:
                    th = m + k*s;
                    break;

            case BhThresholdMethod::SAUVOLA:
                    th = m * (1 + k*(s/dR-1));
                    break;

            case BhThresholdMethod::WOLFJOLION:
                    th = m + k * (s/max_s-1) * (m-min_I);
                    break;

                default:
                    cerr << "Unknown threshold type in ImageThresholder::surfaceNiblackImproved()\n";
                    exit (1);
            }

            thsurf.fset(i+wxh,j,th);

            if (i==0) {
                // LEFT BORDER
                for (int i=0; i<=x_firstth; ++i)
                    thsurf.fset(i,j,th);

                // LEFT-UPPER CORNER
                if (j==y_firstth)
                    for (int u=0; u<y_firstth; ++u)
                    for (int i=0; i<=x_firstth; ++i)
                        thsurf.fset(i,u,th);

                // LEFT-LOWER CORNER
                if (j==y_lastth)
                    for (int u=y_lastth+1; u<src.rows; ++u)
                    for (int i=0; i<=x_firstth; ++i)
                        thsurf.fset(i,u,th);
            }

            // UPPER BORDER
            if (j==y_firstth)
                for (int u=0; u<y_firstth; ++u)
                    thsurf.fset(i+wxh,u,th);

            // LOWER BORDER
            if (j==y_lastth)
                for (int u=y_lastth+1; u<src.rows; ++u)
                    thsurf.fset(i+wxh,u,th);
        }

        // RIGHT BORDER
        for (int i=x_lastth; i<src.cols; ++i)
            thsurf.fset(i,j,th);

        // RIGHT-UPPER CORNER
        if (j==y_firstth)
            for (int u=0; u<y_firstth; ++u)
            for (int i=x_lastth; i<src.cols; ++i)
                thsurf.fset(i,u,th);

        // RIGHT-LOWER CORNER
        if (j==y_lastth)
            for (int u=y_lastth+1; u<src.rows; ++u)
            for (int i=x_lastth; i<src.cols; ++i)
                thsurf.fset(i,u,th);
    }
    cerr << "surface created" << endl;


    for (int y=0; y<src.rows; ++y)
    for (int x=0; x<src.cols; ++x)
    {
        if (src.uget(x,y) >= thsurf.fget(x,y))
        {
            dst.uset(x,y,255);
        }
        else
        {
            dst.uset(x,y,0);
        }
    }
}

void doThreshold(InputArray _src ,OutputArray _dst,const BhThresholdMethod &method)
{
    Mat src = _src.getMat();

    int winx = 0;
    int winy = 0;
    float optK=0.5;
    if (winx==0 || winy==0) {
        winy = (int) (2.0 * src.rows - 1)/3;
        winx = (int) src.cols-1 < winy ? src.cols-1 : winy;

        // if the window is too big, than we asume that the image
        // is not a single text box, but a document page: set
        // the window size to a fixed constant.
        if (winx > 100)
            winx = winy = 40;
    }

    // Threshold
    _dst.create(src.size(), CV_8UC1);
    Mat dst = _dst.getMat();

    //medianBlur(src,dst,5);
    GaussianBlur(src,dst,Size(5,5),0);
//#define _BH_SHOW_IMAGE
#ifdef _BH_DEBUG
    #define _BH_SHOW_IMAGE
#endif
    //medianBlur(src,dst,7);
    switch (method)
    {
    case BhThresholdMethod::OTSU :
        threshold(dst,dst,128,255,CV_THRESH_OTSU);
        break;
    case BhThresholdMethod::SAUVOLA :
    case BhThresholdMethod::WOLFJOLION :
        NiblackSauvolaWolfJolion (src, dst, method, winx, winy, optK, 128);


    }

    bitwise_not(dst,dst);


#ifdef _BH_SHOW_IMAGE

#undef _BH_SHOW_IMAGE
#endif
}

void removeNoise(cv::Mat& img) {
	Mat bi;
	cv::threshold(img, bi, 127, 1, THRESH_BINARY);
	Blobs blobs;
	findBlobs(bi, blobs);
	int minp = img.cols * img.rows;
	int maxp = 0;
	float mina = img.cols;
	float maxa = 0;
	for (int i = 0; i < blobs.size(); ++i) {
		auto blob = blobs[i];
		minp = std::min(minp, (int) blob->points.size());
		maxp = std::max(maxp, (int) blob->points.size());
		auto rect = blob->boundingRect();
		float aspect = rect.height / (float) rect.width;
		mina = std::min(mina, aspect);
		maxa = std::max(maxa, aspect);
	}
	for (int i = 0; i < blobs.size(); ++i) {
		auto blob = blobs[i];
		float tp = (blob->points.size() - minp) / (float) (maxp - minp);
		auto rect = blob->boundingRect();
		float aspect = rect.height / (float) rect.width;
		float ta = (aspect - mina) / (float) (maxa - mina);
		if (tp < 0.002 || ta < 0.002) {
			blobs.erase(i);
			--i;
		}
	}
	drawBinaryBlobs(blobs, img);
}

void removeNoise(Blobs& blobs) {
	float mina = FLT_MAX;
	float maxa = 0;
	for (int i = 0; i < blobs.size(); ++i) {
		auto blob = blobs[i];
		auto rect = blob->boundingRect();
		float aspect = rect.height / (float) rect.width;
		mina = std::min(mina, aspect);
		maxa = std::max(maxa, aspect);
	}
	for (int i = 0; i < blobs.size(); ++i) {
		auto blob = blobs[i];
		auto rect = blob->boundingRect();
		float aspect = rect.height / (float) rect.width;
		float ta = (aspect - mina) / (float) (maxa - mina);
		if (rect.height == 1 || (ta < 0.002 && rect.height < 5) || (aspect < 0.2 && rect.height < 3)) {
			blobs.erase(i);
			--i;
		}
	}
}

cv::Mat cropBlob(Blob& blob, int pad) {
	cv::Rect bound = blob.boundingRect();
	cv::Mat rs = cv::Mat::zeros(bound.height + 2 * pad, bound.width + 2 * pad, CV_8UC1);
	for (size_t j = 0; j < blob.points.size(); j++) {
		int x = blob.points[j].x - bound.x + pad;
		int y = blob.points[j].y - bound.y + pad;

		rs.at<uchar>(y, x) = 255;
	}
	return rs;
}

bool cropMat(cv::Mat& src, cv::Mat& dst, int pad) {
	auto rect = getROI(src);
	if (rect.width == 0) {
		return false;
	}
	if (pad > 0) {
		dst = cv::Mat::zeros(rect.height + 2 * pad, rect.width + 2 * pad, src.type());
		src(rect).copyTo(dst(cv::Rect(pad, pad, rect.width, rect.height)));
	} else {
		dst = src(rect);
	}
	return true;
}

cv::Rect getROI(cv::Mat& src) {
	int minX = src.cols - 1;
	int maxX = 0;
	int minY = src.rows - 1;
	int maxY = 0;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			if (src.at<uchar>(i,j) > 0) {
				if (minX > j) {
					minX = j;
				}
				if (maxX < j) {
					maxX = j;
				}
				if (minY > i) {
					minY = i;
				}
				if (maxY < i) {
					maxY = i;
				}
			}
		}
	}
	//handle black iamge
	if (maxX < minX) {
		return cv::Rect();
	}
	return cv::Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
}

void projectHorizontal(cv::Mat &input, std::vector<int> &output) {
	output.resize(input.rows);
	for (int i = 0; i < input.rows; ++i) {
		int sum = 0;
		for (int j = 0; j < input.cols; ++j) {
			if (input.at<uchar>(i, j) > 0) {
				sum += 1;
			}
		}
		output[i] = sum;
	}
}

void projectVertical(cv::Mat &input, std::vector<int> &output) {
	output.resize(input.cols);
	for (int i = 0; i < input.cols; ++i) {
		int sum = 0;
		for (int j = 0; j < input.rows; ++j) {
			if (input.at<uchar>(j, i) > 0) {
				sum += 1;
			}
		}
		output[i] = sum;
	}
}

void projectVertical(Blob& blob, std::vector<int>& output) {
	cv::Rect box = blob.boundingRect();
	output.resize(box.width);
	std::fill(output.begin(), output.end(), 0);
	for (auto p : blob.points) {
		output[p.x - box.x] += 1;
	}
}


void projectVertical(Blobs &blobs, std::vector<int> &output) {
	std::fill(output.begin(), output.end(), 0);
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		for (auto p : b->points) {
			output[p.x] += 1;
		}
	}
}

void projectHorizontal(Blob& blob, std::vector<int>& output) {
	cv::Rect box = blob.boundingRect();
	output.resize(box.height);
	std::fill(output.begin(), output.end(), 0);
	for (auto p : blob.points) {
		output[p.y - box.y] += 1;
	}
}

int findLastMin(std::vector<int>& vec, int from, int last) {
	int minVal = 99999999;
	int minIdx = last;
	for (int i = from; i < last; ++i) {
		if (vec[i] == 0) {
			return i;
		} else if (vec[i] <= minVal) {
			minVal = vec[i];
			minIdx = i;
		}
	}
	return minIdx;
}

std::vector<int> genVerticalCuts(std::vector<int>& projectV) {
	std::vector<int> rs;
	int start = 0;
	int i = 0;
	int height = *std::max_element(projectV.begin(), projectV.end());
	while (i < projectV.size()) {
		//find start
		for (; i < projectV.size() && projectV[i] == 0; ++i) {}
		start = i;
		//find end = arg(min([start-> start + maxSgmentLen)
		bool inscrea = false;
		int maxSegmentLen = std::min(start + height, (int)projectV.size());
		int end = findLastMin(projectV, start + 1, maxSegmentLen);
		rs.push_back(start);
		rs.push_back(end);
		i = end;
	}
	return rs;
}


float objectWidth(cv::Mat& input) {
	float width = 0;
	for (int i = 0; i < input.cols; ++i) {
		for (int j = 0; j < input.rows; ++j) {
			if (input.at<uchar>(j, i) > 0) {
				width += 1;
				break;
			}
		}
	}
	return width;
}


float blobsWidth(cv::Mat& input) {
	Blobs blobs;
	findBlobs(input, blobs);
	float cost = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		cost += b->boundingRect().width;
	}
	cost = objectWidth(input) * 0.8 + cost * 0.2;
	return cost;
}

#define PI 3.14159265


cv::Mat slant(cv::Mat& src, float degree) {
	Point2f srcTri[3];
	Point2f dstTri[3];
	/// Set the dst image the same type and size as src
	Mat warp_dst = Mat::zeros(src.rows, src.cols + src.rows, src.type());

	/// Set your 3 points to calculate the  Affine Transform
	srcTri[0] = Point2f(0, 0);
	srcTri[1] = Point2f(src.cols - 1.f, 0);
	srcTri[2] = Point2f(0, src.rows - 1.f);
	double tag = tan(std::abs(degree) * PI / 180.0);
	if (degree > 0) {
		dstTri[0] = Point2f(0, 0);
		dstTri[1] = Point2f(src.cols - 1, 0);
		dstTri[2] = Point2f(src.rows * tag, src.rows - 1.f);
	}
	else {
		dstTri[0] = Point2f(src.rows * tag, src.rows*0.0f);
		dstTri[1] = Point2f(src.cols - 1.f + src.rows * tag, src.rows*0.0f);
		dstTri[2] = Point2f(0, src.rows - 1.f);
	}
	/// Get the Affine Transform
	Mat warp_mat = getAffineTransform(srcTri, dstTri);

	/// Apply the Affine Transform just found to the src image
	warpAffine(src, warp_dst, warp_mat, warp_dst.size());
	return warp_dst;
}

float resolveBlobAngle(Blob& blob, int imgHeight, float imgSlantAngle) {
	double tag = tan(std::abs(imgSlantAngle) * PI / 180.0);
	float blobMove = 0;
	auto rect = blob.boundingRect();
	if (imgSlantAngle > 0) {
		blobMove = tag * (rect.y + rect.height);
	} else {
		blobMove = tag * (rect.y - imgHeight);
	}
	float result = atan(blobMove / rect.height) * 180 / PI;
	return result;
}
/* input[0,1] output[0,1] */
float deslant(cv::Mat& input, cv::Mat *dst, float (*fntSlantCost)(cv::Mat&)) {
	float width = fntSlantCost(input);
	float minWidth = width;
	float degree = 0;
	float stepDegree = degree;
	cv::Mat rotated;
	//try rotate +/-5degree
	int step = 16;
	while (step != 0 && (degree <= 48 && degree >= -48)) {
		rotated = slant(input, degree + step);
		width = fntSlantCost(rotated);
		if (width < minWidth) {
			minWidth = width;
			degree += step;
			continue;
		}
		else if (degree != stepDegree) {
			step = step / 2;
			stepDegree = degree;
			continue;
		}
		step = -step;
		rotated = slant(input, degree + step);
		width = fntSlantCost(rotated);
		if (width < minWidth) {
			minWidth = width;
			degree += step;
			continue;
		}
		step = step / 2;
		stepDegree = degree;
		continue;
	}
	if (dst != NULL && degree != 0) {
		*dst = slant(input, degree);
	}
	return degree;
}

/*implement new deslant function*/

void genSlantShiftX(float angle, int imgHeight, std::vector<int>& moveX) {
	moveX.resize(imgHeight);
	double tag = tan(std::abs(angle) * PI / 180.0);
	int x = 0;
	double error = 0;
	int increasement = angle >= 0 ? 1 : -1;
	for (int y = 0; y < imgHeight; ++y) {
		moveX[y] = x;
		error += tag;
		if (error >= 0.5) {
			x += increasement;
			--error;
		}
	}
}

void slant(int imgHeight, Blobs& blobs, float angle) {
	std::vector<int> moveX;
	genSlantShiftX(angle, imgHeight, moveX);
	int padding = angle >= 0 ? 0 : -moveX[imgHeight - 1];
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		for (auto &p : b->points) {
			p.x += moveX[p.y] + padding;
		}
		b->setModify(true);
	}
}

float slantCost(Size imgSize, Blobs& blobs, float angle) {
	float bcost = 0;
	float widCost = 0;
	std::vector<int> moveX;
	genSlantShiftX(angle, imgSize.height, moveX);
	int padding = angle >= 0 ? 0 : -moveX[imgSize.height - 1];
	int newWidth = imgSize.width + std::abs(moveX[imgSize.height - 1]);
	std::vector<bool> pwImg(newWidth, false);
	int newX = 0;
	for (size_t i = 0; i < blobs.size(); ++i) {
		Blob* b = blobs[i];
		std::vector<bool> pwBlob(newWidth, false);
		for (auto &p : b->points) {
			newX = p.x + moveX[p.y] + padding;
			if (!pwImg[newX]) {
				pwImg[newX] = true;
				pwBlob[newX] = true;
				++widCost;
				++bcost;
			} else if (!pwBlob[newX]) {
				pwBlob[newX] = true;
				++bcost;
			}
		}
	}
	return widCost * 0.8 + bcost * 0.2;
}

float deslant(Size imgSize, Blobs& blobs) {
	int cost = slantCost(imgSize, blobs, 0);
	int minCost = cost;
	float degree = 0;
	float stepDegree = degree;
	//try rotate +/-5degree
	int step = 16;
	while (step != 0 && (degree <= 48 && degree >= -48)) {
		cost = slantCost(imgSize, blobs, degree + step);
		if (cost < minCost) {
			minCost = cost;
			degree += step;
			continue;
		}
		else if (degree != stepDegree) {
			step = step / 2;
			stepDegree = degree;
			continue;
		}
		step = -step;
		cost = slantCost(imgSize, blobs, degree + step);
		if (cost < minCost) {
			minCost = cost;
			degree += step;
			continue;
		}
		step = step / 2;
		stepDegree = degree;
		continue;
	}
	if (degree != 0) {
		slant(imgSize.height, blobs, degree);
	}
	return degree;
}
/*end implement new deslant function*/

cv::Mat cropDigitString(cv::Mat& src) {
	vector<int> horizontal;
	projectHorizontal(src, horizontal);
	auto it = std::max_element(horizontal.begin(), horizontal.end());
	int ymin = it - horizontal.begin();
	for (; ymin > 0 && horizontal[ymin] > 0; --ymin) {}
	int ymax = it - horizontal.begin();
	for (; ymax < horizontal.size() - 1 && horizontal[ymax] > 0; ++ymax) {}
	cv::Mat dst = src(cv::Rect(0, ymin, src.cols, ymax - ymin)).clone();
	return dst;
}

cv::Mat makeDigitMat(const cv::Mat& crop) {
	int width = 0;
	int height = 0;
	int paddingX = 0;
	int paddingY = 0;
	if (crop.rows > crop.cols) {
		//scale to height
		height = 20;
		width = (height * crop.cols) / crop.rows;
	}
	else {
		width = 20;
		//scale to width
		height = (width * crop.rows) / crop.cols;
	}
	cv::Size size(width, height);
	cv::Mat resize;
	cv::resize(crop, resize, size);
	cv::Mat padded(28, 28, CV_8UC1);
	padded.setTo(cv::Scalar::all(0));
	paddingX = (28 - resize.cols) / 2;
	paddingY = (28 - resize.rows) / 2;
	resize.copyTo(padded(cv::Rect(paddingX, paddingY, resize.cols, resize.rows)));
	//		cv::copyMakeBorder(resize, pad, 4, 4, 4, 4, cv::BORDER_CONSTANT, cv::Scalar(0));
	return padded;
}

cv::Mat makeDigitMat(Blob& blob, float slantAngle) {
	cv::Mat crop = cropBlob(blob);
	float angle = deslant(crop, NULL, objectWidth);
	if (slantAngle * angle > 0) {
		angle = abs(angle + slantAngle) > 48 ? 0 : angle;
	}
	if (angle != 0) {
		crop = slant(crop, angle);
	}
	cropMat(crop, crop);
	return makeDigitMat(crop);
}
