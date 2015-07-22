/*
 * blob.h
 *
 *  Created on: Aug 21, 2014
 *      Author: thienlong
 */

#ifndef BLOB_H_
#define BLOB_H_
#include "opencv2/core/core.hpp"
#include <vector>
#include <functional>
//blob
class Blob {
private:
	cv::Rect bound;
	bool needNewRect;
	float innerGap;
public:
	Blob();
	std::vector<cv::Point2i> points;
	cv::Rect boundingRect();
	void add(const Blob& other);
	void add(const cv::Point2i& point);
	void setModify(bool flag);
	void move(int x, int y);
	float getInnerGap() const;
	cv::Point2f getMassCenter();
};

class Blobs {
private: std::vector <Blob* > blobs;
public:
	Blobs();
	Blobs(Blobs& blobs);
	size_t size() const;
	void erase(int index);
	Blob* detach(int index);
	Blob* operator[] (int index) const;
	void clear();
	void sort(bool (*compFunct)(Blob* blob1, Blob* blob2));
	void add(Blob* blob);
	void clone(Blobs& other) const;
	template<class _EqPredicate> void partition(std::vector<int>& label, _EqPredicate predicate=_EqPredicate()) {
		cv::partition(blobs, label, predicate);
	}
	int findBiggestBlob(std::function<bool(int)> accept = [](int i) -> bool {return true;});
	cv::Rect boundingRect() const;
	cv::Rect boundingRect(int from, int end) const;
	cv::Point2f getMassCenter() const;
	void move(int x, int y);
	Blob* newBlob(int from, int end) const;
	cv::Mat cropBlobs(int from, int end);
	~Blobs();
};
/*binary image [0, 1]*/
void findBlobs(const cv::Mat &binary, Blobs &blobs);
cv::Rect boundingRect(Blob& b1, Blob& b2);
cv::Rect boundingRect(cv::Rect r1, cv::Rect r2);
void drawBlobs(const Blobs& blobs, cv::Mat& output);
cv::Mat drawBlobs(const Blobs& blobs);
cv::Mat drawBinaryBlobs(const Blobs& blobs);
void drawBinaryBlobs(const Blobs& blobs, cv::Mat& output);
void estHeightVertCenter(Blobs& blobs, float& strHeight, float& middleLine);
//@deprecase. Use defragment instead of
void groupVertical(Blobs& blobs);
bool sortByVertical(Blob* blob1, Blob* blob2);
void sortBlobsByVertical(Blobs &blobs);
void defragment(cv::Mat& strImg, Blobs &blobs);

double distanceBlobs(const Blob& blob1, const Blob& blob2);
double distanceBlobs(const std::vector<cv::Point2i >& blob1, const std::vector<cv::Point2i >& blob2);
#endif /* BLOB_H_ */
