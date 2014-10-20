/*
 * blob.h
 *
 *  Created on: Aug 21, 2014
 *      Author: thienlong
 */

#ifndef BLOB_H_
#define BLOB_H_
#include "opencv2/core/operations.hpp"

//blob
class Blob {
private:
	cv::Rect bound;
	bool needNewRect;
public:
	Blob();
	std::vector<cv::Point2i> points;
	cv::Rect boundingRect();
	void add(const Blob& other);
	void add(const cv::Point2i& point);
	void setModify(bool flag);
};

class Blobs {
private: std::vector <Blob* > blobs;
public:
	Blobs();
	size_t size() const;
	void erase(int index);
	Blob* operator[] (int index) const;
	void clear();
	void sort(bool (*compFunct)(Blob* blob1, Blob* blob2));
	void add(Blob* blob);
	void clone(Blobs& other) const;
	template<class _EqPredicate> void partition(std::vector<int>& label, _EqPredicate predicate=_EqPredicate()) {
		cv::partition(blobs, label, predicate);
	}
	~Blobs();
};
/*binary image [0, 1]*/
void findBlobs(const cv::Mat &binary, Blobs &blobs);
cv::Rect boundingRect(const Blobs &blobs);
cv::Mat drawBlob(const Blobs& blobs);
//@deprecase. Use defragment instead of
void groupVertical(Blobs& blobs, std::vector<int> &labels);
bool sortByVertical(Blob* blob1, Blob* blob2);
void sortBlobsByVertical(Blobs &blobs);
void defragment(cv::Mat& strImg, Blobs &blobs);
double distanceBlobs(Blob& blob1, Blob& blob2);
double distanceBlobs(std::vector<cv::Point2i >& blob1, std::vector<cv::Point2i >& blob2);

#endif /* BLOB_H_ */
