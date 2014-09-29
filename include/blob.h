/*
 * blob.h
 *
 *  Created on: Aug 21, 2014
 *      Author: thienlong
 */

#ifndef BLOB_H_
#define BLOB_H_

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

typedef std::vector <Blob* > Blobs;
void findBlobs(const cv::Mat &binary, Blobs &blobs);
Blobs findBlobs(const cv::Mat &binary);
void clearBlobs(Blobs &blobs);
cv::Rect boundingRect(Blobs &blobs);
cv::Mat drawBlob(const Blobs& blobs);
//@deprecase. Use defragment instead of
void groupVertical(Blobs& blobs, std::vector<int> &labels);
void sortBlobsByVertical(Blobs &blobs);
void defragment(cv::Mat& strImg, Blobs &blobs);
double distanceBlobs(Blob& blob1, Blob& blob2);
double distanceBlobs(std::vector<cv::Point2i >& blob1, std::vector<cv::Point2i >& blob2);

#endif /* BLOB_H_ */
