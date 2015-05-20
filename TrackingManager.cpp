#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "TrackingManager.h"

using namespace cv;
using namespace std;

/*
FeatureDetector::FeatureDetector()
{
}

FeatureDetector::~FeatureDetector()
{
}

void startDescriptorList(){

	//vector<string> detectorList = { "SIFT" };

	initModule_nonfree();
	// Create smart pointer for SIFT feature detector.
	vector<Ptr<FeatureDetector>> featureDetector;
	//Similarly, we create a smart pointer to the extractor.
	vector<Ptr<DescriptorExtractor>> featureExtractor;

	for (int i = 0; i < detectorList.size(); i++){
		namedWindow(detectorList[i]);

		featureDetector.push_back(FeatureDetector::create(detectorList[i]));
		featureExtractor.push_back(DescriptorExtractor::create(detectorList[i]));
	}
}

void detectToEachDescriptor(){
	for (int i = 0; i < detectorList.size(); i++){
		pointsDetection(detectorList[i], frame, featureDetector[i], featureExtractor[i]);
	}
}
*/