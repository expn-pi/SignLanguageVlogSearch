#include "stdafx.h"

#include "opencv2\opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "Flags.h"
#include "TrackingManager.h"

#include <stdio.h>
#include <conio.h>

using namespace cv;
using namespace std;

void drawKeyPoints(int index, Mat frame, TrackingManager *track){
	bool details = true;
	Mat frameClone = frame.clone();

	if (details) cout << "Drawing keypoints: " << index << "\n";
	track->drawKeyPointsImage(index, &frameClone);

	imshow(Flags::getDetectorName(), frameClone);
}

void drawTrackers(int index, Mat frame, TrackingManager *track){
	bool details = true;
	Mat frameClone = frame.clone();

	if (details) cout << "Drawing tracking: " << index << "\n";

	if (index > 0){
		track->drawTrack(index, &frameClone);
		imshow(Flags::getTrackerName(), frameClone);
	}
}

void drawMatches(int index, Mat frame, TrackingManager *track){
	bool details = true;
	Mat frameClone = frame.clone();

	if (details) cout << "Drawing Matching: " << index << "\n";

	if (index > 0){
		if (Flags::isRecuperateTrackerPoints())
			track->drawMatchs(0, index, &frameClone);
		else
			track->drawMatchs(index, &frameClone);

		imshow(Flags::getMatcherName(), frameClone);
	}
}

void drawImages(int index, Mat frame, TrackingManager *track){
	if (Flags::isShowkeypoints())drawKeyPoints(index, frame, track);
	if (Flags::isShowTracking()) drawTrackers(index, frame, track);
	if (Flags::isShowMatches()) drawMatches(index, frame, track);
}

void getStartData(int index, TrackingManager *track, Mat frame, Mat *lastFrame){
	bool debug = true;
	bool details = true;

	if (!Flags::isLoadSaved()){

		if (debug) cout << "\nGeting data of frame: " << index << "\n";

		if (debug) cout << "\tDetecting keypoints\n";
		track->detecKeyPoints(index, frame);

		track->passKeyPointsToTracker(index);

		if (debug) cout << "\tExtraction the features\n";
		track->extractFeatures(index, frame);
	}
}

void getBasisData(int index, TrackingManager *track, Mat frame, Mat *lastFrame){
	bool debug = true;
	bool details = true;

	if (debug) cout << "\nGeting data of frame: " << index << "\n";

	if (Flags::isNewKeypoints()){
		if (debug) cout << "\tDetecting keypoints\n";
		track->detecKeyPoints(index, frame);
	}

	if (Flags::isUseDetected()){
		track->passKeyPointsToTracker(index);
	}

	if (Flags::isNewDescriptors()){
		if (debug) cout << "\tExtraction the features\n";
		track->extractFeatures(index, frame);
	}
}

void getTrackingData(int index, TrackingManager *track, Mat frame, Mat *lastFrame){
	bool debug = true;
	bool details = true;

	if (Flags::isNewTracking()){
		if (debug) cout << "\tTracking\n";
		track->trackElement(index, *lastFrame, frame);

		track->getLostTrackerPoint(index);

		if (Flags::isRecuperateTrackerPoints()){
			if (track->verifyLostTrackRate(Flags::getTrackerLostMax())){
				track->recuperateTrackerPoints(index, frame);
			}
		}
	}

	if (!Flags::isRecuperateTrackerPoints()){
		if (Flags::isNewMatches()){
			if (debug) cout << "\tGetting the matches\n";
			track->matchingFeatures(index);
		}
	}
}

void getFullData(int index, TrackingManager *track, Mat frame, Mat *lastFrame){
	getBasisData(index, track, frame, lastFrame);
	getTrackingData(index, track, frame, lastFrame);
}

template<typename DataTypes>
void getData(int index, TrackingManager *track, Mat frame, Mat *lastFrame, DataTypes dFunction){
	if (Flags::isDebug()) cout << "\nRead the frame: " << index << "\n";

	//Croup the image to remove not used information on bottom
	Rect croup = Rect(0, 0, frame.cols, frame.rows - 65);

	Mat frameCrouped = frame(croup).clone();
	Mat gray;
	cvtColor(frameCrouped, gray, CV_BGR2GRAY);

	if (Flags::isGetNewData()) dFunction(index, track, gray, lastFrame);

	if (Flags::isShowImage()) drawImages(index, frameCrouped, track);

	(*lastFrame) = gray.clone();
}

void start(VideoCapture *capture, TrackingManager *track){
	//Test if movie was load
	if (Flags::isDebug()) cout << "Capturing" << "\n";
	*capture = VideoCapture(Flags::getFileMovieName());
	if (!capture->isOpened())
		throw "Error when reading steam_avi";

	if (Flags::isDebug()) cout << "Initializing the tracker options" << "\n";
	*track = TrackingManager::TrackingManager(Flags::getKeypointsNumber(), 0.001, 1, 3, false);
	//TrackingManager track = TrackingManager::TrackingManager({ "SIFT", "SURF", "GFTT", "HARRIS" });// , "Dense"});
	//TrackingManager track = TrackingManager::TrackingManager({ "HARRIS" });

	//Load the saved data
	if (Flags::isLoadSaved()){
		cout << "Start loading new Data\n";
		track->loadFramesData(Flags::getFileLocation());
	}
}

int _tmain(int argc, _TCHAR* argv[]){

	initModule_nonfree();

	VideoCapture capture;
	TrackingManager track;
	
	start(&capture, &track);

	cv::Mat lastFrame;
	if (Flags::isDebug()) cout << "Puting the actual frame on the variable" << "\n";
	cv::Mat frame;
	capture >> frame;

	int index = 0;
	
	//Loading the first frame
	getData(index, &track, frame, &lastFrame, getStartData); 

	capture >> frame;
	
	index++;

	waitKey(200);

	while (!frame.empty()){

		getData(index, &track, frame, &lastFrame, getFullData);
		
		capture >> frame;

		index++;

		waitKey(200);
	}

	if (Flags::isArmazenateFrameData()){
		cout << "Armazenating data\n";
		track.saveFramesData(Flags::getFileLocation());
	}

	if (Flags::isDebug()) cout << "Finish - press a key";

	waitKey(0);

	return 0;
}