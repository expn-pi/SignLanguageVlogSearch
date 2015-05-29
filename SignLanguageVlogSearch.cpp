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


float getMatchesSum(vector<DMatch> *matches){

	float sum = 0;

	for (int i = 0; i < matches->size(); i++){
		sum += (*matches)[i].distance;
	}

	return sum;
}

int getGoodMatchesCount(vector<DMatch> *matches){

	int count = 0;

	for (int i = 0; i < matches->size(); i++){
		if ((*matches)[i].distance < Flags::getMatcherError()){
			count++;
		}
	}

	return count;
}

void getBestMatch(VideoCapture *capture, TrackingManager *track){

	//track->matchAll(Flags::getFileLocation());
	int size = track->getFramesCount();

	//vector<float> frameMatchingSum;
	vector<float> frameGoodMatchingCount;

	cout << "put 0 in all\n";
	for (int i = 0; i < size; i++){
		//frameMatchingSum.push_back(0.0f);
		frameGoodMatchingCount.push_back(0);
	}

	cout << "Sum all\n";
	for (int i = 0; i < size; i++){
		cout << "Compare ( " << i << ")\n";
		for (int j = 0; j < size; j++){
			if (i != j){
				//cout << "Compare ( " << i << ", " << j << " )\n";
				vector<DMatch> *matches = track->matchingFeatures(i, j);

				frameGoodMatchingCount[i] += getGoodMatchesCount(matches);
				//float sum = getMatchesSum(matches);
				//frameMatchingSum[i] += sum;
			}
		}
	}

	int bestFrame = 0;

	cout << "Results\n";
	for (int i = 0; i < size; i++){
		//cout << i<<": matches sum = "<< frameMatchingSum[i]<<"\n";
		cout << i << ": Good matches count = " << frameGoodMatchingCount[i] << "\n";
		if (frameGoodMatchingCount[i] > frameGoodMatchingCount[bestFrame]){
			bestFrame = i;
		}
	}

	cout << "The best frame is: " << bestFrame << "\n";
	waitKey();
}


void drawKeyPoints(int index, Mat frame, TrackingManager *track){
	bool details = false;
	Mat frameClone = frame.clone();

	if (details) cout << "\tDrawing keypoints: " << index << "\n";
	track->drawKeyPointsImage(index, &frameClone);

	imshow(Flags::getDetectorName(), frameClone);
}

void drawTrackers(int index, Mat frame, TrackingManager *track){
	bool details = false;
	Mat frameHit = frame.clone();
	Mat frameMiss = frame.clone();

	if (details) cout << "\tDrawing tracking: " << index << "\n";

	if (index > 0){
		track->drawTrack(index, &frameHit, &frameMiss);

		if (details) cout << "\t\tShow hit images: " << index << "\n";
		imshow(Flags::getTrackerName()+"-Hit", frameHit);

		if (details) cout << "\t\tShow loose images: " << index << "\n";
		imshow(Flags::getTrackerName() + "-Miss", frameMiss);

		
	}
}

void drawMatches(int index, Mat frame, TrackingManager *track){
	bool details = false;
	Mat frameHit = frame.clone();
	Mat frameMiss = frame.clone();

	if (details) cout << "\tDrawing Matching: " << index << "\n";

	if (index > 0){
		if (Flags::isRecuperateTrackerPoints())
			track->drawMatchs(Flags::getBestFrameToMatch(), index, &frameHit, &frameMiss);
		else
			track->drawMatchs(index, &frameHit, &frameMiss);

		imshow(Flags::getMatcherName()+"-Hit", frameHit);
		imshow(Flags::getMatcherName() + "-Miss", frameMiss);
	}
}

void drawImages(int index, Mat frame, TrackingManager *track){
	if (Flags::isShowkeypoints())drawKeyPoints(index, frame, track);
	if (Flags::isShowTracking()) drawTrackers(index, frame, track);
	if (Flags::isShowMatches()) drawMatches(index, frame, track);

	waitKey(50);
}


void getStartData(int index, TrackingManager *track, Mat frame, Mat *lastFrame){
	bool debug = true;
	bool details = true;

	if (!Flags::isLoadSaved()){

		if (debug) cout << "\n\tGeting data of frame: " << index << "\n";

		if (debug) cout << "\t\tDetecting the first keypoints\n";
		track->detecFirstKeyPoints(frame);
		//detecFirstKeyPoints(Mat frame)
		track->passKeyPointsToTracker(index);

		track->extractTrackFeatures(index, frame);

		if (debug) cout << "\t\tExtraction of the first features\n";
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
		//track->writeKeyPoints(index);
		//waitKey();
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

		track->extractTrackFeatures(index, frame);

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

template<typename FunctionForProcess>
void getData(int index, TrackingManager *track, Mat frame, Mat *lastFrame, FunctionForProcess functionForProcess){
	if (Flags::isDebug()) cout << "\nRead the frame: " << index << "\n";

	//Croup the image to remove not used information on bottom
	Rect croup = Rect(0, 0, frame.cols, frame.rows - 65);

	Mat frameCrouped = frame(croup).clone();
	Mat gray;
	cvtColor(frameCrouped, gray, CV_BGR2GRAY);

	track->armazenateImage(index, &frame);

	if (Flags::isGetNewData()) functionForProcess(index, track, gray, lastFrame);

	if (Flags::isShowImage()) drawImages(index, frameCrouped, track);

	track->drawLostMatches();

	(*lastFrame) = gray.clone();
}

template<typename FunctionForProcess>
void processFrame(int *index, VideoCapture *capture, TrackingManager *track, Mat *frame, Mat *lastFrame, FunctionForProcess functionForProcess){
	
	getData(*index, track, *frame, lastFrame, functionForProcess);

	(*capture) >> (*frame);

	(*index) = (*index)+1;
}

void interateWithVideo(VideoCapture *capture, TrackingManager *track){

	cv::Mat lastFrame;
	cv::Mat frame;
	int index = 0;

	if (Flags::isDebug()) cout << "Puting the first frame on the variable" << "\n";
	(*capture) >> frame;

	processFrame(&index, capture, track, &frame, &lastFrame, getStartData);

	while (!frame.empty()){
		processFrame(&index, capture, track, &frame, &lastFrame, getFullData);
	}
}

void start(VideoCapture *capture, TrackingManager *track){
	//Test if movie was load
	if (Flags::isDebug()) cout << "Capturing" << "\n";
	*capture = VideoCapture(Flags::getFileMovieName());
	if (!capture->isOpened())
		throw "Error when reading steam_avi";

	if (Flags::isDebug()) cout << "Initializing the tracker options" << "\n";
	*track = TrackingManager::TrackingManager(Flags::getKeyPointsNumber(), 0.01, 4, true);
	//TrackingManager track = TrackingManager::TrackingManager({ "SIFT", "SURF", "GFTT", "HARRIS" });// , "Dense"});
	//TrackingManager track = TrackingManager::TrackingManager({ "HARRIS" });
}

int _tmain(int argc, _TCHAR* argv[]){

	initModule_nonfree();

	VideoCapture capture;
	TrackingManager track;
	
	start(&capture, &track);

	//Load the saved data
	if (Flags::isLoadSaved()){
		cout << "Start loading new Data\n";
		track.loadFramesData(Flags::getFileLocation());

		if (Flags::isGetBestMatch()){
			cout << "Getting the best frame to match with others\n";
			getBestMatch(&capture, &track);
		}
	}

	interateWithVideo(&capture, &track);

	if (Flags::isArmazenateFrameData()){
		cout << "Armazenating data\n";
		track.saveFramesData(Flags::getFileLocation());
	}

	if (Flags::isDebug()) cout << "Finish - press a key";

	waitKey(0);

	return 0;
}