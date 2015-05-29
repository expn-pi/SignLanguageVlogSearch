#pragma once

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

//#include "Flags.h"
#include "FrameManager.h"
#include "TrackPointsManager.h"

using namespace cv;
using namespace std;

class TrackingManager
{
private:
	//bool changeTrackPoints = false;
	bool useDetectorPoints = true;

	Ptr<FeatureDetector> featuresDetector;// Create a generic smart pointer for detectors.
	Ptr<FeatureDetector> detectorRecuperate;// Create a generic smart pointer for detectors.
	
	Ptr<DescriptorExtractor> featuresExtractor;//Create a generic smart pointer to the extractor.

	Ptr<DescriptorMatcher> matcher;//Create a generic smart pointer to the matcher.

	FrameManager frameManager;
	FrameManager::gfttParameters parameters;

	TrackingPointsManager trackingPointsManager;
	vector<Scalar> colors;

public:
	TrackingManager(){};

	TrackingManager(int maxCorners, double qualityLevel, double minDistance, bool harris = false, double k = 0.04){

		parameters.maxCorners = maxCorners;
		parameters.qualityLevel = qualityLevel;
		parameters.minDistance = minDistance;
		parameters.harris = harris;
		parameters.k = k;

		if (Flags::isDebug()) cout << "\tCreating the track manager: " << Flags::getTrackerName() << "\n";


		if (Flags::isNewKeypoints() || !Flags::isLoadSaved()){
			if (Flags::isDebug()) cout << "\t\tInitializing the descriptor: " << Flags::getDetectorName() << "\n";
			//featuresDetector = FeatureDetector::create("SIFT");
			if (Flags::getDetectorName() == "Good Features to track")
				featuresDetector = new GoodFeaturesToTrackDetector(maxCorners, qualityLevel, minDistance, Flags::getKeyPointsSize(), harris, k);
			else
				featuresDetector = FeatureDetector::create(Flags::getDetectorName());

			if (Flags::isRecuperateTrackerPoints()){
				detectorRecuperate = new GoodFeaturesToTrackDetector(Flags::getKeyPointsRecNumber(), qualityLevel, minDistance,
					Flags::getKeyPointsSize(), harris, k);
			}
		}

		if (Flags::isNewDescriptors()){
			if (Flags::isDebug()) cout << "\t\tInitializing the extractor: " << Flags::getExtractorName() << "\n";
			featuresExtractor = DescriptorExtractor::create(Flags::getExtractorName());
		}

		featuresExtractor = DescriptorExtractor::create(Flags::getExtractorName());

		matcher = DescriptorMatcher::create(Flags::getMatcherName());
		//BFMatcher matcher(NORM_L2);

		if (Flags::isDebug()) cout << "Starting the frame manager\n";
		
		frameManager = FrameManager();

		nameWindows();
	}

	int getFramesCount(){
		return frameManager.getSize();
	}

	//Window manager
	void nameWindows(){
		
		if (Flags::isShowkeypoints()){
			namedWindow(Flags::getDetectorName());
		}

		if (Flags::isShowTracking()){
			namedWindow(Flags::getTrackerName() + "-Hit");
			namedWindow(Flags::getTrackerName() + "-Miss");
		}

		if (Flags::isShowMatches()){
			namedWindow(Flags::getMatcherName() + "-Hit");
			namedWindow(Flags::getMatcherName() + "-Miss");
		}
	}

	void detecFirstKeyPoints(Mat frame){
		detecKeyPoints(0, frame);
		vector<KeyPoint> * keyPoints = frameManager.getKeypoints(0);
		cout << "Founded: " << keyPoints->size() << ", of: " << Flags::getKeyPointsNumber() << " keyPoints\n";
		trackingPointsManager = TrackingPointsManager(keyPoints->size());
		initialyseColorVector();
	}

	//Detector of interest points
	void detecKeyPoints(int index, Mat frame){
		bool debug = true;
		bool details = false;

		if (debug) cout << "Detecting key points of index: " << index << "\n";

		vector<KeyPoint> * keyPoints = frameManager.getKeypoints(index);	
		//if (index == 0)
			featuresDetector->detect(frame, *keyPoints);
			if (details) cout << "amount of keyPoints: " << keyPoints->size() << "\n";

		//else
			//featuresDetector2->detect(frame, *keyPoints);

			if (details) waitKey();
	}

	void writeKeyPoints(int index){
		vector<KeyPoint> * keyPoints = frameManager.getKeypoints(index);

		for (int i = 0; i < keyPoints->size(); i++){
			KeyPoint keyPoint = (*keyPoints)[i];

			cout << i << ": , Size: " << keyPoint.size << ", Angle: " << keyPoint.angle << ", Octave: " << keyPoint.octave
				<< ", Response: " << keyPoint.response << ", ClassId: " << keyPoint.class_id << "\n";
		}
	}

	void passKeyPointsToTracker(int index){
		bool debug = true;
		vector<KeyPoint> * keyPoints = frameManager.getKeypoints(index);

		if (debug) cout << "\t\tcopying points to tracker\n";
		vector<Point2f> *output = frameManager.getTrackerOutput(index);
		KeyPoint::convert(*keyPoints, *output);
	}

		//Draw the detected keypoints
	void drawKeyPointsImage(int index, Mat *frame){
		vector<KeyPoint> *keyPoints = frameManager.getKeypoints(index);
		drawKeypoints(*frame, *keyPoints, *frame, CV_RGB(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	}


	//Tracker of intest points
	void trackElement(int index, Mat lastFrame, Mat frame){

		bool debug = true;
		bool details = false;
		
		vector<Point2f> *input = frameManager.getTrackerInput(index);
		if (details) cout << "\t\tcopying last points from tracker\n";
		vector<Point2f> *lastOutput = frameManager.getTrackerOutput(index - 1);
		if (details)cout << "Input Size 2: " << input->size() << "\n";
		if (details)cout << "Last Output Size 1: " << lastOutput->size() << "\n";
		//vector<Point2f> lastOutPutBackUp = *lastOutput;

		if (lastOutput->size() > 0){
			if (details) cout << "Last Output Size 2: " << lastOutput->size() << "\n";
			
			(*input) = (*lastOutput);

			//KeyPoint::convert(*keyPoints, *input);
			if (details) cout << "Input Size 2: " << input->size() << "\n";

			vector<Point2f> *output = frameManager.getTrackerOutput(index);
			vector<uchar> *status = frameManager.getTrackerStatus(index);
			vector<float> *error = frameManager.getTrackerError(index);

			if (details) waitKey();

			if (debug)cout << "\nStarting tracking\n";
			TermCriteria t = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01);
			calcOpticalFlowPyrLK(lastFrame, frame, *input, *output, *status, *error, cvSize(21, 21), 3, t, 0, Flags::getTrackerError());
			//calcOpticalFlowPyrLK(lastFrame, frame, *input, *output, *status, *error, cvSize(21, 21), 3, 
			//	TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 30, 0.01), OPTFLOW_USE_INITIAL_FLOW);

			//printPoints(*output, frame);
		}
		else{
			cout << "Nothing to track\n";
		}
	}

	void debugLostTrackerPoints(int frameIdx, int ptIdx, bool tracked, bool detected, float longDistance, float distance){
		cout << "\nFrame: " << frameIdx << ", ";
		cout << "Point: " << ptIdx << "\n";
		cout << "tracked: " << tracked << "\n";

		cout << "Detected: " << detected << "\n";
		cout << "Longe distance: " << longDistance << "\n";
		cout << "\tdistance: " << distance << "\n";
		cout << "\tMax distance: " << Flags::getMaxEuclidianDistance() << "\n";

		cout << "( (longDistance || !tracked) && detected): " << ((longDistance || !tracked) && detected) << "\n";
	}

	void getLostTrackerPoint(int index){

		bool debug = true;
		bool details = false;

		if (debug) cout << "Getting the lost tracker points\n";

		vector<uchar> *status = frameManager.getTrackerStatus(index);
		vector<Point2f> *input = frameManager.getTrackerInput(index);
		vector<Point2f> *output = frameManager.getTrackerOutput(index);

		if (debug) cout << "lost the keypoints: ";
		for (int i = 0; i < status->size(); i++){

			bool tracked = ((int)(*status)[i]) == 1;
			bool detected = trackingPointsManager.isDetected(i);
			//pointsChangeHistoric.status[i].detected;

			Point2f p1 = (*input)[i];
			Point2f p2 = (*output)[i];

			float distance = sqrt(  pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2) );
			bool longDistance = distance > Flags::getMaxEuclidianDistance();

			if ((longDistance || !tracked) && detected){

				if (details) debugLostTrackerPoints(index, i, tracked, detected, longDistance, distance);

				trackingPointsManager.setLostPoint(i, index, p1);

				if (debug) cout << i << ", ";
			}
		}

		//waitKey();

		if (debug) cout << "\nlost " << trackingPointsManager.getLostCount() << " points\n";
	}

	bool verifyLostTrackRate(float maximum){
		return trackingPointsManager.verifyLostTrackRate(maximum);
	}

	void armazenateImage(int index, Mat *frame){
		bool details = false;
		Mat * image = frameManager.getFrameImage(index);

		if(details) cout << "Size of the parameter image is: " << frame->size() << "\n";

		*image = frame->clone();

		if (details)  cout << "Size of the alocate image is: " << image->size() << "\n";
	}

	Mat *getImage(int index){
		return frameManager.getFrameImage(index);
	}

	void recuperateTrackerPoints(int index, Mat frame){
		bool debug = true;
		bool details = true;

		if (!Flags::isLoadSaved()){
			if (details) cout << "\t\tDetecting the actual frame keypoints\n";
			detecKeyPoints(index, frame);
			if (details) cout << "\t\tExtracting the actual keypoints features\n";
			extractFeatures(index, frame);
		}

		trackingPointsManager.recuperateTrackerPoints(index, frameManager);

		//waitKey();

		//matchingFeatures(Flags::getBestFrameToMatch(), index);
		
		/*vector<DMatch> *actualMatching = frameManager.getMatches(index);
		vector<Point2f> *output = frameManager.getTrackerOutput(index);
		vector<KeyPoint> *keyPoints = frameManager.getKeypoints(index);

		if (details) cout << "Matching size: " << actualMatching->size();

		if (debug) cout << "\tGetting Convertion\n";
		vector<Point2f> keyPointsConverted;
		KeyPoint::convert(*keyPoints, keyPointsConverted);

		for (int i = 0; i < actualMatching->size(); i++){
			DMatch auxMatching = (*actualMatching)[i];

			if (lostPoints[i] && (auxMatching.distance < Flags::getMatcherError())){
				lostPoints[i] = false;

				if (debug) cout << "\t\tCorrespondence: " << auxMatching.queryIdx << ", is: " << auxMatching.trainIdx << "\n";

				correspondence c;
				c.p1 = keyPointsConverted[auxMatching.queryIdx];
				c.p2 = keyPointsConverted[auxMatching.trainIdx];
				correspondences.push_back(c);

				(*output)[i].x = keyPointsConverted[auxMatching.trainIdx].x;
				(*output)[i].y = keyPointsConverted[auxMatching.trainIdx].y;
			}
		}*/
		
	}

	void drawLostMatches(){
		trackingPointsManager.drawLostMatches(frameManager);
	}

	//void drawLostMatches(int ptIdx, int frIdx){
	//	vector <DMatch> * matches = trackingPointsManager.getMatchesForFramePoint(ptIdx, frIdx);
	//	if (matches != NULL){

	//	}
	//}

	void printPoints(vector<Point2f> points, Mat image){
		
		int width = image.size().width;
		int height = image.size().height;

		cout << image.size() << "\n";
		waitKey();

		for (int i = 0; i < points.size(); i++){
			
			float x = points[i].x;
			float y = points[i].y;

			if (x<0){
				cout << i << ": " << points[i] << " x<0" <<  "\n";
			}
			if (y<0){
				cout << i << ": " << points[i] << " y<0" << "\n";
			}
			if (y>height){
				cout << i << ": " << points[i] << " y>height" << "\n";
			}
			if(x>width){
				cout << i << ": " << points[i] << " x>width" << "\n";
			}
		}
		waitKey();
	}

		//Drawer of tracking correspondent points
	void drawTrack(int index, Mat *frameHit, Mat *frameMiss){
		bool details = false;

		vector<Point2f> *input = frameManager.getTrackerInput(index);;
		vector<Point2f> *output = frameManager.getTrackerOutput(index);
		vector<uchar> *status = frameManager.getTrackerStatus(index);

		for (int i = 0; i < input->size(); i++){

			bool statusAux = trackingPointsManager.isDetected(i);
			//pointsChangeHistoric.status[i].detected;//(((int)(*status)[i]) == 1);
			
			if (details) cout << "Frame: " << index << ", Point: " << i << ", status(" << trackingPointsManager.isDetected(i) << ")\n";

			if (details) cout << "Converting the points\n";

			CvPoint p1 = cvPoint((*input)[i].x, (*input)[i].y);
			CvPoint p2 = cvPoint((*output)[i].x, (*output)[i].y);

			bool looseNow = false;

			if (details) cout << "Verify losse now points\n";
			if (trackingPointsManager.isLostFrame(i, index)){//pointsChangeHistoric.status[i].lostFrame == index){
				looseNow = true;
				if (details) cout << "Losse now the point: " << i << "\n";
				//waitkey();
			}
			else if (!trackingPointsManager.isDetected(i)){
				CvPoint pAux = trackingPointsManager.getLastPointPosition(i);
				p1 = pAux;
				p2 = pAux;
			}
			
			//if (!statusAux) waitKey();
			//drawCorrespondentPoints(frameHit, frameMiss, p1, p2, color, condition, recuperate, false, losseNow)
			if (details) cout << "Draw correspondences\n";
			drawCorrespondentPoints(frameHit, frameMiss, p1, p2, colors[i], statusAux, false, looseNow);
			if (details) cout << "Finish draw\n";
			//drawCorrespondentPoints(frame, p1, p2, !lostPoints[i]);
			//imshow(getTrackerName(), *frame);
			//waitKey(20);
		}

		//for (int i = 0; i < correspondences.size(); i++){
		//	correspondence c = correspondences[i];
		//	drawCorrespondentPoints(frameHit, frameMiss, c.p1, c.p2, true, true);
		//}
	}
	
	void extractTrackFeatures(int index, Mat frame){

		bool debug = true;

		if (debug) cout << "\Extracting tracking features\n";

		vector<Point2f> * output = frameManager.getTrackerOutput(index);
		Mat *trackerDescriptors = frameManager.getTrackerDescriptors(index);

		vector<KeyPoint> trackerKeypoints;
		

		for (int i = 0; i < output->size(); i++){

			Point2f point = (*output)[i];

			//Size: 8, Angle : -1, Octave : 0, Response : 0, ClassId : -1
			KeyPoint keyPoint = KeyPoint(point, Flags::getKeyPointsSize());

			trackerKeypoints.push_back(keyPoint);
		}

		featuresExtractor->compute(frame, trackerKeypoints, *trackerDescriptors);
	}


	//Extract and Matching keypoints
	void extractFeatures(int index, Mat frame){
		bool debug = false;
		bool details = false;

		vector <KeyPoint> *keyPoints = frameManager.getKeypoints(index);
		Mat *descriptors = frameManager.getDescriptors(index);

		if (debug) cout << "\t Extracting features of index: " << index << "\n";
		// Compute the 128 dimension descriptor at each keypoint.
		// Each row in "descriptors" correspond to the descriptor for each keypoint

		featuresExtractor->compute(frame, *keyPoints, *descriptors);

		if (details) cout << "amount of keyPoints: " << keyPoints->size() << "\n";
		if (details) cout << "amount of Descriptors: " << descriptors->size() << "\n";
		if (details) cout << "Image size: " << frame.size() << "\n";
	}

	void matchingFeatures(int index){
		matchingFeatures(index - 1, index);
	}

	vector<DMatch> * matchingFeatures(int idxDescriptor1, int idxDescriptor2){
		bool debug = false;
		bool details = false;

		//BFMatcher matcher(NORM_L2);

		vector<DMatch> *matches = frameManager.getMatches(idxDescriptor2);

		Mat *lastDescriptors = frameManager.getDescriptors(idxDescriptor1);
		Mat *actualDescriptors = frameManager.getDescriptors(idxDescriptor2);

		//if (lastDescriptors->dims>0 && actualDescriptors->dims>0){
		matcher->match(*lastDescriptors, *actualDescriptors, *matches);

		return matches;
		//}
		//else{
			//cout << "Nothing to match\n";
			//waitKey();
		//}
	}
		
		//Drawer of matching points
	void drawMatchs(int index, Mat *frameHit, Mat *frameMiss){
		drawMatchs(index - 1, index, frameHit, frameMiss);
	}
		
	void drawMatchs(int idxPoint1, int idxPoint2, Mat *frameHit, Mat *frameMiss){
		bool details = false;
		vector<DMatch> *dMatchs = frameManager.getMatches(idxPoint2);

		vector<KeyPoint> *points1 = frameManager.getKeypoints(idxPoint1);
		vector<KeyPoint> *points2 = frameManager.getKeypoints(idxPoint2);

		if(details) cout << "Enter in draw matchs function\n";
		//waitKey();

		//if (points1->size() == points2->size()){
			if (details) cout << "Pass throw basic condition\n";
			for (int i = 0; i < dMatchs->size(); i++){
				
				if (details) cout << i << ": Interating\n";
				
				int idxPoint1 = (*dMatchs)[i].queryIdx;
				int idxPoint2 = (*dMatchs)[i].trainIdx;
				float distance = (*dMatchs)[i].distance;

				KeyPoint keyPoint1 = (*points1)[idxPoint1];
				KeyPoint keyPoint2 = (*points2)[idxPoint2];

				drawCorrespondentPoints(frameHit, frameMiss, keyPoint1.pt, keyPoint2.pt, colors[i], distance < Flags::getMatcherError());
				//drawLineToKeyPoints(frame, keyPoint1, keyPoint2, distance);
			}
		//}
	}


	//Persistence
	void saveFramesData(string fileLocation){
		frameManager.saveFramesData(fileLocation, Flags::getDetectorName(), parameters);
	}

	void loadFramesData2(string fileLocation){
		frameManager.loadFramesData2(fileLocation);
	}

	void loadFramesData(string fileLocation){
		frameManager.loadFramesData(fileLocation);
	}

private:

	//Tracking auxiliar functions

	void initialyseColorVector(){
		RNG rng(12345);
		for (int i = 0; i < trackingPointsManager.getKeyPointsFounded(); i++){

			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			colors.push_back(color);
		}
	}

	//Draw correspondent points, with oberservin a condition of correctness
	void drawCorrespondentPoints(Mat *frameHit, Mat *frameMiss, CvPoint p1, CvPoint p2, Scalar color, bool condition, 
		bool recuperate = false, bool losseNow = false){
		bool details = false;
		int radius = 4;

		if (losseNow){
			radius = 16;
			drawCorrespondences(frameHit, p1, p2, radius, color);
		}

		Mat *image;
		
		if (recuperate){
			if (details) cout << "Recuperate Point\n";
			image = frameHit;
		}
		else if (condition){
			if (details) cout << "Good Point\n";
			image = frameHit;
		}
		else{
			if (details) cout << "BadPoint\n";
			image = frameMiss;
		}

		drawCorrespondences(image, p1, p2, radius, color);
	}

	void drawCorrespondences(Mat *image, Point2f p1, Point2f p2, int radius, Scalar color){
		bool details = true;

		line(*image, p1, p2, color, 1);
		circle(*image, p1, radius, color, 1, CV_AA, 0);
		circle(*image, p2, radius, color, 1, CV_AA, 0);
	}

	/*
	Scalar cPoint1, cPoint2, cLine;

	cPoint1 = color;
	cPoint2 = color;
	cLine = color;

	if (recuperate){
	if (details) cout << "Recuperate Point\n";
	cPoint1 = CV_RGB(255, 0, 255);
	cPoint2 = CV_RGB(100, 0, 100);
	cLine = CV_RGB(200, 0, 200);

	image = frameHit;
	}
	else{
	if (condition){
	if (details) cout << "Good Point\n";
	cPoint1 = CV_RGB(255, 255, 0);
	cPoint2 = CV_RGB(200, 0, 0);
	cLine = CV_RGB(255, 100, 0);
	image = frameHit;
	}
	else{
	if (details) cout << "BadPoint\n";
	cPoint1 = CV_RGB(0, 255, 255);
	cPoint2 = CV_RGB(0, 0, 200);
	cLine = CV_RGB(0, 100, 255);

	image = frameMiss;
	}
	}

	line(*image, p1, p2, cLine, 1);
	circle(*image, p1, radius, cPoint1, 1, CV_AA, 0);
	circle(*image, p2, radius, cPoint2, 1, CV_AA, 0);
	*/
};