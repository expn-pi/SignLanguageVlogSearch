#pragma once

#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

//#include "Flags.h"
#include "FrameManager.h"

using namespace cv;
using namespace std;

class TrackingManager
{
private:
	//bool changeTrackPoints = false;
	bool useDetectorPoints = true;

	struct correspondence{
		Point2f p1;
		Point2f p2;
	};
	vector<correspondence> correspondences;

	Ptr<FeatureDetector> featuresDetector;// Create a generic smart pointer for detectors.
	Ptr<FeatureDetector> featuresDetector2;// Create a generic smart pointer for detectors.
	
	Ptr<DescriptorExtractor> featuresExtractor;//Create a generic smart pointer to the extractor.

	FrameManager frameManager;
	FrameManager::gfttParameters parameters;

	int countLostPoints = 0;
	vector<bool> lostPoints;

public:
	TrackingManager(){};

	TrackingManager(int maxCorners, double qualityLevel, double minDistance, int blockSize = 3, bool harris = false, double k = 0.04){

		parameters.maxCorners = maxCorners;
		parameters.qualityLevel = qualityLevel;
		parameters.minDistance = minDistance;
		parameters.blockSize = blockSize;
		parameters.harris = harris;
		parameters.k = k;

		if (Flags::isDebug()) cout << "\tCreating the track manager: " << Flags::getTrackerName() << "\n";


		if (Flags::isNewKeypoints() || !Flags::isLoadSaved()){
			if (Flags::isDebug()) cout << "\t\tInitializing the descriptor: " << Flags::getDetectorName() << "\n";
			//featuresDetector = FeatureDetector::create("SIFT");
			if (Flags::getDetectorName() == "Good Features to track")
				featuresDetector = new GoodFeaturesToTrackDetector(maxCorners, qualityLevel, minDistance, blockSize, harris, k);
			else
				featuresDetector = FeatureDetector::create(Flags::getDetectorName());

			featuresDetector2 = new GoodFeaturesToTrackDetector(8.0f*maxCorners, qualityLevel, minDistance, blockSize, harris, k);
		}

		if (Flags::isNewDescriptors()){
			if (Flags::isDebug()) cout << "\t\tInitializing the extractor: " << Flags::getExtractorName() << "\n";
			featuresExtractor = DescriptorExtractor::create(Flags::getExtractorName());
		}

		featuresExtractor = DescriptorExtractor::create(Flags::getExtractorName());

		if (Flags::isDebug()) cout << "Starting the frame manager\n";
		
		frameManager = FrameManager();
		
		initialyseLostPointsVector();
		
		nameWindows();
	}

	//Window manager
	void nameWindows(){
		
		if (Flags::isShowkeypoints()){
			namedWindow(Flags::getDetectorName());
		}

		if (Flags::isShowTracking()){
			namedWindow(Flags::getTrackerName());
		}

		if (Flags::isShowMatches()){
			namedWindow(Flags::getMatcherName());
		}
	}

	//Detector of interest points
	void detecKeyPoints(int index, Mat frame){
		bool debug = true;
		bool details = true;

		if (debug) cout << "Detecting key points of index: " << index << "\n";

		vector<KeyPoint> * keyPoints = frameManager.getKeypoints(index);	
		//if (index == 0)
			featuresDetector->detect(frame, *keyPoints);
			if (details) cout << "amount of keyPoints: " << keyPoints->size() << "\n";
		//else
			//featuresDetector2->detect(frame, *keyPoints);
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
		bool details = true;
		
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
			if (details)cout << "Input Size 2: " << input->size() << "\n";

			vector<Point2f> *output = frameManager.getTrackerOutput(index);
			vector<uchar> *status = frameManager.getTrackerStatus(index);
			vector<float> *error = frameManager.getTrackerError(index);

			waitKey();

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

	void getLostTrackerPoint(int index){

		bool debug = true;
		bool details = true;

		vector<uchar> *status = frameManager.getTrackerStatus(index);

		for (int i = 0; i < status->size(); i++){

			bool actualLosted = ((int)(*status)[i]) == 0;
			bool lostPrevisiusly = lostPoints[i];

			if (actualLosted && !lostPrevisiusly){
				countLostPoints++;
				lostPoints[i] = true;
				if (details) cout << "lost the keypoint: " << i << "\n";
			}
		}

		if (debug) cout << "lost " << countLostPoints << " points\n";
	}

	bool verifyLostTrackRate(double maximum){
		bool debug = true;

		float lostPercentual = 1.0f - (((float)countLostPoints) / ((float)Flags::getKeypointsNumber()));

		if (debug) cout << "Lost percentual: " << lostPercentual*100.0f << "\n";

		if (lostPercentual < maximum){
			if (debug) "try recuperate points\n";

			return true;
		}
		else{
			return false;
		}
	}

	void recuperateTrackerPoints(int index, Mat frame){
		bool details = true;

		correspondences.clear();
		//useDetectorPoints = true;

		bool debug = true;
		if (debug) cout << "\tGetting DMatch\n";

		detecKeyPoints(index, frame);
		extractFeatures(index, frame);
		matchingFeatures(0, index);

		if (Flags::isRecuperateTrackerPoints()){
			vector<DMatch> *actualMatching = frameManager.getMatches(index);
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
			}
		}
	}

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
	void drawTrack(int index, Mat *frame){
		bool details = false;

		vector<Point2f> *input = frameManager.getTrackerInput(index);;
		vector<Point2f> *output = frameManager.getTrackerOutput(index);
		vector<uchar> *status = frameManager.getTrackerStatus(index);

		for (int i = 0; i < input->size(); i++){

			bool statusAux = (((int)(*status)[i]) == 1);
			
			if(details) cout << "Frame: "<< index << ", Point: " << i << ", status(" << ((int)(*status)[i]) << ")\n";

			CvPoint p1 = cvPoint((*input)[i].x, (*input)[i].y);
			CvPoint p2 = cvPoint((*output)[i].x, (*output)[i].y);

			//if (!statusAux) waitKey();
			drawCorrespondentPoints(frame, p1, p2, statusAux);
			//drawCorrespondentPoints(frame, p1, p2, !lostPoints[i]);
			//imshow(getTrackerName(), *frame);
			//waitKey(20);
		}

		for (int i = 0; i < correspondences.size(); i++){
			correspondence c = correspondences[i];
			drawCorrespondentPoints(frame, c.p1, c.p2, true, true);
		}
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

	void matchingFeatures(int idxDescriptor1, int idxDescriptor2){
		bool debug = false;
		bool details = false;

		BFMatcher matcher(NORM_L2);

		vector<DMatch> *matches = frameManager.getMatches(idxDescriptor2);

		Mat *lastDescriptors = frameManager.getDescriptors(idxDescriptor1);
		Mat *actualDescriptors = frameManager.getDescriptors(idxDescriptor2);

		//if (lastDescriptors->dims>0 && actualDescriptors->dims>0){
			matcher.match(*lastDescriptors, *actualDescriptors, *matches);
		//}
		//else{
			//cout << "Nothing to match\n";
			//waitKey();
		//}
	}
		
		//Drawer of matching points
	void drawMatchs(int index, Mat *frame){
		bool details = false;
		vector<DMatch> *dMatchs = frameManager.getMatches(index);

		vector<KeyPoint> *points1 = frameManager.getKeypoints(index - 1);
		vector<KeyPoint> *points2 = frameManager.getKeypoints(index);

		if(details) cout << "Enter in draw matchs function\n";
		//waitKey();

		//if (points1->size() == points2->size()){
			cout << "Pass throw basic condition\n";
			for (int i = 0; i < dMatchs->size(); i++){
				
				if (details) cout << i << ": Interating\n";
				
				int idxPoint1 = (*dMatchs)[i].queryIdx;
				int idxPoint2 = (*dMatchs)[i].trainIdx;
				float distance = (*dMatchs)[i].distance;

				KeyPoint keyPoint1 = (*points1)[idxPoint1];
				KeyPoint keyPoint2 = (*points2)[idxPoint2];

				drawCorrespondentPoints(frame, keyPoint1.pt, keyPoint2.pt, distance < Flags::getMatcherError());
				//drawLineToKeyPoints(frame, keyPoint1, keyPoint2, distance);
			}
		//}
	}


	//Persistence
	void loadFramesData2(string fileLocation){
		frameManager.loadFramesData2(fileLocation);
	}

	//Persistence
	void saveFramesData(string fileLocation){
		frameManager.saveFramesData(fileLocation, Flags::getDetectorName(), parameters);
	}

	void loadFramesData(string fileLocation){
		frameManager.loadFramesData(fileLocation);
	}

private:

	//Tracking auxiliar functions
	void initialyseLostPointsVector(){
		for (int i = 0; i < Flags::getKeypointsNumber(); i++){
			lostPoints.push_back(false);
		}
	}

	//Draw correspondent points, with oberservin a condition of correctness
	void drawCorrespondentPoints(Mat *image, CvPoint p1, CvPoint p2, bool condition, bool recuperate = false){
		bool details = false;
		int radius = 2;

		Scalar cPoint1, cPoint2, cLine;

		if (recuperate){
			if (details) cout << "Reuperate Point\n";
			cPoint1 = CV_RGB(255, 0, 255);
			cPoint2 = CV_RGB(100, 0, 100);
			cLine = CV_RGB(200, 0, 200);
		}
		else{
			if (condition){
				if (details) cout << "Good Point\n";
				cPoint1 = CV_RGB(255, 255, 0);
				cPoint2 = CV_RGB(200, 0, 0);
				cLine = CV_RGB(255, 100, 0);
			}
			else{
				if (details) cout << "BadPoint\n";
				cPoint1 = CV_RGB(0, 255, 255);
				cPoint2 = CV_RGB(0, 0, 200);
				cLine = CV_RGB(0, 100, 255);
			}
		}
		line(*image, p1, p2, cLine, 1);
		circle(*image, p1, radius, cPoint1, -1, CV_AA, 0);
		circle(*image, p2, radius, cPoint2, -1, CV_AA, 0);
	}
};