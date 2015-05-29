#include "stdafx.h"
#include "opencv2\opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "FrameManager.h"

using namespace cv;
using namespace std;

class TrackingPointsManager
{
private:

	struct Correspondence{
		Point2f p1;
		Point2f p2;
	};

	struct PointChangeStatus{
		bool detected = true;
		int lastFrame = -1;
		Point2f lastPosition;

		//List of the frame index of the last change positions
		vector <int> indexList;

		//List of the last change positions
		vector <Point2f> positionList;
	};

	struct MatchForFrame{
		//The actual frame that has matches with every others
		int frIdxQuery;
		
		//Matches of a frame with the actual
		vector <DMatch> matches;
	};

	struct LostPoint{
		//Index of the lost point
		int ptIdx;
		
		//Index of the frame with new keyPoints
		int frIdxTrain;

		//List of matches for every frame that this point has some valid features
		vector<MatchForFrame> matchesForFrames;
	};

	struct LostPoints{
		list <LostPoint> points;
		int count = 0;
		int lastIndex = 0;
	}lostPoints;

	vector <PointChangeStatus> status;
	
	int keyPointsFounded = 0;

	vector<Correspondence> recorverPoints;

public:
	TrackingPointsManager(){
	}

	TrackingPointsManager(int keyPointsFounded){
		this->keyPointsFounded = keyPointsFounded;
		
		lostPoints.count = 0;
		lostPoints.lastIndex = 0;
		
		initialyseLostPointsVector();
	}

	int getKeyPointsFounded(){
		return keyPointsFounded;
	}

	int getLostCount(){
		return lostPoints.count;
	}

	bool isDetected(int index){
		return status[index].detected;
	}

	int getLastFrame(int index){
		return status[index].lastFrame;
	}

	int getLastLostFrame(){
		return lostPoints.lastIndex;
	}

	void printMatches(vector<vector<DMatch>> matches){
		for (int i = 0; i < matches.size(); i++){
			cout << "\nPoint: " << i <<"\n";

			for (int j = 0; j < matches[i].size(); j++){
				cout << "Matching: " << j << "\n";

				cout << "distance: " << matches[i][j].distance << "\n";
				cout << "queryIdx: " << matches[i][j].queryIdx << "\n";
				cout << "trainIdx: " << matches[i][j].trainIdx << "\n";
			}
		}
	}

	void drawQueryPoint(Point2f ptQuery, Mat *frQuery, Mat *frTrain){
		int radius = Flags::getKeyPointsSize()*2;

		Scalar color1 = CV_RGB(255, 0, 0);
		Scalar color2 = CV_RGB(0, 255, 0);

		circle(*frQuery, ptQuery, radius, color1, 1);
		//circle(*frTrain, ptQuery, radius, color1, 1);
	}

	void drawLostMatch(Point2f ptQuery, Point2f ptTrain, Mat *frQuery, Mat *frTrain, int interation){
		bool details = false;
		Scalar color = CV_RGB(255, 0, 0);

		int radius = Flags::getKeyPointsSize()*2;

		float fInteration = (float)interation+1;
		if(details) cout << fInteration << "\n";

		float fRadius = (float)radius;
		if (details) cout << fRadius << "\n";

		float fMinorRadius = fRadius / fInteration;
		if (details) cout << fMinorRadius << "\n";

		int minorRadius = (int)fMinorRadius;
		if (details) cout << minorRadius << "\n";

		line(*frTrain, ptQuery, ptTrain, color);
		circle(*frTrain, ptTrain, minorRadius, color, 1);
	}

	void drawLostMatches(FrameManager frameManager){
		bool debug = true;
		bool details = false;

		list<LostPoint>::iterator pBegin = lostPoints.points.begin();
		list<LostPoint>::iterator pEnd = lostPoints.points.end();

		if (debug) cout << "\tStart to draw the new matches for the loose track points\n";
		for (std::list<LostPoint>::iterator point = pBegin; point != pEnd; point++){

			int ptIdx = point->ptIdx;

			if (details) cout << "\t\tAnalising the point: " << ptIdx << "\n";

			int frIdxTrain = point->frIdxTrain;
			//vector<Point2f> *trainOutput = frameManager.getTrackerOutput(frIdxTrain);
			vector<KeyPoint> *trainKeyPoints = frameManager.getKeypoints(frIdxTrain);

			if (debug) cout << "\n\t\t\tGetting the image with keyPoints for mathing: " << frIdxTrain << "\n";
			Mat *frTrain = frameManager.getFrameImage(frIdxTrain);
			//cout << "Frame image adress: " << frTrain << "\n";
			
			vector <MatchForFrame> mFFs = point->matchesForFrames;
			
			if (details) cout << "\t\tFind: " << mFFs.size() << " features frames\n";
			for (int i = 0; i < mFFs.size(); i++){
				if (debug) cout << "\t\t\tAnalising the: " << i + 1 << " of " << mFFs.size() << " match frames\n";
				
				int frIdxQuery = mFFs[i].frIdxQuery;

				if (debug) cout << "\n\t\t\t\tGetting the point image: " << frIdxQuery << "\n";
				Mat *frQuery = frameManager.getFrameImage(frIdxQuery);
				//cout << "Query image adress: " << frQuery << "\n";

				Mat frQueryClone = frQuery->clone();
				Mat frTrainClone = frTrain->clone();

				vector<Point2f> *queryOutput = frameManager.getTrackerOutput(frIdxQuery);

				Point2f ptQuery = (*queryOutput)[ptIdx];

				vector <DMatch> matches = mFFs[i].matches;

				int interation = 0;

				if (details) cout << "\t\t\t\tIndex of the train output: " << frIdxTrain << "\n";
				if (details) cout << "\t\t\t\tSize of the train points: " << trainKeyPoints->size() << "\n";

				for (int j = 0; j < matches.size(); j++){

					int trainIdx = matches[j].trainIdx;

					if (details) cout << "\t\t\t\t\tIndex of the matching train point: " << trainIdx << "\n";
					
					Point2f ptTrain = (*trainKeyPoints)[trainIdx].pt;

					drawLostMatch(ptQuery, ptTrain, &frQueryClone, &frTrainClone, interation);
					interation++;
				}

				drawQueryPoint(ptQuery, &frQueryClone, &frTrainClone);

				imshow("Feature", frQueryClone);
				imshow("Frame", frTrainClone);

				waitKey(200);
				waitKey();
			}
		}
	}

	void calculateLostMatchs(int frIdxTrain, int frIdxQuery, Mat *trackerDescriptors, BFMatcher matcher){
		bool debug = true;
		bool details = false;

		if (debug) cout << "\tLooking for points of frame: " << frIdxQuery << " that has valid features \n";

		//if (details) cout << "\t\t\tBefore enter in loop\n";
		list<LostPoint>::iterator pBegin = lostPoints.points.begin();
		list<LostPoint>::iterator pEnd = lostPoints.points.end();
		for (std::list<LostPoint>::iterator point = pBegin; point != pEnd; point++)
		{
			int ptIdx = point->ptIdx;

			if (details) cout << "\n\t\tAnalising tracking point: " << ptIdx << "\n";
			point->frIdxTrain = frIdxTrain;

			int lastFrIdx = status[ptIdx].lastFrame;
			if (details) cout << "\n\t\t\tLast point frame index: " << lastFrIdx << " > actual point frame index " << frIdxQuery << " ?\n";

			if (lastFrIdx > frIdxQuery){
				if (details) cout << "\t\t\tThis point has valid features for that frame \n";
				vector<vector<DMatch>> matches;

				if (details) cout << "\n\t\t\tTrying match with image of index: " << point->frIdxTrain << "\n";
			
				//if (details) cout << "\n\t\t\tAdd point descriptors: " << ptIdx << "\n";
				Mat pointDescriptor = trackerDescriptors->row(ptIdx);

				//if (details) cout << "\t\t\tTry KnnMatch with k = " << Flags::getKnnValure() << "\n";
				matcher.knnMatch(pointDescriptor, matches, Flags::getKnnValure());

				//if (details) cout << "\t\t\tPrint the matches\n";
				if (details) cout << "\t\t\tmatches size: " << matches[0].size() << "\n";
				//printMatches(matches);

				MatchForFrame mFF;
				mFF.frIdxQuery = frIdxQuery;
				mFF.matches = matches[0];

				point->matchesForFrames.push_back(mFF);
			}
			else{
				if (details) cout << "\n\t\This point don't have valid features for that frame\n";
			}
		}
	}

	void cleanLastMatchs(){
		list<LostPoint>::iterator pBegin = lostPoints.points.begin();
		list<LostPoint>::iterator pEnd = lostPoints.points.end();
		for (std::list<LostPoint>::iterator point = pBegin; point != pEnd; point++)
		{
			point->matchesForFrames.clear();
		}
	}

	void recuperateTrackerPoints(int frIdx, FrameManager frameManager){
		bool debug = true;
		bool details = false;

		if (debug) cout << "\tTrying recuperate the lost tracking\n";

		if (details) cout << "\t\tGetting the actual keypoints features\n";
		Mat *trainDescriptors = frameManager.getDescriptors(frIdx);

		if (details) cout << "\t\tGetting the last frame that has a losse keyPoint\n";
		int lastFrame = getLastLostFrame();

		cleanLastMatchs();

		vector <Mat> frameDescriptor;
		frameDescriptor.push_back(*trainDescriptors);
		//if (details) cout << "\t\tTrain descriptor size: " << trainDescriptors->size() << "\n";

		BFMatcher matcher;
		matcher.add(frameDescriptor);

		//vector <Point2f> *queryPoints = frameManager.getTrackerOutput(frIdx);

		if (debug) cout << "\t\tGetting matchs with actual frame: " << frIdx << "\n";
		for (int i = 0; i < lastFrame; i++){

			if (details) cout << "\t\t\tGetting the descriptors of frame: " << i << "\n";
			Mat *queryDescriptors = frameManager.getTrackerDescriptors(i);
			//if (details) cout << "\t\t\tQuery descriptor size: " << queryDescriptors->size() << "\n";
	
			calculateLostMatchs(frIdx, i, queryDescriptors, matcher);
		}
	}

	bool isLostFrame(int ptIdx, int frameIdx){

		if (status[ptIdx].detected){
			return false;
		}
		else{
			return  (status[ptIdx].lastFrame == frameIdx);
		}
	}

	Point2f getLastPointPosition(int index){
		return status[index].lastPosition;
	}

	void setStatus(int ptIdx, int frameIdx, Point2f position, bool detected){
		
		status[ptIdx].detected = detected;

		status[ptIdx].lastFrame = frameIdx;
		status[ptIdx].indexList.push_back(frameIdx);

		status[ptIdx].lastPosition = position;
		status[ptIdx].positionList.push_back(position);
	}

	void insertLostList(int ptIdx, int frameIdx){
		
		LostPoint lostPoint;

		lostPoint.ptIdx = ptIdx;
		lostPoints.points.push_back(lostPoint);
		
		if (lostPoints.lastIndex < frameIdx){
			lostPoints.lastIndex = frameIdx;
		}

		lostPoints.count++;
	}

	void setLostPoint(int ptIdx, int frameIdx, Point2f position){
		setStatus(ptIdx, frameIdx, position, false);
		insertLostList(ptIdx, frameIdx);
	}

	bool verifyLostTrackRate(float maximum){
		bool debug = true;

		float lostPercentual = 1.0f - (((float)getLostCount()) / ((float)getKeyPointsFounded()));

		if (debug) cout << "Lost percentual: " << lostPercentual*100.0f << "\n";

		if (lostPercentual < maximum){
			if (debug) "try recuperate points\n";

			return true;
		}
		else{
			return false;
		}
	}

	bool verifyAllPointsHistoric(){
		bool details = false;

		bool result = true;

		if (details) cout << "Count lost points: " << lostPoints.count << "\n";

		bool rigthCount = (lostPoints.count == 0);
		result = result && rigthCount;

		if (details) cout << "Ok?: " << rigthCount << "\n";

		if (details) waitKey();

		for (int i = 0; i < status.size(); i++){
			if (details) cout << "Point: " << i << "\n";

			bool detected = status[i].detected;
			if (details) cout << "\tDetected: " << detected << "\n";
			result = result && detected;

			int lastFrame = status[i].lastFrame;
			bool rigthLast = (lastFrame == -1);
			if (details) cout << "\tLast frame = -1: " << rigthLast << "\n";
			result = result && rigthLast;

			if (details) waitKey();
		}
		return result;
	}

private:
	void initialyseLostPointsVector(){

		lostPoints.count = 0;

		for (int i = 0; i < keyPointsFounded; i++){

			PointChangeStatus statusAux;

			statusAux.detected = true;
			statusAux.lastFrame = -1;

			status.push_back(statusAux);
		}
	}
};