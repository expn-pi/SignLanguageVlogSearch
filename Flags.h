#pragma once

#include "stdafx.h"
#include <stdio.h>
#include <conio.h>
#include <vector>

using namespace std;

class Flags
{
private:

	static Flags *instance;
	
	static string nameOfDetector;
	static string nameOfExtractor;
	static string nameOfTracker;//There is no a generic constructor - using a function
	static string nameOfMatcher;//There is no a generic constructor - using a function

	//File location
	static string path;
	static string file;
	static string dataExtension;
	static string movieExtension;
	static string fileLocation;
	static string fileMovieName;
	static string fileDataName;
		
	static bool createFolders;
	static bool showImage;//Show the frame from file

	static bool debug;//Debug the execution
	static bool details;//Show details of debug

	static bool newKeypoints;//Detect new keypoints
	static bool showKeyPoints;
	static int keyPointsNumber;

	static bool newTracking;//Track new points
	static bool showTracking;
	static double trackerError;
	static double trackerLostMax;
	static bool recuperateTrackerPoints;

	static bool newDescriptors;//Extract descriptors from keypoints
	static bool newMatches;//Extract Matches from descriptors
	static bool showMatches;
	static float matcherError;
	static bool getBestMach;
	static int bestFrameToMatch;

	static bool loadSaved;//Load file data
	static bool useDetected;//Use aways new detections to track
	static bool changeTrackPoints;//Change detect points to track points
	static bool armazenateFrameData;//Save the data coleted in eatch frame

	Flags(){}

public:
	static vector<string> kindsOfFile;

	enum filesKind{
		keyPoints,
		descriptors,
		matches,
		input,
		output,
		status,
		error
	};

	static string getDetectorName(){
		return nameOfDetector;
	}

	static string getTrackerName(){
		return nameOfTracker;
	}

	static string getExtractorName(){
		return nameOfExtractor;
	}

	static string getMatcherName(){
		return  nameOfMatcher;
	}

	static Flags& getInstance(){
		if (instance == NULL){
			instance = new Flags();
		}
		else return *instance;
	}

	static string getFileLocation(){
		return fileLocation;
	}

	static string getFileDataName(){
		return fileDataName;
	}

	static string getFileDataExtension(){
		return dataExtension;
	}

	static string getFileMovieName(){
		return fileMovieName;
	}

	static string getFileMovieExtension(){
		return movieExtension;
	}

	/*static int getIndex(){
		return index;
	}
	*/

	static bool isCreateFolders(){
		return createFolders;
	}

	static bool isShowImage(){
		return showImage;
	}

	static bool isGetNewData(){
		return newKeypoints || newTracking || newDescriptors || newMatches;
	}

	static bool isDebug(){
		return debug;
	}

	static bool isDetails(){
		return details;
	}

	static bool isNewKeypoints(){
		return newKeypoints;
	}
	
	static bool isShowkeypoints(){
		return showKeyPoints;
	}
	static int getKeypointsNumber(){
		return keyPointsNumber;
	}

	static bool isNewTracking(){
		return newTracking;
	}
	static bool isShowTracking(){
		return showTracking;
	}
	static double getTrackerError(){
		return trackerError;
	}
	static double getTrackerLostMax(){
		return trackerLostMax;
	}
	static bool isRecuperateTrackerPoints(){
		return recuperateTrackerPoints;
	}

	static bool isNewDescriptors(){
		return newDescriptors;
	}
	static bool isNewMatches(){
		return newMatches;
	}
	static bool isShowMatches(){
		return showMatches;
	}
	static float getMatcherError(){
		return matcherError;
	}
	static bool isGetBestMatch(){
		return getBestMach;
	}
	static int getBestFrameToMatch(){
		return bestFrameToMatch;
	}

	static bool isLoadSaved(){
		return loadSaved;
	}

	static bool isUseDetected(){
		return useDetected;
	}
	
	static bool isChangeTrackPoints(){
		return changeTrackPoints;
	}

	static void changeUseDetect(){
		useDetected = !useDetected;
	}

	static void changeTrackFont(){
		changeTrackPoints = !changeTrackPoints;
	}

	static bool isArmazenateFrameData(){
		return armazenateFrameData;
	}

	static void setArmazenateFrameData(){
		armazenateFrameData = true;
	}
};