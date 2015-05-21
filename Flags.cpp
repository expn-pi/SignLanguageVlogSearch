#include "stdafx.h"
#include "Flags.h"
#include <stdio.h>
#include <conio.h>
#include <vector>

using namespace std;

Flags *Flags::instance;

string Flags::nameOfDetector = "Good Features to track";//Not actualy contructing with this string
//string Flags::nameOfDetector = "ORB";
string Flags::nameOfExtractor = "SIFT";
string Flags::nameOfTracker = "calcOpticalFlowPyrLK";//There is no a generic constructor - using a function
string Flags::nameOfMatcher = "BFMatcher";//There is no a generic constructor - using a function

//\NCSLGR - DataBase\NCSLGRv4\movies\biker buddy

//File location
string Flags::path = "..//NCSLGR-DataBase//NCSLGRv2//video files//master//";
string Flags::file = "539_219_small_0";

//string Flags::path = "..//NCSLGR-DataBase//NCSLGRv4//movies//biker buddy//";
//string Flags::file = "biker_buddy_1069_small_0";

string Flags::movieExtension = ".mov";
string Flags::dataExtension = ".yml";
string Flags::fileLocation = path + file;
string Flags::fileMovieName = fileLocation + movieExtension;
string Flags::fileDataName = fileLocation + dataExtension;

vector<string> Flags::kindsOfFile{ "keyPoints", "descriptors", "matches", "input", "output", "status", "error"};

//Create new folders for the files analised
bool Flags::createFolders = false;

//Show the frame from file
bool Flags::showImage = true;

//Debug the execution
bool Flags::debug = true;

//Show details of debug
bool Flags::details = false;

//Detect new keypoints
bool Flags::newKeypoints = false;
bool Flags::showKeyPoints = true;
int Flags::keyPointsNumber = 128;

//Track new points
bool Flags::newTracking = true;
bool Flags::showTracking = true;
double Flags::trackerError = 0.02;
double Flags::trackerLostMax = 0.99;
bool Flags::recuperateTrackerPoints = true;

//Use aways new detections to track
bool Flags::useDetected = false;

//Prepare to change the tracking points usage (aways initiate as false)
bool Flags::changeTrackPoints = false;

//Extract descriptors from keypoints and they respectives matches
bool Flags::newDescriptors = false;
bool Flags::newMatches = false;
bool Flags::showMatches = true;
float Flags::matcherError = 300.0f;
bool Flags::getBestMach = true;

//Load file data
bool Flags::loadSaved = true;

//Save data in respectives files
bool Flags::armazenateFrameData = false;