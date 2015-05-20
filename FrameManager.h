#pragma once

#include "stdafx.h"
#include <stdio.h>
#include <conio.h>
#include <direct.h>

#include "opencv2\opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/nonfree/nonfree.hpp"

#include "stdafx.h"

#include "Flags.h"

using namespace cv;
using namespace std;

class FrameManager
{
public:
	FrameManager();
	~FrameManager();

	struct gfttParameters{
		int maxCorners;
		double qualityLevel;
		double minDistance;
		int blockSize = 3;
		bool harris = false;
		double k = 0.04;
	};

	struct frameData{
		//Vector of keypoints for the fetures extractor
		vector<KeyPoint> keyPoints;

		//KeyPoints descriptor for matching lost tracking
		Mat descriptors;

		//Maths of descriptors
		vector<DMatch> matches;

		//Input and output points of the tracker
		vector<Point2f> ptOutput;
		vector<Point2f> ptInput;

		//Status of each tracked point
		vector<uchar> status;

		//Error of each traked point
		vector<float> error;
	};

	vector<frameData> framesData;

	//getters
	vector <KeyPoint> * getKeypoints(int index){
		return &(getFrameData(index)->keyPoints);
	}

	Mat * getDescriptors(int index){
		return &(getFrameData(index)->descriptors);
	}

	vector<DMatch> * getMatches(int index){
		return &(getFrameData(index)->matches);
	}

	vector <Point2f> * getTrackerInput(int index){
		return &(getFrameData(index)->ptInput);
	}

	vector <Point2f> * getTrackerOutput(int index){
		return &(getFrameData(index)->ptOutput);
	}

	vector <uchar> * getTrackerStatus(int index){
		return &(getFrameData(index)->status);
	}

	vector <float> * getTrackerError(int index){
		return &(getFrameData(index)->error);
	}

	frameData * getFrameData(int index){

		bool details = false;

		if (index < framesData.size()){
			if (details) cout << "\t\tinput index: " << index << " of " << framesData.size() << ", Existente index\n";
			return &framesData[index];
		}
		else if (framesData.size() == index){
			if (details) cout << "\t\tinput index: " << index << " of " << framesData.size() << ", New index\n";
			frameData data;
			framesData.push_back(data);
			return &framesData[index];
		}
		else{
			cout << "\tIn function: getFrame\n";
			cout << "\t\tthis index could not be acsses now!\n";
			cout << "\t\tindex: " << index << ", last index: " << framesData.size()-1 << "\n";
			return NULL;
		}
	}
	
	int getSize(){
		return framesData.size();
	}


	//Debug
	void printFrameData(frameData actualData){
		int countStatus = 0;

		int vectorSize = actualData.ptOutput.size();
		for (int i = 0; i < vectorSize; i++){
			cout << i << ": status-> (" << (int)actualData.status[i] << ") , error: " << actualData.error[i] <<
				", coordenates (" << actualData.ptOutput[i].x << ", " << actualData.ptOutput[i].y << ")\n";

			countStatus += actualData.status[i];
		}

		cout << "Traked " << countStatus << " of " << vectorSize << "\n";
	}

	void printFramesData(){
		for (int i = 0; i < framesData.size(); i++){

			cout << "Frame: " << i << "\n\n";

			printFrameData(framesData[i]);

			cout << "\n";
		}
	}

	//Persistence
	void saveKeypointsFile(int index, FileStorage fs){
		string number = to_string(index);
		write(fs, "Frame-" + number, framesData[index].keyPoints);
	}

	void saveDescriptorsFile(int index, FileStorage fs){
		string number = to_string(index);
		write(fs, "Frame-" + number, framesData[index].descriptors);
	}
	
	//Override DMatch
	/*friend void operator>>(const cv::FileNode &node, std::vector<cv::DMatch> &dMatch) {

		node["distance"] >> dMatch.distance;
		node["imgIdx"] >> dMatch.imgIdx;
		//node["operator"] >> dMatch.operator<;
		node["queryIdx"] >> dMatch.queryIdx;
		node["trainIdx"] >> dMatch.trainIdx;

	}

	friend cv::FileStorage &operator<<(cv::FileStorage &fs, const std::vector<cv::DMatch> &dMatch) {
	
		fs << "{"
		<< "distance" << dMatch.distance
		<< "imgIdx" << dMatch.imgIdx
		//<< "operator" << dMatch.operator<
		<< "queryIdx" << dMatch.queryIdx
		<< "trainIdx" << dMatch.trainIdx
		<< "}";
		return fs;

	}*/
	

	void saveMatchesFile(int index, FileStorage fs){

		bool debug = false;

		string number = to_string(index);
		//write(fs, "Frame-" + number, framesData[index].matches);

		 vector<DMatch> v = framesData[index].matches;

		if(debug) cout << "Size of frames data: "<<framesData.size()<<"\n";

		if (debug) cout << "Size of matches: " << v.size() << "\n";
		
		if (index > 0){
			fs << "Frame-"+number<<"[";
			
			for (int i = 0; i < v.size(); i++){
				fs << "{";
					fs << "distance" << v[i].distance;
					fs << "imgIdx" << v[i].imgIdx;
					fs << "queryIdx" << v[i].queryIdx;
					fs << "trainIdx" << v[i].trainIdx;
				fs << "}";
			}

			fs << "]";
		 }
	}

	void saveInputFile(int index, FileStorage fs){
		string number = to_string(index);
		write(fs, "Frame-" + number, framesData[index].ptInput);
	}
	
	void saveOutputFile(int index, FileStorage fs){
		string number = to_string(index);
		write(fs, "Frame-" + number, framesData[index].ptOutput);
	}

	void saveStatusFile(int index, FileStorage fs){
		string number = to_string(index);
		write(fs, "Frame-" + number, framesData[index].status);
	}
	
	void saveErrorFile(int index, FileStorage fs){
		string number = to_string(index);
		write(fs, "Frame-" + number, framesData[index].error);
	}

	void saveParameters(string location, string nameOfDetector, gfttParameters parameters){
		
		string name = location + "//" + "configuration" + Flags::getFileDataExtension();
		FileStorage fs(name, FileStorage::WRITE);

		int frameCount = framesData.size();

		fs << "frameCount" << frameCount;

		fs << "nameOfDetector" << nameOfDetector;

		//Parameters of tracking manager
		fs << "maxCorners" << parameters.maxCorners;
		fs << "qualityLevel" << parameters.qualityLevel;
		fs << "minDistance" << parameters.minDistance;
		fs << "blockSize" << parameters.blockSize;
		fs << "harris" << parameters.harris;
		fs << "k" << parameters.k;

		fs.release();
	}

	void saveData(string location){

		for (int i = 0; i < Flags::kindsOfFile.size(); i++){
	
			cout << "Saving " + Flags::kindsOfFile[i] + ", as: " << location + "//" + Flags::kindsOfFile[i] + "\n";
			string finalName = location + "//" + Flags::kindsOfFile[i] + Flags::getFileDataExtension();

			FileStorage fs(finalName, FileStorage::WRITE);

			//Study Point to function in a more simple code
			//typedef int (FrameManager::*function)(int, int, FileStorage);
			//vector<function> functions {saveKeypointsFile, saveDescriptorsFile, saveInputFile, saveOutputFile, saveStatusFile, saveErrorFile};
			for (int j = 0; j < framesData.size(); j++){

				cout << "\tFrame: " << j << "\n";

				switch (i){
				case Flags::filesKind::keyPoints: saveKeypointsFile(j, fs); break;
				case Flags::filesKind::descriptors: saveDescriptorsFile(j, fs); break;
				case Flags::filesKind::matches: saveMatchesFile(j, fs); break;
				case Flags::filesKind::input: saveInputFile(j, fs); break;
				case Flags::filesKind::output: saveOutputFile(j, fs); break;
				case Flags::filesKind::status: saveStatusFile(j, fs); break;
				case Flags::filesKind::error: saveErrorFile(j, fs); break;
				default:break;
				}
			}

			fs.release();
		}
	}
	
	//Change the location depende of type o processing
	void chanceLocation(string location, string *newLocation){
		if (Flags::isUseDetected()){
			*newLocation = location + "\\useDetected";
		}
		else{
			*newLocation = location + "\\notUseDetected";
		}
	}

	void saveFramesData(string location, string nameOfDetector, gfttParameters parameters){

		if (Flags::isCreateFolders()){
			//Create the folder
			_mkdir(location.c_str());
		}

		bool debug = true;

		string newLocation;
		chanceLocation(location, &newLocation);

		if (Flags::isCreateFolders()){
			_mkdir(newLocation.c_str());
		}

		if (debug) cout << "Starting to save\n";
		//save the actual configurations
		saveParameters(newLocation, nameOfDetector, parameters);

		//save all others data
		saveData(newLocation);
	}

	/*
	FileStorage fs(finalName, FileStorage::WRITE);

	//cout << "\n";

	fs << "frameCount" << frameCount;

	//Parameters of tracking manager

	fs << "nameOfDetector" << nameOfDetector;

	fs << "maxCorners" << parameters.maxCorners;
	fs << "qualityLevel" << parameters.qualityLevel;
	fs << "minDistance" << parameters.minDistance;
	fs << "blockSize" << parameters.blockSize;
	fs << "harris" << parameters.harris;
	fs << "k" << parameters.k;

	for (int i = 0; i < frameCount; i++){

	string number = to_string(i);

	cout << "Saving frame data: " << i << "\n";

	write(fs, "keyPoints-" + number, framesData[i].keyPoints);
	write(fs, "descriptors-" + number, framesData[i].descriptors);
	write(fs, "input-" + number, framesData[i].ptInput);
	write(fs, "output-" + number, framesData[i].ptOutput);
	write(fs, "status-" + number, framesData[i].status);
	write(fs, "error-" + number, framesData[i].error);
	}

	cout << "releasing the file\n";
	fs.release();
	*/

	void readMatch(FileNode kptFileNode, vector<DMatch> *dMatch){
		
		DMatch aux;

		FileNodeIterator it = kptFileNode.begin(), it_end = kptFileNode.end();

		int idx = 0;

		for (; it != it_end; ++it, idx++)
		{
			(*it)["distance"] >> aux.distance;
			(*it)["imgIdx"] >> aux.imgIdx;
			(*it)["queryIdx"] >> aux.queryIdx;
			(*it)["trainIdx"] >> aux.trainIdx;
			
			dMatch->push_back(aux);
		}
	}

	void loadData(string location, int frameCount){

		for (int i = 0; i < Flags::kindsOfFile.size(); i++){

			cout << "Loading " + Flags::kindsOfFile[i] + ", as: " << location + "//" + Flags::kindsOfFile[i] + "\n";
			string finalName = location + "//" + Flags::kindsOfFile[i] + Flags::getFileDataExtension();

			cout << "frameCount: " << frameCount << "\n";
			FileStorage fs(finalName, FileStorage::READ);
			

			for (int j = 0; j < frameCount; j++){
				string number = to_string(j);

				FileNode kptFileNode = fs["Frame-" + number];

				cout << "\t reading frame: " << j << "\n";

				frameData auxData;

				switch (i){
				case Flags::filesKind::keyPoints: read(kptFileNode, *getKeypoints(j)); break;
				case Flags::filesKind::descriptors: read(kptFileNode, *getDescriptors(j)); break;
				case Flags::filesKind::matches: readMatch(kptFileNode, getMatches(j)); break;
				case Flags::filesKind::input: read(kptFileNode, *getTrackerInput(j)); break;
				case Flags::filesKind::output: read(kptFileNode, *getTrackerOutput(j)); break;
				case Flags::filesKind::status: read(kptFileNode, *getTrackerStatus(j)); break;
				case Flags::filesKind::error: read(kptFileNode, *getTrackerError(j)); break;
					
				default:break;
				}
			}
			fs.release();			
		}
	}

	void loadParameters(string location, int *frameCount, gfttParameters *parameters){
		
		string name = location + "//" + "configuration" + Flags::getFileDataExtension();
		
		FileStorage fs(name, FileStorage::READ);

		*frameCount = (int)fs["frameCount"];
		string nameOfDetector = (string)fs["nameOfDetector"];
		
		cout << "found: " << *frameCount << " frames\n";
		cout << "Name of detector: " << nameOfDetector<<"\n";

		fs.release();
	}

	void loadFramesData(string location){
		bool debug = true;
		gfttParameters parameters;
		int frameCount;

		if (debug) cout << "Starting to load\n";

		string newLocation;
		chanceLocation(location, &newLocation);

		loadParameters(newLocation, &frameCount, &parameters);
		loadData(newLocation, frameCount);
	}

	void loadFramesData2(string location){
	
		string finalName = location + Flags::getFileDataExtension();

		frameData auxData;

		FileStorage fs(finalName, FileStorage::READ);

		int frameCount = (int)fs["frameCount"];

		cout << "loading frames data\n";

		vector<string> nodes = { "keyPoints-", "descriptors-", "input-", "output-", "status-", "error-" };

		for (int i = 0; i < frameCount; i++){

			vector<FileNode> kptFileNode;

			cout << i << ":";

			string number = to_string(i);

			for (int j = 0; j < nodes.size(); j++){
				kptFileNode.push_back(fs[nodes[j] + number]);
				cout << "\t" << j << ":" << nodes[j] << "\n";
			}

			read(kptFileNode[0], auxData.keyPoints);
			read(kptFileNode[1], auxData.descriptors);
			read(kptFileNode[2], auxData.ptInput);
			read(kptFileNode[3], auxData.ptOutput);
			read(kptFileNode[4], auxData.status);
			read(kptFileNode[5], auxData.error);

			framesData.push_back(auxData);
		}

		fs.release();
	}

private:
	bool debug = true;
	bool details = false;
};

