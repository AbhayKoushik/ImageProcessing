#include <opencv2/opencv.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/video/video.hpp>
#include <vector>
#include <iostream>
#include <math.h>
#include <string>
#include <fstream>
using namespace std;
using namespace cv;

class imageFLow
{
private:
	VideoCapture vidObj;
	//VideoWriter outVid;
	vector<Mat> video;
	Mat frame, frame1, frame2, frames[3];
	Mat gradx, grady,gradmag;
	Mat temp[2],flow[2];
	vector<double> flowSignal;
	Scalar magni, imageflow;
	double totalframes, fps, fheight, fwidth,mag;
	Size fsize;
	int n, count;
public:
	imageFLow(string path)
	{
		if (!vidObj.open(path)) {
			cout << "Cannot open file" << endl;
			exit(0);
		}
		totalframes = vidObj.get(CAP_PROP_FRAME_COUNT);
		fps = vidObj.get(CAP_PROP_FPS);
		fheight = vidObj.get(CAP_PROP_FRAME_HEIGHT);
		fwidth = vidObj.get(CAP_PROP_FRAME_WIDTH);
		fsize.height = fheight;
		fsize.width = fwidth;
		n = 3;//for downsampling
		count = 0;
		mag = 0;
		imageflow=0;
		flowSignal.clear();
	}
	void getImageFlow();
	void compute_flow();
	Mat gradientX(Mat);
	Mat gradientY(Mat);
};

void imageFLow::getImageFlow()
{
	fstream file1;
	Size s;
	s.height = 120;
	s.width = 160;
	file1.open("flowValues.txt", ios::out | ios::trunc);
	//Downsampling and Smoothing
	for (int i = 1; i < totalframes - n + 1; i = i + 3)
	{

		cout <<"Number of frames processed: "<<i<<"\n";
		cout << string(4, '\b');
		vidObj.set(CAP_PROP_POS_FRAMES, i);
		vidObj >> frames[0];
		vidObj >> frames[1];
		vidObj >> frames[2];
		frame2 = frames[0] / 4 + frames[1] / 2 + frames[2] / 4;
		cvtColor(frame2, frame2, COLOR_BGR2GRAY);
		// frame2.convertTo(frame2, CV_8UC1);
		GaussianBlur(frame2, frame2, Size(21, 21), 8, 8);
		// GaussianBlur(frame2, frame2, Size(21, 21), 8, 8);
		resize(frame2, frame2, s, INTER_LINEAR);
		//Processing
		if (count > 0)
		{
			compute_flow();
			multiply(flow[0], flow[0], temp[0]);//, 1, CV_8UC1);
			multiply(flow[1], flow[1], temp[1]);// 1, CV_8UC1);
		    imageflow = sum(temp[0])+sum(temp[1]);
		    //cout<<"image flow"<<imageflow;
		    flowSignal.push_back(double(imageflow[0]));
		}
		frame1 = frame2;
		count++;
	}
	cout << endl;
	for (int i = 0; i <flowSignal.size(); i++)
	{
		file1 << flowSignal[i] << endl;
	}
	cout << "Done Processing" << endl;
	return;
}

Mat imageFLow::gradientX(Mat A)
{
	Mat B;
	Size s;
	s.height = A.rows;
	s.width = A.cols;
	B.create(A.rows, A.cols, CV_8UC1);
	int cols = A.cols, rows = A.rows;
	Scalar int1, int2;
	for (int y = 0; y < A.rows; y++)
	{
		for (int x = 0; x < A.cols; x++)
		{
			if (x == 0)
			{
				int1 = (uchar)(A.at<uchar>(Point(x + 1, y)) - A.at<uchar>(Point(x, y)));
			}
			else if (x == A.cols - 1)
			{	
				int1 = (uchar)(A.at<uchar>(Point(x, y)) - A.at<uchar>(Point(x - 1, y)));
			}
			else
			{
				int1 = (uchar)(A.at<uchar>(Point(x + 1, y)) - A.at<uchar>(Point(x - 1, y))) / 2;
			}
			B.at<uchar>(Point(x, y)) = int1.val[0];
		}
	}
	return B;
}

Mat imageFLow::gradientY(Mat A)
{
	Size s;
	s.height = A.rows;
	s.width = A.cols;
	Mat B;
	B.create(A.rows, A.cols, CV_8UC1);
	int cols = A.cols, rows = A.rows;
	Scalar int1, int2;
	for (int x = 0; x < A.cols; x++)
	{
		for (int y = 0; y < A.rows; y++)
		{
			if (y == 0)
			{
				int1 = (uchar)(A.at<uchar>(Point(x, y + 1)) - A.at<uchar>(Point(x, y)));
			}
			else if (y == A.rows - 1)
			{
				int1 = (uchar)(A.at<uchar>(Point(x, y)) - A.at<uchar>(Point(x, y - 1)));
			}
			else
			{
				int1 = (uchar)(A.at<uchar>(Point(x, y + 1)) - A.at<uchar>(Point(x, y - 1))) / 2;
			}
			B.at<uchar>(Point(x, y)) = int1.val[0];
		}
	}
	return B;
}

void imageFLow::compute_flow()
{
	int i;
	Mat ImageDiff = frame2 - frame1;
	// ImageDiff.convertTo(ImageDiff, CV_8UC1);
	Mat temp1, temp2, temp3, absgradx, absgrady;
	gradx = gradientX(frame2);
	grady = gradientY(frame2);
	multiply(gradx, gradx, temp2);//, 1, CV_8UC1);
	multiply(grady, grady, temp3);//, 1, CV_8UC1);
	gradmag = temp2 + temp3;
	divide(ImageDiff, gradmag, temp1);//, 1, CV_8UC1);
	multiply(gradx, temp1, flow[0]);//, 1, CV_8UC1);
	multiply(grady, temp1, flow[1]);//, 1, CV_8UC1);
	return;
}

int main()
{
	timespec time1, time2;
	clock_gettime(CLOCK_REALTIME, &time1);
	imageFLow I("testVideo.avi"); //baby_breathing_not_breathing.mp4
	I.getImageFlow();
    clock_gettime(CLOCK_REALTIME, &time2);
    cout<<(time2.tv_sec-time1.tv_sec)<<" seconds"<<endl;//<<":"<<diff(time1,time2).tv_nsec<<endl;
	return 0;	
}	