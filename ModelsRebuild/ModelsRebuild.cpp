// ModelsRebuild.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>

#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\xfeatures2d.hpp>

#include <iostream>
#include <list>
#include <vector>
#include <thread>

#include <fstream>

#include "OpenGLWindow.h"

#pragma comment(lib, "OpenGL32.lib")


using namespace cv;
using namespace cv::xfeatures2d;

using namespace std;

static const float IMG_WIDTH = 800;
static const float IMG_HEIGHT = 600;

bool OpenCamera(VideoCapture &cap);
bool CalibrationCamera(VideoCapture &cap, Mat &outResultFrame, Mat &outDistortionCoef, Mat &outCameraMatrix, Mat &outRvec, Mat &outTvec);
std::vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize);
void SearchAndCompare(Mat &frame1, Mat &frame2, Mat &distortionMatrix, Mat &cameraMatrix, Mat &outImage, std::vector<Point2f> &gkpoints_img1, std::vector<Point2f> &gkpoints_img2);

void DrawRedPoint(Mat img, int x, int y);
void TestProgram(VideoCapture &cap, list<Mat> &frames, Mat &distortionMatrix, Mat &cameraMatrix, std::vector<Point2f> &gkpoints_img1, std::vector<Point2f> &gkpoints_img2);

void TakePhoto();

int main()
{

	std::cout << "Starting ModelsRebuild..." << endl;

	VideoCapture cap;		// Camera
	Mat frame;				// Frame?
	Mat distortionMatrix;	// Matrix of distortionCoefficients
	Mat cameraMatrix;		// Matrix of camera
	Mat rvec;				// RotateVector
	Mat tvec;				// TranslationVector

	list<Mat> frames;

	std::vector<Point2f> keypoints_img1;
	std::vector<Point2f> keypoints_img2;
	Mat outImage;			// temp variable for search and compare


	//std::thread thrOpenGLWindow(CreateOpenGLWindow);
	//thrOpenGLWindow.detach();


	while (true)
	{
		std::cout << "What do you to do?" << endl;
		std::cout << "1) Open camera" << endl;
		std::cout << "2) Calibradion camera" << endl;
		std::cout << "3) Search and compare keypoints camera" << endl;
		std::cout << "4) TestProgramm" << endl;
		std::cout << "5) TakePhoto" << endl;
		std::cout << "6) Exit" << endl;

		char key = 0;
		cin >> key;

		switch(key)
		{
		case '1':
		{
			std::thread thrOpenCamera(OpenCamera, std::ref(cap));
			thrOpenCamera.detach();
			break;
		}
		case '2':
			CalibrationCamera(cap, frame, distortionMatrix, cameraMatrix, rvec, tvec);
			break;
		case '3':
		{	
			//Mat frame1;
			//if (!cap.read(frame1))
			//	cout << "Not read frame1" << endl;

			//std::this_thread::sleep_for(5s);

			//Mat frame2;
			//if (!cap.read(frame2))
			//	cout << "Not read frame2" << endl;
	
			Mat frame1 = imread("frame00.jpg", CV_LOAD_IMAGE_GRAYSCALE);	// surf работает только с grayscale изображениями
			Mat frame2 = imread("frame01.jpg", CV_LOAD_IMAGE_GRAYSCALE);	// surf работает только с grayscale изображениями

			SearchAndCompare(frame1, frame2, distortionMatrix, cameraMatrix, outImage, keypoints_img1, keypoints_img2);

			while (true)
			{
				cv::imshow("SearchAndCompare", outImage);
				cv::waitKey(10);
			}
			break;
		}
		case '4':
		{
			std::thread thrTestProgram(TestProgram, std::ref(cap), std::ref(frames), std::ref(distortionMatrix), std::ref(cameraMatrix), std::ref(keypoints_img1), std::ref(keypoints_img2));
			thrTestProgram.detach();
			break;
		}
		case '5':
			TakePhoto();
			break;
		case '6':
			exit(0);
			break;
		}

	}

	std::cout << "Goodbye!" << endl;
	return 0;
}

void TestProgram(VideoCapture &cap,  list<Mat> &frames, Mat &distortionMatrix, Mat &cameraMatrix, std::vector<Point2f> &keypoints_img1, std::vector<Point2f> &keypoints_img2)
{
	Mat frame1 = imread("frame00.jpg", CV_LOAD_IMAGE_GRAYSCALE);	// surf работает только с grayscale изображениями
	Mat frame2 = imread("frame01.jpg", CV_LOAD_IMAGE_GRAYSCALE);	// surf работает только с grayscale изображениями

	Mat outImage;
	SearchAndCompare(frame1, frame2, distortionMatrix, cameraMatrix, outImage, keypoints_img1, keypoints_img2);			// поиск ключевых точек и сравнение на двух изображениях 
	
//--поиск положения камеры для первого изображения------------------------------------------------------
	std::vector<Point3f> objectPoints1;
	
	// преобразовываем найденные ключевые точки изображения 1 в координаты мира openGL 
	for (auto it = keypoints_img1.begin(); it < keypoints_img1.end(); it++)
	{
		objectPoints1.push_back(Point3f(2 * ((it->x / IMG_WIDTH) - 0.5f), 2 * (1 - (it->y / IMG_HEIGHT) - 0.5f) , 0));
		cout << endl << endl << Point3f(2 * ((it->x / IMG_WIDTH) - 0.5f), 2 * (1 - (it->y / IMG_HEIGHT) - 0.5f), 0) << endl << endl;
	}

	Mat rvec1, tvec1;
	cv::solvePnP(objectPoints1, keypoints_img1, cameraMatrix, distortionMatrix, rvec1, tvec1);

	cout << "rvec1: " << rvec1 << endl;
	cout << "tvec1: " << tvec1 << endl;

	Mat rvecRod1;
	Rodrigues(rvec1, rvecRod1);

	cout << "rvecRod1: " << rvecRod1 << endl;
	Mat cameraPosition1;
	cameraPosition1 = rvecRod1.t() * tvec1;
	cout << "cameraPosition1: " << cameraPosition1 << endl;

	 //добавление камеры в качестве точки
	objectPoints1.push_back(Point3f(cameraPosition1.at<double>(0, 0),
		cameraPosition1.at<double>(1, 0),
		cameraPosition1.at<double>(2, 0)));

	CreateOpenGLWindow(objectPoints1);					// вспомогательная функция для отрисовки точек в openGL

	while (true)
	{
		cv::imshow("searchandcompare", outImage);
		if (cv::waitKey(10) == 27) break;
	}


//--Аналогичные действия для второго изображения--------------------------------------------------------------------------------------
	std::vector<Point3f> objectPoints2;

	for (auto it = keypoints_img2.begin(); it < keypoints_img2.end(); it++)
	{
		objectPoints2.push_back(Point3f(2 * ((it->x / IMG_WIDTH) - 0.5f), 2 * (1 - (it->y / IMG_HEIGHT) - 0.5f), 0));
		cout << endl << endl << Point3f(2 * ((it->x / IMG_WIDTH) - 0.5f), 2 * (1 - (it->y / IMG_HEIGHT) - 0.5f), 0) << endl << endl;
	}

	Mat rvec2, tvec2;
	solvePnP(objectPoints2, keypoints_img2, cameraMatrix, distortionMatrix, rvec2, tvec2);

	cout << "rvec2: " << rvec2 << endl;
	cout << "tvec2: " << tvec2 << endl;

	Mat rvecRod2;
	Rodrigues(rvec2, rvecRod2);

	cout << "rvecRod2: " << rvecRod2 << endl;
	Mat cameraPosition2;
	cameraPosition2 = rvecRod2.t() * tvec2;
	cout << "cameraPosition2: " << cameraPosition2 << endl;

	//добавление камеры в качестве точки
	objectPoints2.push_back(Point3f(cameraPosition2.at<double>(0, 0),
		cameraPosition2.at<double>(1, 0),
		cameraPosition2.at<double>(2, 0)));

	//Mat cam1;
	//Mat cam2;

	//cam1 = cameraMatrix * cam1;
	//cam2 = cameraMatrix * cam2;

	//cv::Mat point3D(4, keypoints_img1.size(), CV_64F);
	//cv::triangulatePoints(cam1, cam2, keypoints_img1, keypoints_img2, point3D);

	CreateOpenGLWindow(objectPoints2);

	while(true)
	{
		cv::imshow("searchandcompare", outImage);
		if (cv::waitKey(10) == 27) break;
	}
}

void DrawRedPoint(Mat img, int x, int y)
{
	img.at<Vec3b>(y, x)[0] = 0;
	img.at<Vec3b>(y, x)[1] = 0;
	img.at<Vec3b>(y, x)[2] = 255;

	img.at<Vec3b>(y - 1, x)[0] = 0;
	img.at<Vec3b>(y - 1, x)[1] = 0;
	img.at<Vec3b>(y - 1, x)[2] = 255;

	img.at<Vec3b>(y + 1, x)[0] = 0;
	img.at<Vec3b>(y + 1, x)[1] = 0;
	img.at<Vec3b>(y + 1, x)[2] = 255;

	img.at<Vec3b>(y, x + 1)[0] = 0;
	img.at<Vec3b>(y, x + 1)[1] = 0;
	img.at<Vec3b>(y, x + 1)[2] = 255;

	img.at<Vec3b>(y - 1, x + 1)[0] = 0;
	img.at<Vec3b>(y - 1, x + 1)[1] = 0;
	img.at<Vec3b>(y - 1, x + 1)[2] = 255;

	img.at<Vec3b>(y + 1, x + 1)[0] = 0;
	img.at<Vec3b>(y + 1, x + 1)[1] = 0;
	img.at<Vec3b>(y + 1, x + 1)[2] = 255;

	img.at<Vec3b>(y, x - 1)[0] = 0;
	img.at<Vec3b>(y, x - 1)[1] = 0;
	img.at<Vec3b>(y, x - 1)[2] = 255;

	img.at<Vec3b>(y - 1, x - 1)[0] = 0;
	img.at<Vec3b>(y - 1, x - 1)[1] = 0;
	img.at<Vec3b>(y - 1, x - 1)[2] = 255;

	img.at<Vec3b>(y + 1, x - 1)[0] = 0;
	img.at<Vec3b>(y + 1, x - 1)[1] = 0;
	img.at<Vec3b>(y + 1, x - 1)[2] = 255;
}


bool OpenCamera(VideoCapture &cap)
{
	cap.open(CV_CAP_ANY);

	if (!cap.isOpened())
		return false;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 800);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);
	
	Mat image;
	namedWindow("Video", CV_WINDOW_AUTOSIZE);

	while (cap.read(image))
	{
		//cap >> image;
		cv::imshow("Video", image);
		if (waitKey(1) == 27) break;		// ESC
	}

	return true;
}

bool CalibrationCamera(VideoCapture &cap, Mat &outResultFrame, Mat &outDistortionCoef, Mat &outCameraMatrix, Mat &outRvec, Mat &outTvec)
{
//--Создаем доску и вспомогательные переменные---------------------------------------------------------
	cv::Size boardSize(7, 7);
	cv::Size imageSize;
	std::vector<std::vector<cv::Point2f> > imageChessPoints(1);

	std::vector<std::vector<cv::Point3f> > objectPoints(1);

	std::vector<cv::Mat> rotationVectors;
	std::vector<cv::Mat> translationVectors;

	cv::Mat distortionCoefficients = cv::Mat::zeros(8, 1, CV_64F); // There are 8 distortion coefficients
	cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);


//--Получаем кадр и ищем в нем доску--------------------------------------------------------------------
	Mat frame;
	if (cap.read(frame))
	{
		frame = imread("frame05.jpg", IMREAD_COLOR);				// временная заглушка для калибровки камеры!


		bool found = findChessboardCorners(frame,					// image/frame
										   boardSize,				// count keypount
										   imageChessPoints[0]);	// params of search points;
		if (!found)
		{
			std::cerr << "Could not find chess board!" << std::endl;
			return false;
		}

		drawChessboardCorners(frame, boardSize, cv::Mat(imageChessPoints[0]), found);
		objectPoints[0] = Create3DChessboardCorners(boardSize, 1);

		imageSize = frame.size();

		int flags = 0;
		double rms = calibrateCamera(objectPoints,									//координаты ключевых точек в системе координат объета (x, y, z=0)
									 imageChessPoints,								//в системе координат изображения (u, v)
									 imageSize,										//размер изображения 
									 cameraMatrix,									//МОЖНО использовать уже известную матрицу камеры
									 distortionCoefficients,						//МОЖНО использовать уже известнные коэффициенты дисторсии
									 rotationVectors,								//критерии окончания минимизации 
									 translationVectors,							//какие коэффициенты дисторсии мы хотим получить
									 flags | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

		std::cout << "RMS: " << rms << std::endl;

		std::cout << "Camera matrix: " << cameraMatrix << std::endl;
		std::cout << "RotationVector" << rotationVectors[0] << "\tsize:" << rotationVectors.size() << std::endl;
		std::cout << "TranslationVector" << translationVectors[0] << "\tsize:" << translationVectors.size() << std::endl;
		std::cout << "Distortion _coefficients: " << distortionCoefficients << std::endl;

		outResultFrame= frame;
		outDistortionCoef = distortionCoefficients;
		outCameraMatrix = cameraMatrix;
		outRvec = rotationVectors[0];
		outTvec = translationVectors[0];

		return true;
	}
	
	std::cout << "frame not found" << endl;
	return false;
}

std::vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize)
{
	// This function creates the 3D points of your chessboard in its own coordinate system
	std::vector<cv::Point3f> corners;

	for (int i = 0; i < boardSize.height; i++)
		for (int j = 0; j < boardSize.width; j++)
			corners.push_back(cv::Point3f(float(j*squareSize),
				float(i*squareSize), 0));

	return corners;
}

void SearchAndCompare(Mat &frame1, Mat &frame2, Mat &distortionMatrix, Mat &cameraMatrix, Mat &outImage, std::vector<Point2f> &gkpoints_img1, std::vector<Point2f> &gkpoints_img2)
{
	//проверка на успешную загрузку
	if (!frame1.data && frame2.data)
	{
		cout << "Image is't loading!" << endl;
		return;
	}

	std::vector<KeyPoint> keypoints_img1, keypoints_img2;				// ключевые точки
	Mat descriptors_img1, descriptors_img2;								// descriptors (features)

																		//--Находим ключевые точки и вычисляем дескрипторы изображений
	Ptr<SIFT> surf = SIFT::create(20);									
	surf->detectAndCompute(frame1, noArray(), keypoints_img1, descriptors_img1);
	surf->detectAndCompute(frame2, noArray(), keypoints_img2, descriptors_img2);

	//-- Поиск похожих дескрипторов изображений
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_img1, descriptors_img2, matches);

	//cvFindExtrinsicCameraParams2()

	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_img1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	cout << "Max dist: " << max_dist << endl;
	cout << "Max dist: " << min_dist << endl;

	//-- Выбираем только "хорошие" точки (т.е чье расстояние меньше чем 6*min_dist )
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_img1.rows; i++)
		if (matches[i].distance < 4 * min_dist)			// Поиграться с коэффициетом!
			good_matches.push_back(matches[i]);

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		gkpoints_img1.push_back(keypoints_img1[good_matches[i].queryIdx].pt);
		gkpoints_img2.push_back(keypoints_img2[good_matches[i].trainIdx].pt);

		cout << endl << endl;
		cout << keypoints_img1[good_matches[i].queryIdx].pt.x << "\t" << keypoints_img1[good_matches[i].queryIdx].pt.y << endl;
		cout << keypoints_img2[good_matches[i].trainIdx].pt.x << "\t" << keypoints_img2[good_matches[i].trainIdx].pt.y << endl;
	}


	//-- Рисуем линнии, между найденными точками на новое изображение
	Mat img_matches;
	drawMatches(frame1, keypoints_img1, frame2, keypoints_img2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		std::vector<char>(), DrawMatchesFlags::DEFAULT);

	outImage = img_matches;
}

void TakePhoto()
{
	VideoCapture cap;

	cap.open(CV_CAP_ANY);

	if (!cap.isOpened())
		return;

	cap.set(CV_CAP_PROP_FRAME_WIDTH, 800);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, 600);

	Mat image;
	namedWindow("TakePhoto", CV_WINDOW_AUTOSIZE);

	while (cap.read(image))
	{
		//cap >> image;
		cv::imshow("TakePhoto", image);
		if (waitKey(1) == 27) break;		// ESC

		if (waitKey(1) == 13)				// ENTER
		{		
			static int k = 0;
			char filename[128];
			sprintf_s(filename, "frame0%d.jpg", k++);
			imwrite(filename, image);
		}
	}
}