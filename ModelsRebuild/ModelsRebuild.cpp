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

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/surface/marching_cubes_rbf.h>

#include "OpenGLWindow.h"

#pragma comment(lib, "OpenGL32.lib")


using namespace cv;
using namespace cv::xfeatures2d;

typedef struct Mesh 
{ 
	Mat vertices; 
	Mat faces; 

	Mesh(Mat v, Mat f) :vertices(v), faces(f) {}; 
} Mesh;

using namespace std;

static const float IMG_WIDTH = 800;
static const float IMG_HEIGHT = 600;

std::vector<cv::Point3f> Create3DChessboardCorners(cv::Size boardSize, float squareSize);
void SearchAndCompare(Mat &frame1, Mat &frame2, Mat &distortionMatrix, Mat &cameraMatrix, std::vector<std::vector<Point2f>> &keypoints);

int main()
{
	// Сюда загружу картинки
	std::vector<Mat> frames;

	// Тут ключевые точки каждого кадра
	std::vector<std::vector<Point2f>> keypoints;

	// Загружаю параметры моей камеры из файлов
	CvMat* intrinsic = (CvMat*)cvLoad("intrinsics.xml");
	CvMat* distortion = (CvMat*)cvLoad("distortion.xml");

	// Преобразую их в матрицы
	Mat cameraMatrix = cvarrToMat(intrinsic);
	Mat distortionMatrix = cvarrToMat(distortion);

	// Из внутренних параметров камеры забираю фокусное расстояние и principle point
	double focal_length = cameraMatrix.at<float>(0, 0);
	Point2d principle_point = Point2d(cameraMatrix.at<float>(0, 2), cameraMatrix.at<float>(1, 2));

	// Без этого умножение матриц кривое
	cameraMatrix.convertTo(cameraMatrix, CV_64FC1);
	distortionMatrix.convertTo(distortionMatrix, CV_64FC1);

	// Загружаю изображения
	//frames.push_back(imread("frame00.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("frame01.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("frame03.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("frame04.jpg", CV_LOAD_IMAGE_GRAYSCALE));

	//frames.push_back(imread("Model\\frame01s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("Model\\frame02s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("Model\\frame03s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("Model\\frame04s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("Model\\frame05s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("Model\\frame06s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	//frames.push_back(imread("Model\\frame07s.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame08.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame09.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame10.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame11.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame12.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame13.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame14.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame15.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame16.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame17.jpg", CV_LOAD_IMAGE_GRAYSCALE));
	frames.push_back(imread("Model\\frame18.jpg", CV_LOAD_IMAGE_GRAYSCALE));



	// Ищу точки соответствия между каждой соседней парой кадров
	for (size_t i = 0; i < frames.size() - 1; i++)
		SearchAndCompare(frames[i], frames[i + 1], distortionMatrix, cameraMatrix, keypoints);


#if _DEBUG
	cv::Mat test;
	cv::hconcat(frames[0], frames[1], test);

	for (int i = 0; i < keypoints[0].size(); i++)
	{
		cv::line(test, keypoints[0][i], cv::Point2f(keypoints[1][i].x + frames[1].cols, keypoints[1][i].y),
			1, 1, 0);
	}

	cv::imwrite("match-result.png", test);
#endif


	// Поворот и перенос камеры для каждого кадра, относительно первого
	cv::Mat initial_camera_matrix = cv::Mat::eye(3, 4, CV_64FC1);

	cout << "initial_camera_matrix:" << endl << " " << initial_camera_matrix << endl << endl;

	// Буду работать только с двумя изображениями
	for (size_t i = 0; i < keypoints.size() - 1; i += 2)
	{
		// используя матрицу перехода ищу преобразование камеры между кадрами
		cv::Mat mask;

		cv::Matx33d essential_matrix = cv::findEssentialMat(keypoints[i], keypoints[i + 1], focal_length, principle_point, cv::RANSAC, 0.999, 1.0, mask);

		mask.release();

		// Восстанавливаю матрицу преобразования камеры между двумя кадрами
		cv::Mat rotate;
		cv::Mat translate;

		cv::recoverPose(essential_matrix, keypoints[i], keypoints[i + 1], rotate, translate, focal_length, principle_point, mask);

		mask.release();

		cv::Mat result_matrix = cv::Mat::eye(3, 4, CV_64FC1);

		rotate.copyTo(result_matrix.rowRange(0, 3).colRange(0, 3));
		translate.copyTo(result_matrix.rowRange(0, 3).col(3));

		cout << "result_matrix:" << endl << " " << result_matrix << endl << endl;

		cv::Mat recovered_points(4, keypoints[i].size(), CV_64FC1);

		try
		{
			// Восстанавливаю позиции точек в 3D
			cv::triangulatePoints(cameraMatrix * initial_camera_matrix, cameraMatrix * result_matrix, keypoints[i], keypoints[i + 1], recovered_points);
		}
		catch (Exception ex)
		{
			return 0;
		}

		assert(recovered_points.cols == keypoints[i].size());

		initial_camera_matrix = result_matrix;

		ofstream outfile("pointcloud.ply");

		outfile << "ply\n" << "format ascii 1.0\n" << "comment VTK generated PLY File\n";
		outfile << "obj_info vtkPolyData points and polygons : vtk4.0\n" << "element vertex " << recovered_points.cols << "\n";
		outfile << "property float x\n" << "property float y\n" << "property float z\n" << "element face 0\n";
		outfile << "property list uchar int vertex_indices\n" << "end_header\n";

		for (size_t i = 0; i < recovered_points.cols; i++)
		{
			outfile << recovered_points.at<float>(0, i) / recovered_points.at<float>(3, i) << " "; // Деление - это восстановление из однородных координат
			outfile << recovered_points.at<float>(1, i) / recovered_points.at<float>(3, i) << " ";
			outfile << recovered_points.at<float>(2, i) / recovered_points.at<float>(3, i) << " ";

			outfile << "\n";
		}

		outfile.close();

		// Буду работать только с двумя изображениями
		break;
	}


	std::cout << "press key to exit" << endl;

	return 0;
}

void SearchAndCompare(Mat &frame1, Mat &frame2, Mat &distortionMatrix, Mat &cameraMatrix, std::vector<std::vector<Point2f>> &keypoints)
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
	Ptr<SURF> surf = SURF::create();

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

	std::vector<Point2f> gkpoints_img1, gkpoints_img2;

	for (size_t i = 0; i < good_matches.size(); i++)
	{
		gkpoints_img1.push_back(keypoints_img1[good_matches[i].queryIdx].pt);
		gkpoints_img2.push_back(keypoints_img2[good_matches[i].trainIdx].pt);

		cout << endl << endl;
		cout << keypoints_img1[good_matches[i].queryIdx].pt.x << "\t" << keypoints_img1[good_matches[i].queryIdx].pt.y << endl;
		cout << keypoints_img2[good_matches[i].trainIdx].pt.x << "\t" << keypoints_img2[good_matches[i].trainIdx].pt.y << endl;
	}

	// попарно сохраняю ключевые точки для каждых двух кадров
	keypoints.push_back(gkpoints_img1);
	keypoints.push_back(gkpoints_img2);

//---------------------------------------------------------------------------
	{
		using namespace pcl;

		PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);

		if (io::loadPLYFile<PointXYZ>("pointcloud.ply", *cloud) == -1) {
			cout << "ERROR: couldn't find file" << endl;
			return;
		}
		else {
			cout << "loaded" << endl;

			NormalEstimationOMP<PointXYZ, Normal> ne;
			search::KdTree<PointXYZ>::Ptr tree1(new search::KdTree<PointXYZ>);
			tree1->setInputCloud(cloud);
			ne.setInputCloud(cloud);
			ne.setSearchMethod(tree1);
			ne.setKSearch(20);
			ne.setNumberOfThreads(2);

			PointCloud<Normal>::Ptr normals(new PointCloud<Normal>);
			ne.compute(*normals);

			// Concatenate the XYZ and normal fields*
			PointCloud<PointNormal>::Ptr cloud_with_normals(new PointCloud<PointNormal>);
			concatenateFields(*cloud, *normals, *cloud_with_normals);

			// Create search tree*
			search::KdTree<PointNormal>::Ptr tree(new search::KdTree<PointNormal>);
			tree->setInputCloud(cloud_with_normals);

			cout << "begin marching cubes reconstruction" << endl;


			
			MarchingCubesRBF<PointNormal> mc;
			mc.setInputCloud(cloud_with_normals);

			PolygonMesh mesh;
			mc.reconstruct(mesh);

			Mesh result(Mat(mesh.cloud.width * mesh.cloud.height, 4, CV_32FC1), Mat(mesh.polygons.size(), 3, CV_32SC1));

			//PolygonMesh::Ptr triangles(new PolygonMesh);


			//mc.setSearchMethod(tree);
			//mc.reconstruct(*triangles);

			cout << mesh.polygons.size() << " triangles created" << endl;
		}
	}

}

// convert an uncommon PCL mesh representation to ours
void convert(Mesh dst, std::vector<pcl::Vertices> faces)
{
	for (int i = 0; i<faces.size(); i++) {
		assert(faces[i].vertices.size() == 3);
		for (char j = 0; j<3; j++) {
			dst.faces.at<int32_t>(i, j) = faces[i].vertices[j];
		}
	}
}
