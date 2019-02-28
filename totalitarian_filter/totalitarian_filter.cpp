#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

const float homographyReprojectionThreshold = 2.0f;
Mat _homographyRough;

bool getHomography(const std::vector<cv::KeyPoint>& queryKeypoints, const std::vector<cv::KeyPoint>& trainKeypoints, float reprojectionThreshold, std::vector<cv::DMatch>& matches, cv::Mat& homography, int iterations, double confidece)
{
	const int minNumberMatchesAllowed = 10;
	if (matches.size() < minNumberMatchesAllowed) return false;

	// Prepare data for cv::findHomography
	std::vector<cv::Point2f> srcPoints(matches.size());
	std::vector<cv::Point2f> dstPoints(matches.size());
	for (size_t i = 0; i < matches.size(); i++)
	{
		srcPoints[i] = trainKeypoints[matches[i].queryIdx].pt;
		dstPoints[i] = queryKeypoints[matches[i].trainIdx].pt;
	}

	// Find homography matrix and get inliers mask
	std::vector<unsigned char> inliersMask(srcPoints.size());
	homography = cv::findHomography(srcPoints, dstPoints, FM_RANSAC, reprojectionThreshold, inliersMask, iterations, confidece);

	std::vector<cv::DMatch> inliers;
	for (size_t i = 0; i < inliersMask.size(); i++)
	{
		if (inliersMask[i])
			inliers.push_back(matches[i]);
	}

	matches.swap(inliers);

	return matches.size() > minNumberMatchesAllowed;
}

int main(int argc, char** argv)
{
	Mat image_scene_original = imread("images/ussr_sucks/source_ussr_6.jpg");
	Mat image_template_original = imread("images/ussr_sucks/template_ussr.jpg");

	Mat image_scene;
	Mat image_template;

	image_scene_original.copyTo(image_scene);
	image_template_original.copyTo(image_template);

	//Mat image_scene = imread("images/nazi_germany_sucks/source_nazi_3.jpg");
	//Mat image_template = imread("images/nazi_germany_sucks/template_nazi.jpg");

	//Mat img_1_gray;
	cvtColor(image_scene, image_scene, COLOR_BGR2GRAY);
	blur(image_scene, image_scene, Size(10, 10));
	//cv::adaptiveThreshold(img_1, img_1, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 11, 2);	//block size nepara skaitlis >= 3

	std::vector<KeyPoint> keypoints_scene, keypoints_template;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(image_scene, keypoints_scene);
	detector->detect(image_template, keypoints_template);

	descriptor->compute(image_scene, keypoints_scene, descriptors_1);
	descriptor->compute(image_template, keypoints_template, descriptors_2);

	Mat outimg1;
	drawKeypoints(image_scene, keypoints_scene, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("ORB", outimg1);

	vector<DMatch> matches;
	matcher->match(descriptors_1, descriptors_2, matches);

	double min_dist = 10000, max_dist = 0;

	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	//vector< DMatch > good_matches;
	
	//for (int i = 0; i < descriptors_2.rows; i++)
	//{
	//	if (matches[i].distance <= max(2 * min_dist, 30.0))
	//	{
	//		good_matches.push_back(matches[i]);
	//	}
	//}

	vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_2.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}

	//Mat img_match;
	Mat img_goodmatch;
	//drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
	
	drawMatches(
		image_scene,
		keypoints_scene,
		image_template,
		keypoints_template,
		good_matches, 
		img_goodmatch, 
		
		Scalar::all(-1),
		Scalar::all(-1),
		std::vector<char>(),
		DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
	);

	//imshow("All matching Points", img_match);

	imshow("Filtered matching Points", img_goodmatch);

	//std::vector<Point2f> obj;
	//std::vector<Point2f> scene;

	//for (int i = 0; i < good_matches.size(); i++)
	//{
	//	//-- Get the keypoints from the good matches
	//	obj.push_back(keypoints_template[good_matches[i].queryIdx].pt);
	//	scene.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
	//}

	//Mat H = findHomography(obj, scene, RANSAC);

	////-- Get the corners from the image_1 ( the object to be "detected" )
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(image_template.cols, 0);
	//obj_corners[2] = cvPoint(image_template.cols, image_template.rows); obj_corners[3] = cvPoint(0, image_template.rows);
	//std::vector<Point2f> scene_corners(4);

	//perspectiveTransform(obj_corners, scene_corners, H);

	////-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_goodmatch, scene_corners[0] + Point2f(image_template.cols, 0), scene_corners[1] + Point2f(image_template.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_goodmatch, scene_corners[1] + Point2f(image_template.cols, 0), scene_corners[2] + Point2f(image_template.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_goodmatch, scene_corners[2] + Point2f(image_template.cols, 0), scene_corners[3] + Point2f(image_template.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_goodmatch, scene_corners[3] + Point2f(image_template.cols, 0), scene_corners[0] + Point2f(image_template.cols, 0), Scalar(0, 255, 0), 4);

	////-- Show detected matches
	//imshow("Good Matches & Object detection", img_goodmatch);

	std::vector<Point2f> obj_corners(4), scene_corners(4);

	obj_corners[0] = cv::Point(0, 0);
	obj_corners[1] = cv::Point(image_template.cols, 0);
	obj_corners[2] = cv::Point(image_template.cols, image_template.rows);
	obj_corners[3] = cv::Point(0, image_template.rows);

	if (getHomography(keypoints_scene, keypoints_template, homographyReprojectionThreshold, good_matches, _homographyRough, 1000, 0.995)) {

		cv::perspectiveTransform(obj_corners, scene_corners, _homographyRough);

		cv::line(image_scene_original, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
		cv::line(image_scene_original, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
		cv::line(image_scene_original, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
		cv::line(image_scene_original, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);

		for (int i = 0; i < good_matches.size(); i++) {
			cv::circle(image_scene_original, keypoints_scene[good_matches[i].trainIdx].pt, 2, cv::Scalar(0, 255, 0), 1);
		}
	}

	imshow("FINAL", image_scene_original);


	//std::vector<Point2f> obj;
	//std::vector<Point2f> scene;

	//for (unsigned int i = 0; i < good_matches.size(); i++)
	//{
	//	//-- Get the keypoints from the good matches
	//	obj.push_back(keypoints_scene[good_matches[i].queryIdx].pt);
	//	scene.push_back(keypoints_template[good_matches[i].trainIdx].pt);
	//}

	//Mat H = findHomography(obj, scene, RANSAC);

	////-- Get the corners from the image_1 ( the object to be "detected" )
	//std::vector<Point2f> obj_corners(4);
	//obj_corners[0] = cvPoint(0, 0); obj_corners[1] = cvPoint(img_2.cols, 0);
	//obj_corners[2] = cvPoint(img_2.cols, img_2.rows); obj_corners[3] = cvPoint(0, img_2.rows);
	//std::vector<Point2f> scene_corners(4);

	//perspectiveTransform(obj_corners, scene_corners, H);

	////-- Draw lines between the corners (the mapped object in the scene - image_2 )
	//line(img_goodmatch, scene_corners[0] + Point2f(img_2.cols, 0), scene_corners[1] + Point2f(img_2.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_goodmatch, scene_corners[1] + Point2f(img_2.cols, 0), scene_corners[2] + Point2f(img_2.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_goodmatch, scene_corners[2] + Point2f(img_2.cols, 0), scene_corners[3] + Point2f(img_2.cols, 0), Scalar(0, 255, 0), 4);
	//line(img_goodmatch, scene_corners[3] + Point2f(img_2.cols, 0), scene_corners[0] + Point2f(img_2.cols, 0), Scalar(0, 255, 0), 4);

	////-- Show detected matches
	//imshow("Good Matches & Object detection", img_goodmatch);
	
	waitKey(0);

	return 0;
}