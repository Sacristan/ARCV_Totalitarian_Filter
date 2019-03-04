#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	Mat image_scene_original = imread("images/ussr_sucks/source_ussr_6.jpg");
	Mat image_template_original = imread("images/ussr_sucks/template_ussr.jpg");

//Mat image_scene_original = imread("images/nazi_germany_sucks/source_nazi_3.jpg");
//Mat image_template_original = imread("images/nazi_germany_sucks/template_nazi.jpg");

	Mat image_scene;
	Mat image_template;

	image_scene_original.copyTo(image_scene);
	image_template_original.copyTo(image_template);

	//Mat img_1_gray;
	//cvtColor(image_scene, image_scene, COLOR_BGR2GRAY);
	//blur(image_scene, image_scene, Size(10, 10));
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

	vector< DMatch > good_matches;
	
	for (int i = 0; i < descriptors_2.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 30.0))
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

	waitKey(0);

	return 0;
}