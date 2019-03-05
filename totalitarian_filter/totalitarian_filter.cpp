#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/core/types_c.h>

using namespace std;
using namespace cv;

void Filter(int match_dis) {

	//Mat image_scene_original = imread("images/ussr_sucks/source_ussr_6.jpg"); //6
	//Mat image_template_original = imread("images/ussr_sucks/template_ussr.jpg");

	Mat image_scene_original = imread("images/nazi_germany_sucks/source_nazi_6.jpg"); //5, 6, 3
	Mat image_template_original = imread("images/nazi_germany_sucks/template_nazi.jpg");

	Mat image_scene;
	Mat image_template;

	Mat img_keypoints;
	Mat img_goodmatch;

	image_scene_original.copyTo(image_scene);
	image_template_original.copyTo(image_template);

	std::vector<KeyPoint> keypoints_scene, keypoints_template;
	Mat descriptors_1, descriptors_2;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();

	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

	detector->detect(image_scene, keypoints_scene);
	detector->detect(image_template, keypoints_template);

	descriptor->compute(image_scene, keypoints_scene, descriptors_1);
	descriptor->compute(image_template, keypoints_template, descriptors_2);

	drawKeypoints(image_scene, keypoints_scene, img_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	//imshow("Image keypoints ORB", img_keypoints);

	vector<DMatch> matches;
	matcher->match(descriptors_1, descriptors_2, matches);

	vector< DMatch > good_matches;

	for (int i = 0; i < descriptors_2.rows; i++)
	{
		if (matches[i].distance <= match_dis)
		{
			good_matches.push_back(matches[i]);
		}
	}

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

	std::vector<Point2f> obj;

	for (unsigned int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(keypoints_scene[good_matches[i].queryIdx].pt);
	}
	//imshow("Good Matches & Object detection", img_goodmatch);

	cv::Rect brect = cv::boundingRect(cv::Mat(obj).reshape(2));
	cv::Size deltaSize(brect.width * 0.5f, brect.height * 0.5f);
	cv::Point offset(deltaSize.width / 2, deltaSize.height / 2);
	brect += deltaSize;
	brect -= offset;

	cv::rectangle(img_keypoints, brect.tl(), brect.br(), cv::Scalar(255, 0, 0), 5);

	//imshow("Box", img_keypoints);

	cv::GaussianBlur(image_scene(brect), image_scene(brect), Size(0, 0), 12);

	imshow("Final", image_scene);

}

int main(int argc, char** argv)
{
	int match_distance = 40;

	Filter(match_distance);
	
	waitKey(0);

	return 0;
}


