#include<opencv2/opencv.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
using namespace std;
using namespace cv;

Mat source_image;
Mat template_image;
Mat result;

const char* image_window = "Source Image";
const char* result_window = "Result window";

int match_method = 5;
int max_Trackbar = 5;

void MatchingMethod(int, void*);

int main(int, char** argv)
{
	//source_image = imread("images/nazi_germany_sucks/source_nazi_2.jpg");
	//template_image = imread("images/nazi_germany_sucks/template_nazi.jpg");

	source_image = imread("images/ussr_sucks/source_ussr_6.jpg");
	template_image = imread("images/ussr_sucks/template_ussr.jpg");

	namedWindow(image_window, WINDOW_AUTOSIZE);
	namedWindow(result_window, WINDOW_AUTOSIZE);

	const char* trackbar_label = "Method: \n 0: SQDIFF \n 1: SQDIFF NORMED \n 2: TM CCORR \n 3: TM CCORR NORMED \n 4: TM COEFF \n 5: TM COEFF NORMED";
	createTrackbar(trackbar_label, image_window, &match_method, max_Trackbar, MatchingMethod);

	MatchingMethod(0, 0);

	//imshow(image_window, source_image);
	//imshow(result_window, template_image);

	waitKey(0);
	return 0;
}
void MatchingMethod(int, void*)
{
	// Source image to display
	Mat img_display;
	source_image.copyTo(img_display);

	// Result matrix
	int result_cols = source_image.cols - template_image.cols + 1;
	int result_rows = source_image.rows - template_image.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	// Matching and Normalization
	matchTemplate(source_image, template_image, result, match_method);
	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

	// Best match from minMaxLoc
	double minVal; double maxVal; Point minLoc; Point maxLoc;
	Point matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());

	// For SQDIFF and SQDIFF_NORMED, the best matches are lower values. For all the other methods, the higher the better
	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
	{
		matchLoc = minLoc;
	}
	else
	{
		matchLoc = maxLoc;
	}

	//output
	rectangle(img_display, matchLoc, Point(matchLoc.x + template_image.cols, matchLoc.y + template_image.rows), Scalar::all(0), 2, 8, 0);
	rectangle(result, matchLoc, Point(matchLoc.x + template_image.cols, matchLoc.y + template_image.rows), Scalar::all(0), 2, 8, 0);

	imshow(image_window, img_display);
	imshow(result_window, result);

	return;
}