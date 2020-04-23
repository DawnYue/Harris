#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
using namespace cv;
using namespace std;
//练习1

int main()
{
	Mat img_1 = imread("E:\\1.png");
	//初始化
	std::vector<KeyPoint> keypoints_1;
	Mat descriptors_1;
	Ptr<FeatureDetector> detector = ORB::create();
	Ptr<DescriptorExtractor> descriptor = ORB::create();
	//检测 Oriented FAST 角点位置
	detector->detect(img_1, keypoints_1);
	//根据角点位置计算 BRIEF 描述子
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	Mat outimg1;
	drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB特征点", outimg1);

 
/* 	//以灰度模式载入图像并显示
	Mat srcImage = imread("E:\\1.png", 0);*/
	Mat srcImage = imread("E:\\1.png");
	imshow("原始图", srcImage);
	//灰度图	
	Mat grayImage;
	cvtColor(srcImage, grayImage, CV_BGR2GRAY);

	//进行Harris角点检测找出角点
	Mat cornerStrength;
	cornerHarris(grayImage, cornerStrength, 2, 3, 0.04, BORDER_DEFAULT);
	imshow("角点检测后的图", cornerStrength);
	//对灰度图进行阈值操作，得到二值图并显示  
	Mat harrisCorner;
	threshold(cornerStrength, harrisCorner, 0.00001, 255, THRESH_BINARY);
	imshow("角点检测后的二值效果图", harrisCorner);

	Mat dstImage;//目标图
	Mat normImage;//归一化后的图
	Mat scaledImage;//线性变换后的八位无符号整型的图
	//初始化
	//置零当前需要显示的两幅图，即清除上一次调用此函数时他们的值
	dstImage = Mat::zeros(srcImage.size(), CV_32FC1);
	//进行角点检测
	cornerHarris(grayImage, dstImage, 2, 3, 0.04, BORDER_DEFAULT);
	// 归一化与转换
	normalize(dstImage, normImage, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
	convertScaleAbs(normImage, scaledImage);//将归一化后的图线性变换成8位无符号整型 
	// 将检测到的，且符合阈值条件的角点绘制出来
	for (int j = 0; j < normImage.rows; j++)
	{
		for (int i = 0; i < normImage.cols; i++)
		{
			if ((int)normImage.at<float>(j, i) > 0 + 80)
			{
				circle(srcImage, Point(i, j), 5, Scalar(10, 10, 255), 2, 8, 0);
				circle(scaledImage, Point(i, j), 5, Scalar(0, 10, 255), 2, 8, 0);
			}
		}
	}
	imshow(" ", srcImage);
	imshow(" s ", scaledImage);


	waitKey();
	return 0;
}


//https://github.com/ooooooops/HarrisCornersDetector/blob/master/harris.cpp