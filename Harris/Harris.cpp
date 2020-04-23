#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>
using namespace cv;
using namespace std;
//课前准备
/*
*description:
*	比较s1和s2，比较结果写入d
*/
void compareMat(Mat& s1, Mat& s2, Mat& d)
{
	if (d.empty())
	{
		d.create(s1.rows, s1.cols, s1.type());
	}

	for (int y = 0; y < s1.rows; y++)
	{
		for (int x = 0; x < s1.cols; x++)
		{
			if (s1.at<float>(y, x) == s2.at<float>(y, x))
			{
				d.at<float>(y, x) = s1.at<float>(y, x);
			}
			else
			{
				d.at<float>(y, x) = 0.0f;
			}
		}
	}
}


void Harris(vector<Point2i>& corners, Mat& I)
{

	if (!I.data)
	{
		return;
	}

	double GaussKernel[5][5] = {
		0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
		0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
		0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
		0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
		0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
	};

	Mat gray;
	cvtColor(I, gray, CV_BGR2GRAY);

	Mat IxIx(gray.rows, gray.cols, CV_8UC1);
	Mat IyIy(gray.rows, gray.cols, CV_8UC1);
	Mat IxIy(gray.rows, gray.cols, CV_8UC1);
	for (int y = 1; y <= I.rows - 2; y++)
	{
		//const uchar* grayRow = gray.ptr<uchar>(y);
		for (int x = 1; x <= I.cols - 2; x++)
		{
			//计算当前像素在两个方向的偏导数(即梯度)
			/*本应使用Harris卷积核进行卷积，Harris卷积核：
			[ 0  -1  0 ]
			[ -1  0  1 ]
			[ 0   1  0 ]
			此处分离计算比较简单，因此分离计算
			*/
			uchar ix = gray.at<uchar>(y + 1, x) - gray.at<uchar>(y - 1, x);
			uchar iy = gray.at<uchar>(y, x + 1) - gray.at<uchar>(y, x - 1);

			IxIx.at<uchar>(y, x) = ix * ix;
			IyIy.at<uchar>(y, x) = iy * iy;
			IxIy.at<uchar>(y, x) = ix * iy;
		}
	}

	//Harris响应矩阵
	Mat R(gray.rows, gray.cols, CV_32FC1);
	//最大响应值
	float maxResponse = 0.0f;

	for (int y = 0; y <= I.rows - 5; y++)
	{
		for (int x = 0; x <= I.cols - 5; x++)
		{
			uchar a = 0;  /*  自相似矩阵 M：                       */
			uchar b = 0;  /*  [ ∑g*a  ∑g*c ]   即  [ ∑g*IxIx ∑g*IxIy ]        */
			uchar c = 0;  /*  [ ∑g*c  ∑g*b ]       [ ∑g*IxIy ∑g*IyIy ]        */
			for (int m = 0; m < 5; m++)
			{
				for (int n = 0; n < 5; n++)
				{
					//分别对IxIx，IyIy，IxIy的5*5邻域计算高斯均值以求得自相似矩阵 M
					a += IxIx.at<uchar>(y + m, x + n) * GaussKernel[m][n];
					b += IyIy.at<uchar>(y + m, x + n) * GaussKernel[m][n];
					c += IxIy.at<uchar>(y + m, x + n) * GaussKernel[m][n];
				}
			}

			//计算当前像素的Harris响应值
			float t = (a * c - b * b) - 0.05f * (a + c) * (a + c);
			R.at<float>(y, x) = t;

			if (t > maxResponse)
			{
				maxResponse = t;
			}
		}
	}

	/*
	为了求得局部极大值，先对R进行膨胀，将膨胀结果与R比较，相同的就是局部极大值
	*/
	Mat dilated;
	dilate(R, dilated, Mat());
	Mat localMax;
	compareMat(R, dilated, localMax);

	//剔除部分局部极大值，这是因为如果不提出，局部极大值有可能很多，不是想要的结果
	Mat threshold2;
	threshold(localMax, threshold2, 0.5 * maxResponse, 255, THRESH_BINARY);
	//imshow("localMax", threshold2);

	for (int y = 0; y < threshold2.rows - 1; y++)
	{
		const float* row = threshold2.ptr<float>(y);
		for (int x = 0; x < threshold2.cols - 1; x++)
		{
			if (row[x])
			{
				corners.push_back(Point2i(x, y));
			}
		}
	}
}



int main()
{
	Mat m = imread("E:\\1.png");
	vector<Point2i> corners;
	Harris(corners, m);
	for (vector<Point2i>::iterator it = corners.begin(); it != corners.end(); it++)
	{
		circle(m, *it, 2, Scalar(255, 242, 0));
	}
	imshow("ok", m);

	Mat img_1 = imread("E:\\1.png");
	//-- 初始化
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