//练习1
#include<opencv2/opencv.hpp>
#include<iostream>
#include<math.h>
#include <vector>

using namespace cv;
using namespace std;

void myharris(Mat & src, float * hist);

int main()
{
    int bins = 1000000;
	Mat src = imread("E:\\9\\hogTemplate.jpg");
	Mat src1 = imread("E:\\9\\img1.jpg");
	Mat src2 = imread("E:\\9\\img2.jpg");

	
	float * ref_hist = new float[bins];//bins
	memset(ref_hist, 0, sizeof(float)*bins);

	float * ref_hist1 = new float[bins];//bins
	memset(ref_hist1, 0, sizeof(float)*bins);

	float * ref_hist2 = new float[bins];//bins
	memset(ref_hist2, 0, sizeof(float)*bins);

	myharris(src,ref_hist);
	myharris(src1, ref_hist1);
	myharris(src2, ref_hist2);


	float sum1 = 0;
	for (int i = 0; i < bins; i++) {
		sum1 += (ref_hist[i] - ref_hist1[i])*(ref_hist[i] - ref_hist1[i]);
	}
	sum1=sqrt(sum1);
	cout << sum1 << endl;

	float sum2 = 0;
	for (int i = 0; i < bins; i++) {
		sum2 += (ref_hist[i] - ref_hist2[i])*(ref_hist[i] - ref_hist2[i]);
	}
	sum2 = sqrt(sum2);
	cout  << sum2 << endl;

	if (sum1 <= sum2) {
		cout << "img1更与原图相似"<< endl;
	}
	if (sum1 > sum2) {
		cout << "img2更与原图相似" << endl;
	}

	delete[] ref_hist;
	delete[] ref_hist1;
	delete[] ref_hist2;
	waitKey();
	return 0;
}

void myharris(Mat & src, float * ref_hist) {
	Mat gray;
	cvtColor(src, gray, CV_BGR2GRAY);

	int cellSize = 16;
	float scale = 360 / 8;
	int nX = gray.cols / cellSize;
	int nY = gray.rows / cellSize;

	Mat gx, gy;
	Mat mag, angle;
	cv::Sobel(gray, gx, CV_32F, 1, 0, 1);
	cv::Sobel(gray, gy, CV_32F, 0, 1, 1);
	cartToPolar(gx, gy, mag, angle, true);

	int c = 0;
	//遍历所有
	for (int a = 0; ((a-1)*nY) <= gray.rows; a++) {
		for (int b = 0; ((b-1)*nX) <= gray.cols; b++) {

			//遍历其中一个cell
			for (int j = (a*nY); j <= (a*nY+ nY); j++) {
				uchar* mag_row_ptr = mag.ptr <uchar>(j);// 第j行的头指针
				uchar* angle_row_ptr = angle.ptr <uchar>(j);// 第j行的头指针
				for (int i = (b*nX); i <= (b*nX+ nX); i++) {
					// 访问位于 x,y 处的像素
					uchar* mag_ptr = &mag_row_ptr[i]; //  指向待访问的像素数据
					uchar* angle_ptr = &angle_row_ptr[i]; //  指向待访问的像素数据

					float data_mag = mag_ptr[0]; // data为I(x,y)第0个通道的值    	梯度
					float data_angle = angle_ptr[0]; // data为I(x,y)第0个通道的值   角度
					
					for (int i = 0; i <= 8; i++)
					{
						if (data_angle <= (i*scale))
						{
							int d = i + c;
							ref_hist[d] = ref_hist[d] + data_mag;//遍历像素
							//cout << "histgram" << ref_hist[d] << endl;
						}
					}			
				}	
			}

        c = c + 8;
		}
	}
}