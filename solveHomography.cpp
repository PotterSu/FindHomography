#include <iostream>

#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"

using namespace std;
using namespace cv;

static void gaussian_elimination(float *input, int n)
{
	float * A = input;
	int i = 0;
	int j = 0;
    //m = 8 rows, n = 9 cols
	int m = n-1;
	while (i < m && j < n)
    {
		// Find pivot in column j, starting in row i:
		int maxi = i;
		for(int k = i+1; k < m; k++)
        {
            //选取第j列最大的数，并记录行
			if(fabs(A[k * n + j]) > fabs(A[maxi * n + j]))
            {
				maxi = k;
			}
		}
		if (A[maxi * n + j] != 0)
        {
			//swap rows i and maxi, but do not change the value of i
			if(i != maxi)
				for(int k = 0; k < n; k++)
                {
					float aux = A[i * n + k];
					A[i * n + k] = A[maxi * n + k];
					A[maxi * n + k] = aux;
				}
			//Now A[i,j] will contain the old value of A[maxi,j].
			//divide each entry in row i by A[i,j]
            //将主行归一化
			float A_ij = A[i * n + j];
			for(int k = 0; k < n; k++)
            {
				A[i * n + k] /= A_ij;
			}
			//Now A[i,j] will have the value 1.
            //主行*A[u,j]，再用A[u,j]-该数即可消除
			for(int u = i+1; u< m; u++)
            {
				//subtract A[u,j] * row i from row u
				float A_uj = A[u * n + j];
				for(int k = 0; k <n; k++)
                {
					A[u * n + k] -= A_uj * A[i * n + k];
				}
				//Now A[u,j] will be 0, since A[u,j] - A[i,j] * A[u,j] = A[u,j] - 1 * A[u,j] = 0.
			}
			i++;
		}
		j++;
	}

	//back substitution
    //最后一位不用管，其他各行用最后一个数-前面各列数*已求的未知数
	for(int i = m-2; i >= 0; i--)
    {
		for(int j = i+1; j < n-1; j++)
        {
			A[i * n + m] -= A[i * n + j] * A[j * n + m];
		}
	}

}

static void findHomography(std::vector<cv::Point2f> src, std::vector<cv::Point2f> dst, float homography[9])
{
	// create the equation system to be solved
	// src and dst must implement [] operator for point access
	//
	// from: Multiple View Geometry in Computer Vision 2ed
	//       Hartley R. and Zisserman A.
	//
	// x' = xH
	// where H is the homography: a 3 by 3 matrix
	// that transformed to inhomogeneous coordinates for each point
	// gives the following equations for each point:
	//
	// x' * (h31*x + h32*y + h33) = h11*x + h12*y + h13
	// y' * (h31*x + h32*y + h33) = h21*x + h22*y + h23
	//
	// as the homography is scale independent we can let h33 be 1 (indeed any of the terms)
	// so for 4 points we have 8 equations for 8 terms to solve: h11 - h32
	// after ordering the terms it gives the following matrix
	// that can be solved with gaussian elimination:

	float P[8][9]=
    {
		{-src[0].x, -src[0].y, -1,   0,   0,  0, src[0].x * dst[0].x, src[0].y * dst[0].x, -dst[0].x }, // h11
		{  0,   0,  0, -src[0].x, -src[0].y, -1, src[0].x * dst[0].y, src[0].y * dst[0].y, -dst[0].y }, // h12

		{-src[1].x, -src[1].y, -1,   0,   0,  0, src[1].x * dst[1].x, src[1].y * dst[1].x, -dst[1].x }, // h13
		{  0,   0,  0, -src[1].x, -src[1].y, -1, src[1].x * dst[1].y, src[1].y * dst[1].y, -dst[1].y }, // h21

		{-src[2].x, -src[2].y, -1,   0,   0,  0, src[2].x * dst[2].x, src[2].y * dst[2].x, -dst[2].x }, // h22
		{  0,   0,  0, -src[2].x, -src[2].y, -1, src[2].x * dst[2].y, src[2].y * dst[2].y, -dst[2].y }, // h23

		{-src[3].x, -src[3].y, -1,   0,   0,  0, src[3].x * dst[3].x, src[3].y * dst[3].x, -dst[3].x }, // h31
		{  0,   0,  0, -src[3].x, -src[3].y, -1, src[3].x * dst[3].y, src[3].y * dst[3].y, -dst[3].y }, // h32
	};

	gaussian_elimination(&P[0][0], 9);

	// gaussian elimination gives the results of the equation system
	// in the last column of the original matrix.
	// opengl needs the transposed 4x4 matrix:
    float aux_H[]={ P[0][8], P[1][8], P[2][8],	// h11  h21 0 h31
                    P[3][8], P[4][8], P[5][8],	// h12  h22 0 h32
                    P[6][8], P[7][8], 1};		// h13  h23 0 h33

	for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            homography[i * 3 + j] = aux_H[i * 3 + j];
        }
    }
}

int main()
{
    //原始四点坐标
    cv::Point2f srcPoint1(0.0f, 0.0f);
    cv::Point2f srcPoint2(0.0f, 1.0f);
    cv::Point2f srcPoint3(1.0f, 0.0f);
    cv::Point2f srcPoint4(1.0f, 1.0f);
    //目标四点坐标
    cv::Point2f dstPoint1(0.0f, 0.0f);
    cv::Point2f dstPoint2(0.0f, 1.0f);
    cv::Point2f dstPoint3(1.0f, 0.0f);
    cv::Point2f dstPoint4(2.0f, 2.0f);

    std::vector<cv::Point2f> src;
    src.push_back(srcPoint1);
    src.push_back(srcPoint2);
    src.push_back(srcPoint3);
    src.push_back(srcPoint4);
    std::vector<cv::Point2f> dst;
    dst.push_back(dstPoint1);
    dst.push_back(dstPoint2);
    dst.push_back(dstPoint3);
    dst.push_back(dstPoint4);

    std::cout << "----------------------------------" << std::endl;
    std::cout << "FindHomography" << std::endl;
    float homography[9];
    findHomography(src, dst, homography);
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout << homography[i * 3 + j] << setw(16) << " ";
        }
        std::cout << std::endl;
    }

    Mat cvHomography = cv::findHomography(src, dst);
    std::cout << "----------------------------------" << std::endl;
    std::cout << "OpenCV FindHomography" << std::endl;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            std::cout << cvHomography.at<double>(i, j) << setw(16) << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
