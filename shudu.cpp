#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
#define DEBUG 1

// the image width is 3*n
IplImage * cropImageToSquare3(IplImage * src, CvPoint tlPoint, CvPoint trPoint, CvPoint blPoint, CvPoint brPoint)
{
	double dx1 =  (trPoint.x + brPoint.x)/2.0 - (tlPoint.x + blPoint.x)/2.0;
	double dy1 =  (trPoint.y + brPoint.y)/2.0 - (tlPoint.y + blPoint.y)/2.0;
	double dx2 = (tlPoint.x + trPoint.x)/2.0 - (blPoint.x + brPoint.x)/2.0;
	double dy2 = (tlPoint.y + trPoint.y)/2.0 - (blPoint.y + brPoint.y)/2.0;
	int width = sqrt(dx1*dx1 + dy1*dy1) + 0.5;
	int height = sqrt(dx2*dx2 + dy2*dy2) + 0.5;
	width = (width + height)/2.0;
	width = width - width % 3;
	height = width;
	IplImage * dst = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, src->nChannels);
	int nchannels = src->nChannels;
	Point2f srcPoints[4];
	Point2f dstPoints[4];
	
	srcPoints[0].x = 0;           dstPoints[0].x = tlPoint.x;
	srcPoints[0].y = 0;           dstPoints[0].y = tlPoint.y;

	srcPoints[1].x = width -1;    dstPoints[1].x = trPoint.x;
	srcPoints[1].y = 0;           dstPoints[1].y = trPoint.y;

	srcPoints[2].x = 0;           dstPoints[2].x = blPoint.x;
	srcPoints[2].y = height - 1;  dstPoints[2].y = blPoint.y;

	srcPoints[3].x = width - 1;   dstPoints[3].x = brPoint.x;
	srcPoints[3].y = height - 1;  dstPoints[3].y = brPoint.y;

	Mat t = getPerspectiveTransform(srcPoints,dstPoints);
	printf("transform matrix\n");  
    for(int i =0;i<3;i++)  
    {  
        printf("% .4f ",t.at<double>(0,i));  
        printf("% .4f ",t.at<double>(1,i));  
        printf("% .4f \n",t.at<double>(2,i));  
    }

	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			Mat sample = (Mat_<double>(3,1)<<i,j,1);
			Mat r = t*sample;
			double s = r.at<double>(2,0);
			int x = round(r.at<double>(0,0)/s);
			int y = round(r.at<double>(1,0)/s);

			for(int c = 0; c < nchannels; c++)
			{
				CV_IMAGE_ELEM(dst, unsigned char, j, nchannels*i + c) = CV_IMAGE_ELEM(src, unsigned char, y, nchannels*x+c);
			}
		}
	}
	return dst;
}

IplImage * cropImage(IplImage * src, int x, int y, int width, int height)
{
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}

typedef std::pair<int,int> mypair;
bool comparator( const mypair& l, const mypair& r)
{
	return l.first < r.first; 
}

// e.g.: data = {2, 5, 4, 1}
// indices = {3, 0, 2, 1}
vector<int> sort_indices(vector<int> data)
{
	vector<mypair> dataPair;
	for(int i = 0; i < data.size(); i++)
	{
		mypair p(data[i], i);
		dataPair.push_back(p);
	}
	sort(dataPair, comparator);
	vector<int> indices;
	for(int i = 0; i < data.size(); i++)
	{
		mypair p = dataPair[i];
		indices.push_back(p.second);
	}
	return indices;
}

// e.g.: data = {2, 5, 4, 1}
// orders = {1, 3, 2, 0}
vector<int> get_orders(vector<int> data)
{
	vector<mypair> dataPair;
	for(int i = 0; i < data.size(); i++)
	{
		mypair p(data[i], i);
		dataPair.push_back(p);
	}
	sort(dataPair, comparator);
	vector<int> orders(data.size());
	for(int i = 0; i < data.size(); i++)
	{
		mypair p = dataPair[i];
		orders[p.second] = i;
	}
	return orders;
}

// src is a mask image, 0 should be backgound , forground is non-zero points
bool boundaryPoints(IplImage * src, vector<CvPoint> &leftPoints, vector<CvPoint> &rightPoints, vector<CvPoint> & topPoints, vector<CvPoint> & bottomPoints)
{
	if(!src || src->depth != IPL_DEPTH_8U || src->nChannels != 1)
	{
		cerr<<"Error: bottomPoints - src should be non-empty, IPL_DEPTH_8U, and 1 channel"<<endl;
		return false;
	}
	if(!leftPoints.empty() || !rightPoints.empty() || !topPoints.empty() || !bottomPoints.empty())
	{
		cerr<<"Warning: boundaryPoints - leftPoints, rightPoints, topPoints, bottomPoints is not empty"<<endl;
	}
	int width = src->width;
	int height = src->height;
	int widthStep = src->widthStep;
	int i , j;
	CvPoint p;
	bool isok;

	// leftPoints
	for(j = 0; j < height; j++)
	{
		isok = false;
		for(i = 0; i < width; i++)
		{
			if(CV_IMAGE_ELEM(src, unsigned char, j, i) > 0)
			{
				isok = true;
				break;
			}
		}
		p.x = i;
		p.y = j;
		leftPoints.push_back(p);
	}
	// rightPoints
	for(j = 0; j < height; j++)
	{
		isok = false;
		for(i = width-1; i >= 0; i--)
		{
			if(CV_IMAGE_ELEM(src, unsigned char, j, i) > 0)
			{
				isok = true;
				break;
			}
		}
		p.x = i;
		p.y = j;
		rightPoints.push_back(p);
	}
	// topPoints
	for(i = 0; i < width; i++)
	{
		isok = false;
		for(j = 0; j < height; j++)
		{
			if(CV_IMAGE_ELEM(src, unsigned char, j, i) > 0)
			{
				isok = true;
				break;
			}
		}
		p.x = i;
		p.y = j;
		topPoints.push_back(p);
	}
	// bottomPoints
	for(i = 0; i < width; i++)
	{
		isok = false;
		for(j = height-1; j >= 0; j--)
		{
			if(CV_IMAGE_ELEM(src, unsigned char, j, i) > 0)
			{
				isok = true;
				break;
			}
		}
		p.x = i;
		p.y = j;
		bottomPoints.push_back(p);
	}
	return true;
}

CvRect getMaskBounding(IplImage * mask)
{
	CvRect rect;
	if(!mask || mask->depth != IPL_DEPTH_8U || mask->nChannels != 1)
	{
		cerr<<"Error: getMaskBounding - mask should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return rect;
	}

	int width = mask->width;
	int height = mask->height;
	int min_i = width - 1, max_i = 0;
	int min_j = height - 1, max_j = 0;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(CV_IMAGE_ELEM(mask, unsigned char, j, i) > 0)
			{
				min_i = (i < min_i) ? i : min_i;
				max_i = (i > max_i) ? i : max_i;
				min_j = (j < min_j) ? j : min_j;
				max_j = (j > max_j) ? j : max_j;
			}
		}
	}
	rect.x = min_i;
	rect.y = min_j;
	rect.width = max_i - min_i + 1;
	rect.height = max_j - min_j + 1;
	return rect;
}

CvRect getMaskBounding2(IplImage * mask, int maskval)
{
	CvRect rect;
	if(!mask || mask->depth != IPL_DEPTH_8U || mask->nChannels != 1)
	{
		cerr<<"Error: getMaskBounding - mask should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return rect;
	}

	int width = mask->width;
	int height = mask->height;
	int min_i = width - 1, max_i = 0;
	int min_j = height - 1, max_j = 0;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(CV_IMAGE_ELEM(mask, unsigned char, j, i) == maskval)
			{
				min_i = (i < min_i) ? i : min_i;
				max_i = (i > max_i) ? i : max_i;
				min_j = (j < min_j) ? j : min_j;
				max_j = (j > max_j) ? j : max_j;
			}
		}
	}
	rect.x = min_i;
	rect.y = min_j;
	rect.width = max_i - min_i + 1;
	rect.height = max_j - min_j + 1;
	return rect;
}

bool recogizeBoxNums(IplImage * box, int boxId)
{
	int swid = box->width/3;
	int shei = swid;
	for(int j = 0; j < 3; j++)
	{
		for(int i = 0; i < 3; i++)
		{
			IplImage * numImg = cropImage(box, i*swid, j*shei, swid, shei);
#if DEBUG
			ostringstream oss;
			oss <<"box"<<boxId<<"."<<j<<"."<<i<<".png";
			cvSaveImage(oss.str().c_str(), numImg);
#endif
			cvReleaseImage(&numImg);
		}
	}
	return true;
}
bool splitNineBoxes(IplImage * src, IplImage * mask)
{
	if(!src || src->depth != IPL_DEPTH_8U || src->nChannels != 1)
	{
		cerr<<"Error: splitNineBoxes - src should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return 0;
	}
	if(!mask || mask->depth != IPL_DEPTH_8U || mask->nChannels != 1)
	{
		cerr<<"Error: splitNineBoxes - mask should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return 0;
	}
	if(mask->width != src->width || mask->height != src->height)
	{
		cerr<<"Error: splitNineBoxes - mask should be the same size of src"<<endl;
	}
	int width = src->width;
	int height = src->height;
	IplImage * drawImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	cvCvtColor(src , drawImg, CV_GRAY2BGR);
	for(int n = 1; n <= 9; n++)
	{
		CvRect rect = getMaskBounding2(mask, n);
		double mid_w = rect.x + rect.width/2.0;
		double mid_h = rect.y + rect.height/2.0;
		double max_tl_dist = 0;
		double max_tr_dist = 0;
		double max_bl_dist = 0;
		double max_br_dist = 0;
		int Mtlx, Mtly;
		int Mtrx, Mtry;
		int Mblx, Mbly;
		int Mbrx, Mbry;

		for(int h = 0; h < height; h++)
		{
			for(int w = 0; w < width; w++)
			{
				if(CV_IMAGE_ELEM(mask, unsigned char, h, w) == n)
				{
					double dist = (w-mid_w)*(w-mid_w) + (h-mid_h)*(h-mid_h);
					if(w < mid_w && h < mid_h && dist > max_tl_dist) {Mtlx = w; Mtly = h; max_tl_dist = dist;}
					if(w >= mid_w && h < mid_h && dist > max_tr_dist) {Mtrx = w; Mtry = h; max_tr_dist = dist;}
					if(w < mid_w && h >= mid_h && dist > max_bl_dist) {Mblx = w; Mbly = h; max_bl_dist = dist;}
					if(w >= mid_w && h >= mid_h && dist > max_br_dist) {Mbrx = w; Mbry = h; max_br_dist = dist;}
				}
			}
		}
		CvPoint tlPoint; tlPoint.x = Mtlx; tlPoint.y = Mtly;
		CvPoint trPoint; trPoint.x = Mtrx; trPoint.y = Mtry;
		CvPoint blPoint; blPoint.x = Mblx; blPoint.y = Mbly;
		CvPoint brPoint; brPoint.x = Mbrx; brPoint.y = Mbry;
		IplImage * box = cropImageToSquare3(src, tlPoint, trPoint, blPoint, brPoint);
#if DEBUG
		ostringstream oss;
		oss<<"box"<<n<<".png";
		cvSaveImage(oss.str().c_str(), box);
#endif
		recogizeBoxNums(box, n);
		cvReleaseImage(&box);

		cvCircle(drawImg, tlPoint, 2, CV_RGB(0xff, 0x0, 0x0), 2, CV_AA, 0);
		cvCircle(drawImg, trPoint, 2, CV_RGB(0x0, 0xff, 0x0), 2, CV_AA, 0);
		cvCircle(drawImg, blPoint, 2, CV_RGB(0x0, 0x0, 0xff), 2, CV_AA, 0);
		cvCircle(drawImg, brPoint, 2, CV_RGB(0xff, 0xff, 0xff), 2, CV_AA, 0);
	}
#if DEBUG
	cvSaveImage("test2.bin2.png", drawImg);
	cvReleaseImage(&drawImg);
#endif
	return true;
}
// src is grayscale image
bool findNineBoxes(IplImage * src, IplImage * mask)
{
	if(!src || src->depth != IPL_DEPTH_8U || src->nChannels != 1)
	{
		cerr<<"Error: findNineBoxes - src should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return 0;
	}
	if(!mask || mask->depth != IPL_DEPTH_8U || mask->nChannels != 1)
	{
		cerr<<"Error: findNineBoxes - mask should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return 0;
	}
	if(mask->width != src->width || mask->height != src->height)
	{
		cerr<<"Error: findNineBoxes - mask should be the same size of src"<<endl;
	}
	CvRect bounding = getMaskBounding(mask);
	IplImage * src2 = cropImage(src, bounding.x, bounding.y, bounding.width, bounding.height);
	IplImage * mask2 = cropImage(mask, bounding.x, bounding.y, bounding.width, bounding.height);

	int width = src2->width;
	int height = src2->height;
	IplImage * binImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(src2, unsigned char, j, i);
			double maskval = CV_IMAGE_ELEM(mask2, unsigned char, j, i);
			if(maskval > 0 && val > 60) 
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
			}
			else 
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
			}
		}
	}

#if DEBUG
	cvSaveImage("test2.bin1.png", binImg);
#endif

	int max_color = 256*256*256 - 1;
	int color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(binImg, cvPoint(w,h), CV_RGB(hig, mid, low));
					color++;
				}
			}
			else
			{
				cerr<<"Error: findNineBoxes - too many connected areas. "<<endl;
				return 0;
			}
		}
	}
	int colorNum = color;

	vector<int>colorsum(colorNum, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= colorNum)
			{
				cerr<<"Error: findNineBoxes - invalid rgb"<<endl;
				return 0;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}
	vector<int> colorsum_orders = get_orders(colorsum);

	cout<<"Areas: "<<endl;
	for(int i = 0; i < colorNum; i++)
	{
		cout<<colorsum[i]<<" ";
	}
	cout<<endl;

	IplImage * indexImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			int order = colorNum - 1 - colorsum_orders[val];
			if(val > 0 && order < 9)
			{
				int rr = val * 89 % 256;
				int gg = val * 191 % 256;
				int bb = val * 311 % 256;
				CV_IMAGE_ELEM(indexImage, unsigned char, h, 3*w) = rr;  
				CV_IMAGE_ELEM(indexImage, unsigned char, h, 3*w+1) = gg;  
				CV_IMAGE_ELEM(indexImage, unsigned char, h, 3*w+2) = bb;  
				CV_IMAGE_ELEM(mask2, unsigned char, h, w) = order+1;
			}
			else
			{  
				CV_IMAGE_ELEM(indexImage, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(indexImage, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(indexImage, unsigned char, h, 3*w+2) = 0;

				CV_IMAGE_ELEM(mask2, unsigned char, h, w) = 0;
			}
		}
	}
#if DEBUG
	cvSaveImage("test2.bin1.png", indexImage);
#endif
	splitNineBoxes(src2, mask2);
	cvReleaseImage(&src2);
	cvReleaseImage(&mask2);
	cvReleaseImage(&binImg);
	cvReleaseImage(&indexImage);
	return true;
}

// src is grayscale image
// dst is retured mask
bool xiaomiScreen(IplImage * src, IplImage * dst)
{
	if(!src || src->depth != IPL_DEPTH_8U || src->nChannels != 1)
	{
		cerr<<"Error: xiaomiScreen - src should be non-empty, IPL_DEPTH_8U and 1 channel"<<endl;
		return 0;
	}
	int width = src->width;
	int height = src->height;
// find maximum component first time
	IplImage * smthImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
	cvSmooth(src, smthImg, CV_BLUR, 5, 5);
	IplImage * binImg = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3);
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(smthImg, unsigned char, j, i);
			if(val > 30) 
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
			}
			else 
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
			}
		}
	}

#if DEBUG
	cvSaveImage("test.bin1.png", binImg);
#endif

	int max_color = 256*256*256 - 1;
	int color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(binImg, cvPoint(w,h), CV_RGB(hig, mid, low));
					color++;
				}
			}
			else
			{
				cerr<<"Error: xiaomiScreen - too many connected areas. "<<endl;
				return 0;
			}
		}
	}
	int colorNum = color;

	vector<int>colorsum(colorNum, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= colorNum)
			{
				cerr<<"Error: xiaomiScreen - invalid rgb"<<endl;
				return 0;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}

	int maxcolorLabel = max_element(colorsum.begin(), colorsum.end()) - colorsum.begin();

	int min_w = width-1, max_w = 0;
	int min_h = height-1, max_h = 0;
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val == maxcolorLabel)
			{
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 255;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 255;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 255;  
				min_w = (w < min_w) ? w : min_w;
				max_w = (w > max_w) ? w : max_w;
				min_h = (h < min_h) ? h : min_h;
				max_h = (h > max_h) ? h : max_h;
			}
			else
			{  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1) = 0;  
				CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2) = 0;  
			}
		}
	}

#if DEBUG
	cvSaveImage("test.bin2.png", binImg);
#endif
/*
	IplConvKernel *element1 = cvCreateStructuringElementEx(10, 10, 0, 0, CV_SHAPE_ELLIPSE);
	cvMorphologyEx(binImg, binImg, NULL, element1, CV_MOP_CLOSE);//关运算，填充(去除)内部的细线
#if DEBUG
	cvSaveImage("test.bin3.png", binImg);
#endif

	IplConvKernel *element2 = cvCreateStructuringElementEx(10, 10, 0, 0, CV_SHAPE_ELLIPSE);
	cvMorphologyEx(binImg, binImg, NULL, element2, CV_MOP_OPEN);//关运算，填充(去除)内部的细线
#if DEBUG
	cvSaveImage("test.bin4.png", binImg);
#endif
*/
	// find maximum component second time
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(i < min_w || i > max_w || j < min_h || j > max_h)
			{
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
			}
			else
			{
				unsigned char b = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i);
				unsigned char g = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1);
				unsigned char r = CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2);

				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255 - b;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255 - g;
				CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255 - r;
			}
		}
	}

#if DEBUG
	cvSaveImage("test.bin5.png", binImg);
#endif

	max_color = 256*256*256 - 1;
	color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(binImg, cvPoint(w,h), CV_RGB(hig, mid, low));
					color++;
				}
			}
			else
			{
				cerr<<"Error: xiaomiScreen - too many connected areas. "<<endl;
				return 0;
			}
		}
	}
	colorNum = color;

	//colorsum.resize(colorNum, 0);
	vector<int> colorsum2(colorNum, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= colorNum)
			{
				cerr<<"Error: xiaomiScreen - invalid rgb"<<endl;
				return 0;
			}
			if(rgb > 0)
			{
				colorsum2[rgb]++; //统计每种颜色的数量
			}
		}
	}

	maxcolorLabel = max_element(colorsum2.begin(), colorsum2.end()) - colorsum2.begin();

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(w < min_w || w > max_w || h < min_h || h > max_h)
			{
				CV_IMAGE_ELEM(dst, unsigned char, h, w) = 0;
			}
			else
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int val = r*256*256 + g*256 + b;
				if(val == maxcolorLabel)
					CV_IMAGE_ELEM(dst, unsigned char, h, w) = 255;
				else
					CV_IMAGE_ELEM(dst, unsigned char, h, w) = 0;
			}
		}
	}

#if DEBUG
	cvSaveImage("test.bin6.png", dst);
#endif
	cvReleaseImage(&smthImg);
	cvReleaseImage(&binImg);
	return true;
}
// return the maximum connected component of src into dst
bool maximumConnectedComponent(IplImage * src, IplImage * dst, double threshold, int max_value=255, int threshold_type = CV_THRESH_BINARY)
{
	if(!src || !dst || src->depth != IPL_DEPTH_8U ||dst->depth != IPL_DEPTH_8U)
	{
		cerr<<"Error: maximumConnectedComponent - src and dst should be non empty with type IPL_DEPTH_8U"<<endl;
		return false;
	}
	if(src->width != dst->width || src->height != dst->height)
	{
		cerr<<"Error: maximumConnectedComponent - src and dst should be the same size"<<endl;
		return false;
	}
	int width = src->width;
	int height = src->height;
	IplImage * binImg = cvCreateImage(cvGetSize(src), IPL_DEPTH_8U, 3); // 由于floodfill不支持16位数据, 所以用3通道
	
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			double val = CV_IMAGE_ELEM(src, unsigned char, j, i);
			if(threshold_type == CV_THRESH_BINARY)
			{
				if(val > threshold)
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
				}
				else
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
				}
			}
			else if(threshold_type == CV_THRESH_BINARY_INV)
			{
				if(val < threshold)
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 255;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 255;
				}
				else
				{
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+1) = 0;
					CV_IMAGE_ELEM(binImg, unsigned char, j, 3*i+2) = 0;
				}
			}
		}
	}

	int max_color = 256*256*256 - 1;
	int color = 1;

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			if(color < max_color)
			{
				int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
				int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
				int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
				int rgb = r*256*256 + g*256 + b;
				if(rgb==max_color)
				{
					unsigned char low = color % 256;
					unsigned char mid = (color / 256) % 256;
					unsigned char hig = color / (256*256);
					cvFloodFill(binImg, cvPoint(w,h), CV_RGB(hig, mid, low));
					color++;
				}
			}
			else
			{
				cerr<<"Error: maximumConnectedComponent - too many connected areas. "<<endl;
				return false;
			}
		}
	}
	int colorNum = color;

	vector<int>colorsum(colorNum, 0);
	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int rgb = r*256*256 + g*256 + b;
			if(rgb >= colorNum)
			{
				cerr<<"Error: maximumConnectedComponent - invalid rgb"<<endl;
				return false;
			}
			if(rgb > 0)
			{
				colorsum[rgb]++; //统计每种颜色的数量
			}
		}
	}

	int maxcolorLabel = max_element(colorsum.begin(), colorsum.end()) - colorsum.begin();

	for(int h = 0; h < height; h++)
	{
		for(int w = 0; w < width; w++)
		{
			int b = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w);
			int g = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+1);
			int r = CV_IMAGE_ELEM(binImg, unsigned char, h, 3*w+2);
			int val = r*256*256 + g*256 + b;
			if(val == maxcolorLabel)
				CV_IMAGE_ELEM(dst, unsigned char, h, w) = max_value;  
			else
				CV_IMAGE_ELEM(dst, unsigned char, h, w) = 0;  
		}
	}
	cvReleaseImage(&binImg);
	return true;
}
int main(int argc, char ** argv)
{
	if(argc != 2)
	{
		printf("No input image\n");
		return -1;
	}
	IplImage * image0 = cvLoadImage(argv[1], 0); // load as gray image
	IplImage * image1 = cvLoadImage(argv[1], 1); // load as color image
	string filename0 = argv[1];
	string filename1 = filename0;
	if(!image0 || !image1)
	{
		printf( "No image data \n" );
		return -1;
	}

	IplImage * mask0 = cvCreateImage(cvGetSize(image0), IPL_DEPTH_8U, 1);
	IplImage * mask1 = cvCreateImage(cvGetSize(image0), IPL_DEPTH_8U, 3);
	//maximumConnectedComponent(image0, mask0, 40, 255, CV_THRESH_BINARY);
	xiaomiScreen(image0, mask0);
	cvCvtColor(mask0, mask1, CV_GRAY2BGR);

	vector<CvPoint> leftPoints, rightPoints, topPoints, bottomPoints;
	boundaryPoints(mask0, leftPoints, rightPoints, topPoints, bottomPoints);
	for(int i = 0; i < leftPoints.size(); i++)
	{
		CvPoint p = leftPoints[i];
		cvCircle(mask1, p , 2, CV_RGB( 0xff, 0x0, 0x0 ), 2, CV_AA, 0);  //画圆函数  
		cvCircle(image1, p , 2, CV_RGB( 0xff, 0x0, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	}
	for(int i = 0; i < rightPoints.size(); i++)
	{
		CvPoint p = rightPoints[i];
		cvCircle(mask1, p , 2, CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
		cvCircle(image1, p , 2, CV_RGB( 0x0, 0xff, 0x0 ), 2, CV_AA, 0);  //画圆函数  
	}
	for(int i = 0; i < topPoints.size(); i++)
	{
		CvPoint p = topPoints[i];
		cvCircle(mask1, p , 2, CV_RGB( 0x0, 0x0, 0xff ), 2, CV_AA, 0);  //画圆函数  
		cvCircle(image1, p , 2, CV_RGB( 0x0, 0x0, 0xff ), 2, CV_AA, 0);  //画圆函数  
	}
	for(int i = 0; i < bottomPoints.size(); i++)
	{
		CvPoint p = bottomPoints[i];
		cvCircle(mask1, p , 2, CV_RGB( 0xff, 0x0, 0xff ), 2, CV_AA, 0);  //画圆函数  
		cvCircle(image1, p , 2, CV_RGB( 0xff, 0x0, 0xff ), 2, CV_AA, 0);  //画圆函数  
	}

	//cvNamedWindow("mask", 0);
	//cvShowImage("mask", mask);
	//cvWaitKey(0);
	filename1 = filename0 + ".mask.png";
	cvSaveImage(filename1.c_str(), mask0);
	
	filename1 = filename0 + ".boundary1.png";
	cvSaveImage(filename1.c_str(), mask1);

	filename1 = filename0 + ".boundary2.png";
	cvSaveImage(filename1.c_str(), image1);

	findNineBoxes(image0, mask0);

	cvReleaseImage(&image0);
	cvReleaseImage(&image1);
	cvReleaseImage(&mask0);
	cvReleaseImage(&mask1);
	return 0;
}
