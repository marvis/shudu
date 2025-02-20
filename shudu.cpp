#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>
#include <string>
#include <fstream>
#include "sudoku.h"
#include "classes/feature.h"
#include "svm.h"

using namespace std;
using namespace cv;
#define DEBUG 1

string filename0, filename1;
string modelfile = "classes/trainfile_scale.model";
string rulefile = "classes/rules";
struct svm_model* model = 0;
vector<pair<double, double> > scale_params;
bool load_svm_rules(string rulefile)
{
	assert(scale_params.empty());
	ifstream ifs;
	ifs.open(rulefile.c_str());
	char name[256];
	ifs.getline(name, 256);
	ifs.getline(name, 256);
	int id;
	double min_val, max_val;
	while(ifs.good())
	{
		ifs >> id;
		ifs >> min_val;
		ifs >> max_val;
		scale_params.push_back(pair<double,double>(min_val, max_val));
	}
	ifs.close();
	return true;
}

void scale_svm_node(struct svm_node * x, vector<pair<double, double> > &scale_params, double scale_min_val, double scale_max_val)
{
	int nFeat = scale_params.size();
	for(int i = 0; i < nFeat; i++)
	{
		double min_val = scale_params[i].first;
		double max_val = scale_params[i].second;
		//x[i].index = i+1;
		x[i].value = (x[i].value - min_val)/(max_val - min_val) * 2.0 - 1.0;
		//cout<<" "<<x[i].index<<":"<<x[i].value;
	}
	//cout<<endl;
}

// the image width is 3*n
IplImage * cropImageToSquare3(IplImage * src, CvPoint tlPoint, CvPoint trPoint, CvPoint blPoint, CvPoint brPoint)
{
	double dx1 =  (trPoint.x + brPoint.x)/2.0 - (tlPoint.x + blPoint.x)/2.0;
	double dy1 =  (trPoint.y + brPoint.y)/2.0 - (tlPoint.y + blPoint.y)/2.0;
	double dx2 = (tlPoint.x + trPoint.x)/2.0 - (blPoint.x + brPoint.x)/2.0;
	double dy2 = (tlPoint.y + trPoint.y)/2.0 - (blPoint.y + brPoint.y)/2.0;
	int width = 300;//sqrt(dx1*dx1 + dy1*dy1) + 0.5;
	int height = 300;//sqrt(dx2*dx2 + dy2*dy2) + 0.5;
	//width = (width + height)/2.0;
	//width = width - width % 3;
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
	/*printf("transform matrix\n");  
    for(int i =0;i<3;i++)  
    {  
        printf("% .4f ",t.at<double>(0,i));  
        printf("% .4f ",t.at<double>(1,i));  
        printf("% .4f \n",t.at<double>(2,i));  
    }*/

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

int whichNum1(string file)
{
	string featfile = file + ".feat.txt";
	string featfile_scale = file + ".feat.scale.txt";
	string outfile = file + ".out.txt";
	string cmd1 = "classes/hogfeat " + file + " 1 > " + featfile;
	string cmd2 = "svm-scale -r classes/rules " + featfile + " > " + featfile_scale;
	string cmd3 = "svm-predict " + featfile_scale + " classes/trainfile_scale.model " + outfile + " > /dev/null";
	string allcmds = cmd1 + " && " + cmd2 + " && " + cmd3;
	system(allcmds.c_str());
	ifstream ifs;
	ifs.open(outfile);
	int c;
	ifs >> c;
	ifs.close();

	return c;
}

int whichNum2(IplImage * __image)
{
	IplImage * _image = cvCreateImage(cvGetSize(__image), IPL_DEPTH_8U, 3);//cvLoadImage("classes/test/1/7.png.box1.1.1.png",1);
	cvCvtColor(__image, _image, CV_GRAY2BGR);
	IplImage * image = cropImage(_image, 2, 2, _image->width - 4, _image->height-4);
	FeatureMap ** map = new (FeatureMap*);
	getFeature(image, 8, map);
	normalizehog(*map, 0.2);
	PCAFeature(*map);

	int nFeat = (*map)->numFeatures * (*map)->sizeX * (*map)->sizeY;
	float * feats = (*map)->map;
	struct svm_node * x = (struct svm_node*) malloc((nFeat+1) * sizeof(struct svm_node));
	for(int i = 0; i < nFeat; i++)
	{
		x[i].index = i+1;
		x[i].value = feats[i];
	}
	x[nFeat].index = -1;
	scale_svm_node(x, scale_params, -1.0, 1.0);
	double predict_label = svm_predict(model, x);
	delete map;
	delete [] (*map);
	cvReleaseImage(&image);
	cvReleaseImage(&_image);
	return (int)(predict_label);
}

bool isBackground(IplImage * numImg)
{
	// 去除边框的影响
	IplImage * tmpImg = cropImage(numImg, 15, 15, 70, 70);
	int width = tmpImg->width;
	int height = tmpImg->height;
	int count =  0;
	for(int j = 0; j < height; j++)
	{
		for(int i = 0; i < width; i++)
		{
			if(CV_IMAGE_ELEM(tmpImg, unsigned char, j, i) < 80)
			{
				count++;
			}
		}
	}
	cvReleaseImage(&tmpImg);
	if(count == 0) return true;
	else return false;
}
// boxId is used in the filename
vector<int> recogizeBoxNums(IplImage * box, int boxId)
{
	int swid = box->width/3;
	int shei = swid;
	vector<int> nums;
	for(int j = 0; j < 3; j++)
	{
		for(int i = 0; i < 3; i++)
		{
			IplImage * numImg = cropImage(box, i*swid, j*shei, swid, shei);
#if DEBUG
			ostringstream oss;
			oss <<filename0<<".box"<<boxId<<"."<<j<<"."<<i<<".png";
			cvSaveImage(oss.str().c_str(), numImg);
#endif
			if(isBackground(numImg))
				nums.push_back(0);
			else
			{
				//int c1 = whichNum1(oss.str());
				int c2 = whichNum2(numImg);
				//if(c1 != c2)
				//{
				//	cerr<<"c1 = "<<c1<<" c2 = "<<c2<<endl;
				//}
				nums.push_back(c2);
			}
			cvReleaseImage(&numImg);
		}
	}
	return nums;
}
bool splitNineBoxes(IplImage * src, IplImage * mask, int (&matrix)[9][9])
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
	vector<int> Mtlxs, Mtlys;
	vector<vector<int> > allnums;
	for(int n = 1; n <= 9; n++)
	{
		cout<<"n = "<<n<<endl;
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
		Mtlxs.push_back(Mtlx); Mtlys.push_back(Mtly);

		CvPoint tlPoint; tlPoint.x = Mtlx; tlPoint.y = Mtly;
		CvPoint trPoint; trPoint.x = Mtrx; trPoint.y = Mtry;
		CvPoint blPoint; blPoint.x = Mblx; blPoint.y = Mbly;
		CvPoint brPoint; brPoint.x = Mbrx; brPoint.y = Mbry;
		IplImage * box = cropImageToSquare3(src, tlPoint, trPoint, blPoint, brPoint);
#if DEBUG
		ostringstream oss;
		oss<<"box"<<n<<".png";
		//cvSaveImage(oss.str().c_str(), box);
#endif
		vector<int> boxnums = recogizeBoxNums(box, n);
		allnums.push_back(boxnums);
		cvReleaseImage(&box);

		cvCircle(drawImg, tlPoint, 2, CV_RGB(0xff, 0x0, 0x0), 2, CV_AA, 0);
		cvCircle(drawImg, trPoint, 2, CV_RGB(0x0, 0xff, 0x0), 2, CV_AA, 0);
		cvCircle(drawImg, blPoint, 2, CV_RGB(0x0, 0x0, 0xff), 2, CV_AA, 0);
		cvCircle(drawImg, brPoint, 2, CV_RGB(0xff, 0xff, 0xff), 2, CV_AA, 0);
	}
	vector<int> xorders = get_orders(Mtlxs);
	vector<int> yorders = get_orders(Mtlys);

	for(int n = 0; n < 9; n++)
	{
		int i = xorders[n]/3;
		int j = yorders[n]/3;
		for(int jj = 0; jj < 3; jj++)
		{
			for(int ii = 0; ii < 3; ii++)
			{
				int I = i * 3 + ii;
				int J = j * 3 + jj;
				matrix[J][I] = allnums[n][jj*3+ii];
			}
		}
	}
#if DEBUG
	cvSaveImage("test2.bin2.png", drawImg);
	cvReleaseImage(&drawImg);
#endif
	return true;
}
// src is grayscale image
bool findNineBoxes(IplImage * src, IplImage * mask, int (&matrix)[9][9])
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
			if(maskval > 0 && val > 80) 
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

	/*cout<<"Areas: "<<endl;
	for(int i = 0; i < colorNum; i++)
	{
		cout<<colorsum[i]<<" ";
	}
	cout<<endl;
	*/

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
	splitNineBoxes(src2, mask2, matrix);
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
	cvSmooth(src, smthImg, CV_BLUR, 7, 7);
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
	if((model = svm_load_model(modelfile.c_str())) == 0)
	{
		cerr<<"Can't open model file "<<modelfile<<endl;
		return 0;
	}
	if(!load_svm_rules(rulefile.c_str()))
	{
		cerr<<"Can't load svm rule file"<<rulefile<<endl;
		return 0;
	}
	IplImage * image0 = cvLoadImage(argv[1], 0); // load as gray image
	IplImage * image1 = cvLoadImage(argv[1], 1); // load as color image
	filename0 = argv[1];
	filename1 = filename0;
	if(!image0 || !image1)
	{
		printf( "No image data \n" );
		return -1;
	}

	IplImage * mask0 = cvCreateImage(cvGetSize(image0), IPL_DEPTH_8U, 1);
	//maximumConnectedComponent(image0, mask0, 40, 255, CV_THRESH_BINARY);
	xiaomiScreen(image0, mask0);
/*
	IplImage * mask1 = cvCreateImage(cvGetSize(image0), IPL_DEPTH_8U, 3);
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
*/
	int matrix[9][9];
	findNineBoxes(image0, mask0, matrix);
	for(int j = 0; j < 9; j++)
	{
		for(int i = 0; i < 9; i++)
		{
			cout<<matrix[j][i]<<" ";
		}
		cout<<endl;
	}

	cout<<endl;
	cout<<"=========== Solution ============"<<endl;
	if(sudoku(matrix))
	{
		print_matrix(matrix);
	}
	else
	{
		cout<<"Failed!"<<endl;
	}

	cvReleaseImage(&image0);
	cvReleaseImage(&image1);
	cvReleaseImage(&mask0);
	//cvReleaseImage(&mask1);
	svm_free_model_content(model);
	svm_free_and_destroy_model(&model);
	//svm_destroy_model(&model);
	return 0;
}
