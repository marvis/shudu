#include <opencv/cv.h>
#include <opencv/highgui.h>

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <cmath>
#include "feature.h"

using namespace cv;
using namespace std;

IplImage * cropImage(IplImage * src, int x, int y, int width, int height)
{
	cvSetImageROI(src, cvRect(x, y, width , height));
	IplImage * dst = cvCreateImage(cvSize(width,height), IPL_DEPTH_8U , src->nChannels);
	cvCopy(src, dst, 0);
    cvResetImageROI(src);
	return dst;
}

int main(int argc, char** argv)
{
	if(argc < 2)
	{
		cerr<<"Error: no input image!"<<endl;
		return -1;
	}
	IplImage * _image = cvLoadImage(argv[1], 1);
	IplImage * image = cropImage(_image, 2, 2, _image->width - 4, _image->height-4);

	//cout<<"ndim = "<<image->nChannels<<endl;
	FeatureMap ** map = new (FeatureMap*);
	getFeature(image, 8, map);
	normalizehog(*map, 0.2);
	PCAFeature(*map);

	int nFeat = (*map)->numFeatures * (*map)->sizeX * (*map)->sizeY;
	//cout<<"nFeat = "<<nFeat<<endl;
	float * _map = (*map)->map;
	vector<float> descriptors(_map, _map+nFeat);
	//cout<<"des.size() = "<<descriptors.size()<<endl;

	if(argc == 3)
		printf("%s", argv[2]);

	for(int j = 0; j < nFeat; j++)
	{
		printf(" %d:%f", j+1, descriptors[j]); 
	}
	printf("\n");
	delete map;
	delete [] (*map);

	cvReleaseImage(&_image);
	cvReleaseImage(&image);
	return 0;
}
