#include <iostream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cassert>


const double OVERSAMPLE = 1.4;
const double kernel_width=4.5;
const double pie=3.14159265258979323846264338327950288;
const int image_sz=141;

double calc_w(double u,double k_w){
	if(0.0==u)return 1.;
	if(fabs(u)>(k_w/2.))return 0.;
	double piu=pie*u;
	return (0.54347826 + 0.45652174*cos(2.0*piu/k_w))*(sin(piu/OVERSAMPLE)/(piu/OVERSAMPLE));
}

double calc_inv(double x,double k_w){
	double invf=1.;
	double tpx=2.*pie*x;
	for(int m=1;m<=(int)floor(k_w/2.);m++){
		invf+=2.*calc_w((double)m,k_w)*cos(tpx*((double)m));
	}
	return 1./invf;
}


void calc_deapod_window(const int gridSizeX,
						const int gridSizeY,
						const int imageSizeX,
						const int imageSizeY,
						const int resolutionX,
						const int resolutionY,
						float *post_window,
						float *post_window_crop,
						float *post_ones){
							//assume all pointers have already been allocated to size

	const int halfImageX = imageSizeX>>1;
	const int halfImageY = imageSizeY>>1;
	const double invgridsizeX = 1.0/((double)gridSizeX);
	const double invgridsizeY = 1.0/((double)gridSizeY);
	const double rmaxX = (double)(resolutionX*resolutionX)/4.0;
	const double rmaxY = (double)(resolutionY*resolutionY)/4.0;
	float *windowX=(float *)malloc(imageSizeX*sizeof(float));
	float *windowY=(float *)malloc(imageSizeY*sizeof(float));
	float value;
	
	for(int mm=0;mm<imageSizeX;mm++){
		windowX[mm]=(float)calc_inv((double)(mm-halfImageX)*invgridsizeX,kernel_width);
	}
	for (int mm=0;mm<imageSizeY;mm++){
		windowY[mm]=(float)calc_inv((double)(mm-halfImageY)*invgridsizeY,kernel_width);
	}
	for(int y=0;y<imageSizeY;y++){
		value=windowY[y];
		for(int x=0;x<imageSizeX;x++){
			post_window[y*imageSizeX+x]=value*windowX[x];
			post_window_crop[y*imageSizeX+x]=post_window[y*imageSizeX+x]*( ((x-halfImageX)*(x-halfImageX)/rmaxX +(y-halfImageY)*(y-halfImageY)/rmaxY)>1.0f ? 0.0f:1.0f);
			post_ones[y*imageSizeX+x]=1.0f;
		}
	}

	free(windowX);
	free(windowY);
}