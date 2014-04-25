#include <cstdlib>
#include <cstdio>

double calc_w(double u,double k_w);
double calc_inv(double x,double k_w);

void calc_deapod_window(const int gridSizeX,
						const int gridSizeY,
						const int imageSizeX,
						const int imageSizeY,
						const int resolutionX,
						const int resolutionY,
						float *post_window,
						float *post_window_crop,
						float *post_ones);