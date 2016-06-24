#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <map>
#include <ctime>
#include <cuda.h>
#include <math_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
using namespace std;

typedef long long ll;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice

#define THREADS 256
#define NUM_ELEMENTS 200
#define MEGA (1LL<<29)

const int max_x=NUM_ELEMENTS+1;
//const int max_y=NUM_ELEMENTS;

const int blockSize0=8192;

struct four_p{
	int4 a;
	int num;
};

inline int get_adj_size(const long long num_elem){
	double p=double(num_elem)/double(MEGA);
	if(p>0.8)return 5;
	else if(p>0.6)return 4;
	else if(p>0.4)return 3;
	else if(p>0.2)return 2;
	else
		return 1;
}
inline int get_dynamic_block_size(const int adj_size,const int blkSize){
	return (1<<(adj_size-1))*blkSize;//chk
}

void generate_random_points(float2 *Arr, const int sz,const int mx);

bool is_in_polygon_4(float2 p_arr[4],const float2 cur_point);
four_p CPU_version(const float2 *Arr,const int sz);

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

ll choose(int n, int k);

inline int choose2(int n){return n>0 ? ((n*(n-1))>>1):0;}

inline long long choose3_big(int n){
	long long nn=long long(n);
	return ((((nn*(nn-1LL))>>1LL)*(nn-2LL))/3LL);
}

inline long long choose4_big(int n){
	long long nn=long long(n);
	return (((((nn*(nn-1LL))>>1LL)*((nn-2LL)*(nn-3LL))>>1LL)>>1LL)/3LL);
}

__device__ __forceinline__ int d_choose2(int n){return n>0 ? ((n*(n-1))>>1):0;}

__device__ __forceinline__ long long d_choose3_big(int n){
	return ((((long long(n)*(long long(n)-1LL))>>1LL)*(long long(n)-2LL))/3LL);
}

__device__ __forceinline__ long long d_choose4_big(int n){
	long long nn=long long(n);
	return (((((nn*(nn-1LL))>>1LL)*((nn-2LL)*(nn-3LL))>>1LL)>>1LL)/3LL);
}

__constant__ float2 Pnt_Arr[NUM_ELEMENTS+1];//careful here, __constant__ memory has a 65536 byte limit. 

template<int blockWork>
__global__ void gpu_optimal_four(int4 *combo, int *best_num, const int sz){
	const long long offset=long long(threadIdx.x)+long long(blockIdx.x)*long long(blockWork);
	const int reps=blockWork>>8;
	const int warpIndex = threadIdx.x%32;

	__shared__ int blk_best[8];
	__shared__ int4 combo_best[8];

	int thread_best=0;
	int4 cur_best;
	float2 p,P[4];
	int ii=0,i,j,k,m,lo,hi,mid;
	long long pos,cur;

	for(;ii<reps;ii++){
		pos=offset+long long(ii*THREADS);//will be combo number
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=d_choose4_big(mid);
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=d_choose4_big(lo);
		i=lo;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=d_choose3_big(mid);
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=d_choose3_big(lo);
		j=lo;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=long long(d_choose2(mid));
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=long long(d_choose2(lo));
		k=lo;
		m=int(pos);

		P[0]=Pnt_Arr[i];P[1]=Pnt_Arr[j];P[2]=Pnt_Arr[k];P[3]=Pnt_Arr[m];

		lo=0;
		for(mid=0;mid<sz;mid++)if(mid!=i && mid!=j && mid!=k && mid!=m){
			p=Pnt_Arr[mid];
			hi=0;

			if( ((P[0].y>=p.y)!=(P[3].y>=p.y)) &&
					(p.x<=(P[3].x-P[0].x)*(p.y-P[0].y)/(P[3].y-P[0].y) +P[0].x ) ){hi=!hi;}

			if( ((P[1].y>=p.y)!=(P[0].y>=p.y)) &&
					(p.x<=(P[0].x-P[1].x)*(p.y-P[1].y)/(P[0].y-P[1].y) +P[1].x ) ){hi=!hi;}

			if( ((P[2].y>=p.y)!=(P[1].y>=p.y)) &&
					(p.x<=(P[1].x-P[2].x)*(p.y-P[2].y)/(P[1].y-P[2].y) +P[2].x ) ){hi=!hi;}

			if( ((P[3].y>=p.y)!=(P[2].y>=p.y)) &&
					(p.x<=(P[2].x-P[3].x)*(p.y-P[3].y)/(P[2].y-P[3].y) +P[3].x ) ){hi=!hi;}

			if(hi)lo++;
		}
		if(lo>thread_best){
			thread_best=lo;
			cur_best=make_int4(i,j,k,m);
		}
	}

	for(ii=16;ii>0;ii>>=1){
		mid=__shfl(thread_best,warpIndex+ii);
		i=__shfl(cur_best.w,warpIndex+ii);
		j=__shfl(cur_best.x,warpIndex+ii);
		k=__shfl(cur_best.y,warpIndex+ii);
		m=__shfl(cur_best.z,warpIndex+ii);
		if(mid>thread_best){
			thread_best=mid;
			cur_best=make_int4(i,j,k,m);
		}
	}
	if(warpIndex==0){
		blk_best[threadIdx.x>>5]=thread_best;
		combo_best[threadIdx.x>>5]=cur_best;
	}
	__syncthreads();

	if(threadIdx.x==0){

		mid=blk_best[0];
		cur_best=combo_best[0];

		if(blk_best[1]>mid){
			mid=blk_best[1];
			cur_best=combo_best[1];
		}
		if(blk_best[2]>mid){
			mid=blk_best[2];
			cur_best=combo_best[2];
		}
		if(blk_best[3]>mid){
			mid=blk_best[3];
			cur_best=combo_best[3];
		}
		if(blk_best[4]>mid){
			mid=blk_best[4];
			cur_best=combo_best[4];
		}
		if(blk_best[5]>mid){
			mid=blk_best[5];
			cur_best=combo_best[5];
		}
		if(blk_best[6]>mid){
			mid=blk_best[6];
			cur_best=combo_best[6];
		}
		if(blk_best[7]>mid){
			mid=blk_best[7];
			cur_best=combo_best[7];
		}
		best_num[blockIdx.x]=mid;
		combo[blockIdx.x]=cur_best;
	}
}

__global__ void four_last_step(int4 *combo, int *best_num, const int sz,const long long rem_start,const long long bound,const int num_blox){
	const long long offset=long long(threadIdx.x)+rem_start;
	const int warpIndex = threadIdx.x%32;

	__shared__ int blk_best[8];
	__shared__ int4 combo_best[8];

	int thread_best=0;
	int4 cur_best;
	float2 p,P[4];

	int ii=1,i,j,k,m,lo,hi,mid;
	long long pos,cur,adj=0LL;

	for(;(offset+adj)<bound;ii++){
		pos=offset+adj;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=d_choose4_big(mid);
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=d_choose4_big(lo);
		i=lo;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=d_choose3_big(mid);
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=d_choose3_big(lo);
		j=lo;
		lo=0;hi=sz+1;

		while(lo<hi){
			mid=(hi+lo+1)>>1;
			cur=long long(d_choose2(mid));
			if(cur>pos)hi=mid-1;
			else
				lo=mid;
		}
		pos-=long long(d_choose2(lo));
		k=lo;
		m=int(pos);
		P[0]=Pnt_Arr[i];P[1]=Pnt_Arr[j];P[2]=Pnt_Arr[k];P[3]=Pnt_Arr[m];

	
		lo=0;
		for(mid=0;mid<sz;mid++)if(mid!=i && mid!=j && mid!=k && mid!=m){
			p=Pnt_Arr[mid];
			hi=0;

			if( ((P[0].y>=p.y)!=(P[3].y>=p.y)) &&
					(p.x<=(P[3].x-P[0].x)*(p.y-P[0].y)/(P[3].y-P[0].y) +P[0].x ) ){hi=!hi;}

			if( ((P[1].y>=p.y)!=(P[0].y>=p.y)) &&
					(p.x<=(P[0].x-P[1].x)*(p.y-P[1].y)/(P[0].y-P[1].y) +P[1].x ) ){hi=!hi;}

			if( ((P[2].y>=p.y)!=(P[1].y>=p.y)) &&
					(p.x<=(P[1].x-P[2].x)*(p.y-P[2].y)/(P[1].y-P[2].y) +P[2].x ) ){hi=!hi;}

			if( ((P[3].y>=p.y)!=(P[2].y>=p.y)) &&
					(p.x<=(P[2].x-P[3].x)*(p.y-P[3].y)/(P[2].y-P[3].y) +P[3].x ) ){hi=!hi;}

			if(hi)lo++;
		}
		if(lo>thread_best){
			thread_best=lo;
			cur_best=make_int4(i,j,k,m);
		}

		adj=(long long(ii)<<8LL);
	}
	adj=0LL;
	for(ii=1;(threadIdx.x+int(adj))<num_blox;ii++){
		mid=(threadIdx.x+int(adj));
		if(best_num[mid]>thread_best){
			thread_best=best_num[mid];
			cur_best=combo[mid];
		}
		adj=(long long(ii)<<8LL);
	}

	for(ii=16;ii>0;ii>>=1){
		mid=__shfl(thread_best,warpIndex+ii);
		i=__shfl(cur_best.w,warpIndex+ii);
		j=__shfl(cur_best.x,warpIndex+ii);
		k=__shfl(cur_best.y,warpIndex+ii);
		m=__shfl(cur_best.z,warpIndex+ii);
		if(mid>thread_best){
			thread_best=mid;
			cur_best=make_int4(i,j,k,m);
		}
	}
	if(warpIndex==0){
		blk_best[threadIdx.x>>5]=thread_best;
		combo_best[threadIdx.x>>5]=cur_best;
	}
	__syncthreads();

	if(threadIdx.x==0){
		mid=blk_best[0];
		cur_best=combo_best[0];

		if(blk_best[1]>mid){
			mid=blk_best[1];
			cur_best=combo_best[1];
		}
		if(blk_best[2]>mid){
			mid=blk_best[2];
			cur_best=combo_best[2];
		}
		if(blk_best[3]>mid){
			mid=blk_best[3];
			cur_best=combo_best[3];
		}
		if(blk_best[4]>mid){
			mid=blk_best[4];
			cur_best=combo_best[4];
		}
		if(blk_best[5]>mid){
			mid=blk_best[5];
			cur_best=combo_best[5];
		}
		if(blk_best[6]>mid){
			mid=blk_best[6];
			cur_best=combo_best[6];
		}
		if(blk_best[7]>mid){
			mid=blk_best[7];
			cur_best=combo_best[7];
		}
		best_num[0]=mid;
		combo[0]=cur_best;
	}
}


int main(){

	srand(time(NULL));
	
	const int num_points=NUM_ELEMENTS;
	const long long range=choose4_big(num_points);
	cout<<"\nNumber of 2-D points "<<num_points<<'\n';
	cout<<"\nNumber of possible 4 point combinations= "<<range<<'\n';
	const int num_bytes_arr=num_points*sizeof(float2);
	float2 *CPU_Arr=(float2 *)malloc(num_bytes_arr);

	generate_random_points(CPU_Arr,num_points,max_x);
	int GPU_ans=-1;
	int4 GPU_points;
	


	cudaError_t err=cudaFree(0);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	cout<<"\nRunning CPU implementation..\n";
    UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
    bool init = InitMMTimer(wTimerRes);
    DWORD startTime=timeGetTime();

	four_p CPU_ans=CPU_version(CPU_Arr,num_points);

	DWORD endTime = timeGetTime();
    CPU_time=endTime-startTime;

    cout<<"CPU solution timing: "<<CPU_time<<'\n';
	cout<<"CPU best number of points= "<<CPU_ans.num<<" , point indices ( "<<CPU_ans.a.w<<" , "<<CPU_ans.a.x<<" , "<<CPU_ans.a.y<<" , "<<CPU_ans.a.z<<" ).\n";

    DestroyMMTimer(wTimerRes, init);

	err=cudaMemcpyToSymbol(Pnt_Arr,CPU_Arr,num_bytes_arr);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	
	const int adj_size=get_adj_size(range);
	const int temp_blocks_sz=get_dynamic_block_size(adj_size,blockSize0);
	const int num_blx=max(1,int(range/long long(temp_blocks_sz)));
	//cout<<"\nnum_blx= "<<num_blx;
	const long long rem_start=range-(range-long long(num_blx)*long long(temp_blocks_sz));
	//cout<<"\nrem_start= "<<rem_start;

	int *GPU_best;
	int4 *GPU_combo;
	err=cudaMalloc((void**)&GPU_best,num_blx*sizeof(int));
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaMalloc((void**)&GPU_combo,num_blx*sizeof(int4));
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	wTimerRes = 0;
    init = InitMMTimer(wTimerRes);
    startTime = timeGetTime();

	if(adj_size==1){
		gpu_optimal_four<blockSize0><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);			
	}else if(adj_size==2){
		gpu_optimal_four<blockSize0*2><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}else if(adj_size==3){
		gpu_optimal_four<blockSize0*4><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}else if(adj_size==4){
		gpu_optimal_four<blockSize0*8><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}else{
		gpu_optimal_four<blockSize0*16><<<num_blx,THREADS>>>(GPU_combo,GPU_best,num_points);
	}
	err = cudaDeviceSynchronize();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	four_last_step<<<1,THREADS>>>(GPU_combo,GPU_best,num_points,rem_start,range,num_blx);
	err = cudaDeviceSynchronize();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemcpy(&GPU_ans,GPU_best,sizeof(int),_DTH);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	err=cudaMemcpy(&GPU_points,GPU_combo,sizeof(int4),_DTH);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	endTime = timeGetTime();
    GPU_time=endTime-startTime;
	DestroyMMTimer(wTimerRes, init);

	
	err=cudaFree(GPU_best);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	err=cudaFree(GPU_combo);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	cout<<"CUDA timing: "<<GPU_time<<'\n';
	cout<<"GPU highest num internal points= "<<GPU_ans<<" , point indexes ( "<<GPU_points.w<<" , "<<GPU_points.x<<" , "<<GPU_points.y<<" , "<<GPU_points.z<<" ).\n";
	cout<<"\nNote: If there is more than one polygon which has the same optimal value, the GPU version will return a valid polygon, but not necessarily the first encountered.\n";
	cout<<"\nAlso point order may be different in GPU and CPU versions, but order does not matter as they are just points of optimal polygon.\n";
	if(GPU_ans==CPU_ans.num){
		cout<<"\nSuccess!. GPU value matches CPU results!. GPU was "<<double(CPU_time)/double(GPU_time)<<" faster than serial CPU implementation.\n";
	}else{
		cout<<"\nError in calculation!\n";
	}

	err=cudaDeviceReset();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}


	free(CPU_Arr);

	return 0;
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}

ll choose(int n, int k) {
	if((n==0 && k==0)|| (k==0 || n==k))return 1LL;
	if(k>n || n==0 || n<0 || k<0)return 0LL;
    ll res=1LL; 
    for(ll i=1LL,j=(ll)n; i<=(ll)k;++i,--j){res*=j;res/=i;} 
    return res; 
}



void generate_random_points(float2 *Arr, const int sz,const int mx){
	bool *B=(bool *)malloc(mx*mx*sizeof(bool));
	memset(B,false,mx*mx*sizeof(bool));
	int a,b;
	for(int i=0;i<sz;i++){
		do{
			a=rand()%mx;
			b=rand()%mx;
		}while(B[a*mx+b]);

		Arr[i].x=float(a);
		Arr[i].y=float(b);
		B[a*mx+b]=true;

	}
	free(B);
}


bool is_in_polygon_4(float2 p_arr[4],const float2 cur_point){
	bool ret=false;
	int i,j;
	for(i=0,j=3;i<4;j=i++){
		if( (p_arr[i].y>=cur_point.y)!=(p_arr[j].y>=cur_point.y) &&
			(cur_point.x<=(p_arr[j].x-p_arr[i].x)*(cur_point.y-p_arr[i].y)/(p_arr[j].y-p_arr[i].y)+p_arr[i].x) ){
				ret=!ret;
		}
	}	
	return ret;
}

four_p CPU_version(const float2 *Arr,const int sz){
	four_p ret={0};
	float2 p_arr[4],p;
	for(int i=0;i<sz;i++)for(int j=0;j<i;j++)for(int k=0;k<j;k++)for(int m=0;m<k;m++){
		p_arr[0]=Arr[i];p_arr[1]=Arr[j];p_arr[2]=Arr[k];p_arr[3]=Arr[m];
		int c=0;
		for(int n=0;n<sz;n++)if(n!=i && n!=j && n!=k && n!=m){
			p=Arr[n];
			if(is_in_polygon_4(p_arr,p))c++;
		}
		if(c>ret.num){
			ret.num=c;
			ret.a.w=i;
			ret.a.x=j;
			ret.a.y=k;
			ret.a.z=m;
		}
	}
	return ret;
}
