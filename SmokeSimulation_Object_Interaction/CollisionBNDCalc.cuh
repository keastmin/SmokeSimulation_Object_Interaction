#ifndef __COLLISIONBNDCALC_H__
#define __COLLISIONBNDCALC_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define IX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))

__global__ void coll_plane_bnd(int N, int b, double* x, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int idx = IX(i, j, k);
		if		(d_calc[idx] == 23) x[idx] = b == 1 ? -x[IX(i + 1, j, k)] : x[IX(i + 1, j, k)];
		else if (d_calc[idx] == 24) x[idx] = b == 1 ? -x[IX(i - 1, j, k)] : x[IX(i - 1, j, k)];
		else if (d_calc[idx] == 25) x[idx] = b == 2 ? -x[IX(i, j + 1, k)] : x[IX(i, j + 1, k)];
		else if (d_calc[idx] == 26) x[idx] = b == 2 ? -x[IX(i, j - 1, k)] : x[IX(i, j - 1, k)];
		else if (d_calc[idx] == 27) x[idx] = b == 3 ? -x[IX(i, j, k + 1)] : x[IX(i, j, k + 1)];
		else if (d_calc[idx] == 28) x[idx] = b == 3 ? -x[IX(i, j, k - 1)] : x[IX(i, j, k - 1)];
	}
}

__device__ void outter_edge_bnd(int N, int idx, int i, int j, int k, double* x, int* d_calc) {
	if (d_calc[idx] == 11) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j - 1, k)]);
	else if (d_calc[idx] == 12) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j + 1, k)]);
	else if (d_calc[idx] == 13) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 14) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 15) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)]);
	else if (d_calc[idx] == 16) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j + 1, k)]);
	else if (d_calc[idx] == 17) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 18) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 19) x[idx] = (1.0 / 2.0) * (x[IX(i, j - 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 20) x[idx] = (1.0 / 2.0) * (x[IX(i, j - 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 21) x[idx] = (1.0 / 2.0) * (x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 22) x[idx] = (1.0 / 2.0) * (x[IX(i, j + 1, k)] + x[IX(i, j, k + 1)]);
}

__device__ void inner_edge_bnd(int N, int idx, int i, int j, int k, double* x, int* d_calc) {
	if (d_calc[idx] == 37) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j + 1, k)]);
	else if (d_calc[idx] == 38) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)]);
	else if (d_calc[idx] == 39) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 40) x[idx] = (1.0 / 2.0) * (x[IX(i + 1, j, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 41) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j + 1, k)]);
	else if (d_calc[idx] == 42) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j - 1, k)]);
	else if (d_calc[idx] == 43) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 44) x[idx] = (1.0 / 2.0) * (x[IX(i - 1, j, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 45) x[idx] = (1.0 / 2.0) * (x[IX(i, j + 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 46) x[idx] = (1.0 / 2.0) * (x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 47) x[idx] = (1.0 / 2.0) * (x[IX(i, j - 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 48) x[idx] = (1.0 / 2.0) * (x[IX(i, j - 1, k)] + x[IX(i, j, k - 1)]);
}

__global__ void coll_edge_bnd(int N, double* x, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int idx = IX(i, j, k);
		if (d_calc[idx] >= 11 && d_calc[idx] <= 22) {
			outter_edge_bnd(N, idx, i, j, k, x, d_calc);
		}
		else if (d_calc[idx] >= 37 && d_calc[idx] <= 48) {
			inner_edge_bnd(N, idx, i, j, k, x, d_calc);
		}
	}
}

__device__ void outter_corner_bnd(int N, int idx, int i, int j, int k, double* x, int* d_calc) {
	if (d_calc[idx] == 3) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 4) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 5) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 6) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 7) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 8) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 9) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 10) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k + 1)]);
}

__device__ void inner_corner_bnd(int N, int idx, int i, int j, int k, double* x, int* d_calc) {
	if (d_calc[idx] == 3) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 4) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 5) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 6) (1.0 / 3.0)* (x[IX(i + 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 7) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 8) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j + 1, k)] + x[IX(i, j, k - 1)]);
	else if (d_calc[idx] == 9) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k + 1)]);
	else if (d_calc[idx] == 10) (1.0 / 3.0)* (x[IX(i - 1, j, k)] + x[IX(i, j - 1, k)] + x[IX(i, j, k - 1)]);
}

__global__ void coll_corner_bnd(int N, double* x, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int idx = IX(i, j, k);
		if (d_calc[idx] >= 3 && d_calc[idx] <= 10) {
			outter_corner_bnd(N, idx, i, j, k, x, d_calc);
		}
		else if (d_calc[idx] >= 29 && d_calc[idx] <= 36) {
			inner_corner_bnd(N, idx, i, j, k, x, d_calc);
		}
	}
}

#endif __COLLISIONBNDCALC_H__