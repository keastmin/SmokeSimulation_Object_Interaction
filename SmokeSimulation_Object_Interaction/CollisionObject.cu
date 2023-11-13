#include "CollisionObject.cuh"

#define DIX(i, j, k) ((i) + (N)*(j) + (N)*(N)*(k))
#define CIX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))

int* CollisionObject::d_calcCollision = nullptr;
int* CollisionObject::d_drawCollision = nullptr;
int* CollisionObject::d_calcID = nullptr;

CollisionObject::CollisionObject(int N, float size, glm::vec3 oInfo[], float vel, int id) {
	_N = N;
	_size = size;
	_start_pos = oInfo[0];
	_curr_pos = oInfo[0];
	_prev_pos = oInfo[0];
	_dir = oInfo[1];
	_vel = vel;
	_ID = id;
}

CollisionObject::~CollisionObject() {

}

void CollisionObject::initialize_memory(int N) {
	cudaMalloc((void**)&d_calcCollision, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
	cudaMalloc((void**)&d_drawCollision, N * N * N * sizeof(int));
	cudaMalloc((void**)&d_calcID, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
	cudaMemset(d_drawCollision, 0, N * N * N * sizeof(int));
	cudaMemset(d_calcCollision, 0, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
	cudaMemset(d_calcID, 0, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
}

void CollisionObject::finalize_memory() {
	cudaFree(d_calcCollision);
	cudaFree(d_drawCollision);
	cudaFree(d_calcID);
}

__device__ bool CK_O(int i, int j, int k, int N, int* d_calc) {
	int idx = CIX(i, j, k);
	if (i < 1 || i > N || j < 1 || j > N || k < 1 || k > N) return false;
	else if (d_calc[idx] != 2 && d_calc[idx] != 49) return false;
	return true;
}

__device__ bool CK_I(int i, int j, int k, int N, int* d_calc) {
	int idx = CIX(i, j, k);
	if (i < 1 || i > N || j < 1 || j > N || k < 1 || k > N) return false;
	else if (d_calc[idx] >= 3 && d_calc[idx] <= 28) return false;
	return true;
}

__device__ bool CK_DO(int i, int j, int k, int N, int* d_calc) {
	int idx = DIX(i, j, k);
	if (i < 0 || i > N - 1 || j < 0 || j>N - 1 || k < 0 || k > N - 1) return false;
	else if (d_calc[idx] != 2 && d_calc[idx] != 49) return false;
	return true;
}

__device__ bool CK_DI(int i, int j, int k, int N, int* d_calc) {
	int idx = DIX(i, j, k);
	if (i < 0 || i > N - 1 || j < 0 || j>N - 1 || k < 0 || k > N - 1) return false;
	else if (d_calc[idx] < 3 && d_calc[idx] > 28) return false;
	return true;
}

// 충돌처리 연산 함수
__global__ void divide_midCell_calc_outter(int N, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int cIdx = CIX(i, j, k);
		if (d_calc[cIdx] == 1) {
			bool check[6] = { false };
			int di[6] = { 1, -1, 0, 0, 0, 0 };
			int dj[6] = { 0, 0, 1, -1, 0, 0 };
			int dk[6] = { 0, 0, 0, 0, 1, -1 };

			// check[0]: 오른쪽, check[1]: 왼쪽, check[2]: 위쪽
			// check[3]: 아래쪽, check[4]: 뒤쪽, check[5]: 앞쪽
			for (int u = 0; u < 6; u++) {
				check[u] = CK_O(i + di[u], j + dj[u], k + dk[u], N, d_calc);
			}

			if		(check[0] && check[2] && check[4]) d_calc[cIdx] = 3;  // 오 위 뒤 모서리
			else if (check[0] && check[2] && check[5]) d_calc[cIdx] = 4;  // 오 위 앞 모서리
			else if (check[0] && check[3] && check[4]) d_calc[cIdx] = 5;  // 오 아 뒤 모서리
			else if (check[0] && check[3] && check[5]) d_calc[cIdx] = 6;  // 오 아 앞 모서리
			else if (check[1] && check[2] && check[4]) d_calc[cIdx] = 7;  // 왼 위 뒤 모서리
			else if (check[1] && check[2] && check[5]) d_calc[cIdx] = 8;  // 왼 위 앞 모서리
			else if (check[1] && check[3] && check[4]) d_calc[cIdx] = 9;  // 왼 아 뒤 모서리
			else if (check[1] && check[3] && check[5]) d_calc[cIdx] = 10; // 왼 아 앞 모서리
			else if (check[0] && check[2]) d_calc[cIdx] = 11;			  // 오 위 엣지
			else if (check[0] && check[3]) d_calc[cIdx] = 12;			  // 오 아 엣지
			else if (check[0] && check[4]) d_calc[cIdx] = 13;			  // 오 뒤 엣지
			else if (check[0] && check[5]) d_calc[cIdx] = 14;			  // 오 앞 엣지
			else if (check[1] && check[2]) d_calc[cIdx] = 15;			  // 왼 위 엣지
			else if (check[1] && check[3]) d_calc[cIdx] = 16;			  // 왼 아 엣지
			else if (check[1] && check[4]) d_calc[cIdx] = 17;			  // 왼 뒤 엣지
			else if (check[1] && check[5]) d_calc[cIdx] = 18;			  // 왼 앞 엣지
			else if (check[2] && check[4]) d_calc[cIdx] = 19;			  // 위 뒤 엣지
			else if (check[2] && check[5]) d_calc[cIdx] = 20;			  // 위 앞 엣지
			else if (check[3] && check[4]) d_calc[cIdx] = 21;			  // 아 뒤 엣지
			else if (check[3] && check[5]) d_calc[cIdx] = 22;			  // 아 앞 엣지
			else if (check[0]) d_calc[cIdx] = 23;						  // 오 평면
			else if (check[1]) d_calc[cIdx] = 24;						  // 왼 평면
			else if (check[2]) d_calc[cIdx] = 25;						  // 위 평면
			else if (check[3]) d_calc[cIdx] = 26;						  // 아 평면
			else if (check[4]) d_calc[cIdx] = 27;						  // 뒤 평면
			else if (check[5]) d_calc[cIdx] = 28;						  // 앞 평면
		}
	}
}

// draw 디버깅 함수
__global__ void divide_midCell_draw_outter(int N, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		int cIdx = DIX(i, j, k);
		if (d_calc[cIdx] == 1) {
			bool check[6] = { false };
			int di[6] = { 1, -1, 0, 0, 0, 0 };
			int dj[6] = { 0, 0, 1, -1, 0, 0 };
			int dk[6] = { 0, 0, 0, 0, 1, -1 };

			// check[0]: 오른쪽, check[1]: 왼쪽, check[2]: 위쪽
			// check[3]: 아래쪽, check[4]: 뒤쪽, check[5]: 앞쪽
			for (int u = 0; u < 6; u++) {
				check[u] = CK_DO(i + di[u], j + dj[u], k + dk[u], N, d_calc);
			}

			if		(check[0] && check[2] && check[4]) d_calc[cIdx] = 3;  // 오 위 뒤 모서리
			else if (check[0] && check[2] && check[5]) d_calc[cIdx] = 4;  // 오 위 앞 모서리
			else if (check[0] && check[3] && check[4]) d_calc[cIdx] = 5;  // 오 아 뒤 모서리
			else if (check[0] && check[3] && check[5]) d_calc[cIdx] = 6;  // 오 아 앞 모서리
			else if (check[1] && check[2] && check[4]) d_calc[cIdx] = 7;  // 왼 위 뒤 모서리
			else if (check[1] && check[2] && check[5]) d_calc[cIdx] = 8;  // 왼 위 앞 모서리
			else if (check[1] && check[3] && check[4]) d_calc[cIdx] = 9;  // 왼 아 뒤 모서리
			else if (check[1] && check[3] && check[5]) d_calc[cIdx] = 10; // 왼 아 앞 모서리
			else if (check[0] && check[2]) d_calc[cIdx] = 11;			  // 오 위 엣지
			else if (check[0] && check[3]) d_calc[cIdx] = 12;			  // 오 아 엣지
			else if (check[0] && check[4]) d_calc[cIdx] = 13;			  // 오 뒤 엣지
			else if (check[0] && check[5]) d_calc[cIdx] = 14;			  // 오 앞 엣지
			else if (check[1] && check[2]) d_calc[cIdx] = 15;			  // 왼 위 엣지
			else if (check[1] && check[3]) d_calc[cIdx] = 16;			  // 왼 아 엣지
			else if (check[1] && check[4]) d_calc[cIdx] = 17;			  // 왼 뒤 엣지
			else if (check[1] && check[5]) d_calc[cIdx] = 18;			  // 왼 앞 엣지
			else if (check[2] && check[4]) d_calc[cIdx] = 19;			  // 위 뒤 엣지
			else if (check[2] && check[5]) d_calc[cIdx] = 20;			  // 위 앞 엣지
			else if (check[3] && check[4]) d_calc[cIdx] = 21;			  // 아 뒤 엣지
			else if (check[3] && check[5]) d_calc[cIdx] = 22;			  // 아 앞 엣지
			else if (check[0]) d_calc[cIdx] = 23;						  // 오 평면
			else if (check[1]) d_calc[cIdx] = 24;						  // 왼 평면
			else if (check[2]) d_calc[cIdx] = 25;						  // 위 평면
			else if (check[3]) d_calc[cIdx] = 26;						  // 아 평면
			else if (check[4]) d_calc[cIdx] = 27;						  // 뒤 평면
			else if (check[5]) d_calc[cIdx] = 28;						  // 앞 평면
		}
	}
}

__device__ bool CK_CCI(int i, int j, int k, int N, int* d_calc) {
	if (i > 0 && i < N + 1 && j > 0 && j < N + 1 && k > 0 && k < N + 1) {
		int idx = CIX(i, j, k);
		if (d_calc[idx] == 2 || d_calc[idx] == 49) {
			return true;
		}
	}
	return false;
}

__global__ void divide_midCell_calc_inner(int N, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int cIdx = CIX(i, j, k);
		if (d_calc[cIdx] == 1) {
			bool check[6] = { false };
			bool checkIn = false;
			int di[6] = { 1, -1, 0, 0, 0, 0 };
			int dj[6] = { 0, 0, 1, -1, 0, 0 };
			int dk[6] = { 0, 0, 0, 0, 1, -1 };

			// check[0]: 오른쪽, check[1]: 왼쪽, check[2]: 위쪽
			// check[3]: 아래쪽, check[4]: 뒤쪽, check[5]: 앞쪽
			for (int u = 0; u < 6; u++) {
				check[u] = CK_I(i + di[u], j + dj[u], k + dk[u], N, d_calc);
			}

			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					for (int w = -1; w <= 1; w++) {
						checkIn = CK_CCI(i + u, j + v, k + w, N, d_calc);
					}
				}
			}

			if		(check[0] && check[2] && check[4] && checkIn) d_calc[cIdx] = 29; // 오 위 뒤 모서리
			else if (check[0] && check[2] && check[5] && checkIn) d_calc[cIdx] = 30; // 오 위 앞 모서리
			else if (check[0] && check[3] && check[4] && checkIn) d_calc[cIdx] = 31; // 오 아 뒤 모서리
			else if (check[0] && check[3] && check[5] && checkIn) d_calc[cIdx] = 32; // 오 아 앞 모서리
			else if (check[1] && check[2] && check[4] && checkIn) d_calc[cIdx] = 33; // 왼 위 뒤 모서리
			else if (check[1] && check[2] && check[5] && checkIn) d_calc[cIdx] = 34; // 왼 위 앞 모서리
			else if (check[1] && check[3] && check[4] && checkIn) d_calc[cIdx] = 35; // 왼 아 뒤 모서리
			else if (check[1] && check[3] && check[5] && checkIn) d_calc[cIdx] = 36; // 왼 아 앞 모서리
			else if (check[0] && check[2] && checkIn) d_calc[cIdx] = 37;			  // 오 위 엣지
			else if (check[0] && check[3] && checkIn) d_calc[cIdx] = 38;			  // 오 아 엣지
			else if (check[0] && check[4] && checkIn) d_calc[cIdx] = 39;			  // 오 뒤 엣지
			else if (check[0] && check[5] && checkIn) d_calc[cIdx] = 40;			  // 오 앞 엣지
			else if (check[1] && check[2] && checkIn) d_calc[cIdx] = 41;			  // 왼 위 엣지
			else if (check[1] && check[3] && checkIn) d_calc[cIdx] = 42;			  // 왼 아 엣지
			else if (check[1] && check[4] && checkIn) d_calc[cIdx] = 43;			  // 왼 뒤 엣지
			else if (check[1] && check[5] && checkIn) d_calc[cIdx] = 44;			  // 왼 앞 엣지
			else if (check[2] && check[4] && checkIn) d_calc[cIdx] = 45;			  // 위 뒤 엣지
			else if (check[2] && check[5] && checkIn) d_calc[cIdx] = 46;			  // 위 앞 엣지
			else if (check[3] && check[4] && checkIn) d_calc[cIdx] = 47;			  // 아 뒤 엣지
			else if (check[3] && check[5] && checkIn) d_calc[cIdx] = 48;			  // 아 앞 엣지
		}
	}
}

__device__ bool CK_DCI(int i, int j, int k, int N, int* d_calc) {
	if (i >= 0 && i <= N - 1 && j >= 0 && j <= N - 1 && k >= 0 && k <= N - 1) {
		int idx = DIX(i, j, k);
		if (d_calc[idx] == 2 || d_calc[idx] == 49) {
			return true;
		}
	}
	return false;
}

__global__ void divide_midCell_draw_inner(int N, int* d_calc) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		int cIdx = DIX(i, j, k);
		if (d_calc[cIdx] == 1) {
			bool check[6] = { false };
			bool checkIn = false;
			int di[6] = { 1, -1, 0, 0, 0, 0 };
			int dj[6] = { 0, 0, 1, -1, 0, 0 };
			int dk[6] = { 0, 0, 0, 0, 1, -1 };

			// check[0]: 오른쪽, check[1]: 왼쪽, check[2]: 위쪽
			// check[3]: 아래쪽, check[4]: 뒤쪽, check[5]: 앞쪽
			for (int u = 0; u < 6; u++) {
				check[u] = CK_DI(i + di[u], j + dj[u], k + dk[u], N, d_calc);
			}

			for (int u = -1; u <= 1; u++) {
				for (int v = -1; v <= 1; v++) {
					for (int w = -1; w <= 1; w++) {
						checkIn = CK_DCI(i + u, j + v, k + w, N, d_calc);
					}
				}
			}

			if		(check[0] && check[2] && check[4] && checkIn) d_calc[cIdx] = 29; // 오 위 뒤 모서리
			else if (check[0] && check[2] && check[5] && checkIn) d_calc[cIdx] = 30; // 오 위 앞 모서리
			else if (check[0] && check[3] && check[4] && checkIn) d_calc[cIdx] = 31; // 오 아 뒤 모서리
			else if (check[0] && check[3] && check[5] && checkIn) d_calc[cIdx] = 32; // 오 아 앞 모서리
			else if (check[1] && check[2] && check[4] && checkIn) d_calc[cIdx] = 33; // 왼 위 뒤 모서리
			else if (check[1] && check[2] && check[5] && checkIn) d_calc[cIdx] = 34; // 왼 위 앞 모서리
			else if (check[1] && check[3] && check[4] && checkIn) d_calc[cIdx] = 35; // 왼 아 뒤 모서리
			else if (check[1] && check[3] && check[5] && checkIn) d_calc[cIdx] = 36; // 왼 아 앞 모서리
			else if (check[0] && check[2] && checkIn) d_calc[cIdx] = 37;			  // 오 위 엣지
			else if (check[0] && check[3] && checkIn) d_calc[cIdx] = 38;			  // 오 아 엣지
			else if (check[0] && check[4] && checkIn) d_calc[cIdx] = 39;			  // 오 뒤 엣지
			else if (check[0] && check[5] && checkIn) d_calc[cIdx] = 40;			  // 오 앞 엣지
			else if (check[1] && check[2] && checkIn) d_calc[cIdx] = 41;			  // 왼 위 엣지
			else if (check[1] && check[3] && checkIn) d_calc[cIdx] = 42;			  // 왼 아 엣지
			else if (check[1] && check[4] && checkIn) d_calc[cIdx] = 43;			  // 왼 뒤 엣지
			else if (check[1] && check[5] && checkIn) d_calc[cIdx] = 44;			  // 왼 앞 엣지
			else if (check[2] && check[4] && checkIn) d_calc[cIdx] = 45;			  // 위 뒤 엣지
			else if (check[2] && check[5] && checkIn) d_calc[cIdx] = 46;			  // 위 앞 엣지
			else if (check[3] && check[4] && checkIn) d_calc[cIdx] = 47;			  // 아 뒤 엣지
			else if (check[3] && check[5] && checkIn) d_calc[cIdx] = 48;			  // 아 앞 엣지
		}
	}
}

void CollisionObject::divide_midCell(int N) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	divide_midCell_calc_outter<<<gridDim, blockDim>>>(N, d_calcCollision);
	//divide_midCell_draw_outter<<<gridDim, blockDim>>>(N, d_drawCollision);
	cudaDeviceSynchronize();
	divide_midCell_calc_inner<<<gridDim, blockDim>>>(N, d_calcCollision);
	//divide_midCell_draw_inner<<<gridDim, blockDim>>>(N, d_drawCollision);
	cudaDeviceSynchronize();
}