#include "drawDensity.cuh"

#define CIX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define DIX(i, j, k) ((i) + (N)*(j) + (N)*(N)*(k))

drawDensity::drawDensity(int N, double dx, double dy, double dz) {
	init(N, dx, dy, dz);
}

drawDensity::~drawDensity() {
	glDeleteBuffers(1, &densityBuffer);
	glDeleteBuffers(1, &densityColorBuffer);
	cudaGraphicsUnregisterResource(cudaVBODens);
	cudaGraphicsUnregisterResource(cudaVBODensColor);
}

// 위치 버퍼 값 삽입
__device__ void addCubeFaceDevice(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3* buffer, int& index) {
	buffer[index++] = p0;
	buffer[index++] = p1;
	buffer[index++] = p2;

	buffer[index++] = p2;
	buffer[index++] = p3;
	buffer[index++] = p0;
}

// 색상 버퍼 값 삽입
__device__ void addCubeFaceColorDevice(glm::vec4 p0, glm::vec4 p1, glm::vec4 p2, glm::vec4 p3, glm::vec4* buffer, int& index) {
	buffer[index++] = p0;
	buffer[index++] = p1;
	buffer[index++] = p2;

	buffer[index++] = p2;
	buffer[index++] = p3;
	buffer[index++] = p0;
}

// density 초기화 커널 함수
__global__ void init_dens(int N, glm::vec3* dens, double dx, double dy, double dz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		int idx = DIX(i, j, k);
		double x, y, z, h;
		h = 1.0 / N;
		x = (i - 1.0) * h + dx;
		y = (j - 1.0) * h + dy;
		z = (k - 1.0) * h + dz;

		glm::vec3 p000(x, y, z);
		glm::vec3 p100(x + h, y, z);
		glm::vec3 p110(x + h, y + h, z);
		glm::vec3 p101(x + h, y, z + h);
		glm::vec3 p111(x + h, y + h, z + h);
		glm::vec3 p010(x, y + h, z);
		glm::vec3 p011(x, y + h, z + h);
		glm::vec3 p001(x, y, z + h);

		int localIdx = 36 * idx;
		addCubeFaceDevice(p000, p010, p110, p100, dens, localIdx);
		addCubeFaceDevice(p001, p011, p111, p101, dens, localIdx);
		addCubeFaceDevice(p000, p001, p101, p100, dens, localIdx);
		addCubeFaceDevice(p010, p011, p111, p110, dens, localIdx);
		addCubeFaceDevice(p000, p010, p011, p001, dens, localIdx);
		addCubeFaceDevice(p100, p110, p111, p101, dens, localIdx);
	}
}

// density color 초기화 커널 함수
__global__ void init_dens_color(int N, glm::vec4* cDens) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		int idx = DIX(i, j, k);

		glm::vec4 icolor(0.0f, 0.0f, 0.0f, 0.0f);
		int localIdx = 36 * idx;
		addCubeFaceColorDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceColorDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceColorDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceColorDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceColorDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
		addCubeFaceColorDevice(icolor, icolor, icolor, icolor, cDens, localIdx);
	}
}

void drawDensity::init(int N, double dx, double dy, double dz) {
	int size = N * N * N;
	size_t d_size = size * sizeof(glm::vec3);

	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);

	// 위치 버퍼
	glGenBuffers(1, &densityBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, densityBuffer);
	glBufferData(GL_ARRAY_BUFFER, 36 * d_size, NULL, GL_STATIC_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBODens, densityBuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBODens, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_dens_buffer, &numByteDens, cudaVBODens);
	init_dens << <gridDim, blockDim >> > (N, d_dens_buffer, dx, dy, dz);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODens, 0);

	// 컬러 버퍼
	d_size = size * sizeof(glm::vec4);
	glGenBuffers(1, &densityColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, densityColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, 36 * d_size, NULL, GL_STREAM_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBODensColor, densityColorBuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBODensColor, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_dens_color_buffer, &numByteDensColor, cudaVBODensColor);
	init_dens_color << <gridDim, blockDim >> > (N, d_dens_color_buffer);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODensColor, 0);
}

__global__ void update_dens(int N, glm::vec4* densC, double* kd, int* d_draw) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		double d000, d100, d110, d101, d111, d010, d011, d001;
		int dIdx = DIX(i, j, k);
		d000 = kd[CIX(i + 1, j + 1, k + 1)];
		d100 = kd[CIX(i + 2, j + 1, k + 1)];
		d110 = kd[CIX(i + 2, j + 2, k + 1)];
		d101 = kd[CIX(i + 2, j + 1, k + 2)];
		d111 = kd[CIX(i + 2, j + 2, k + 2)];
		d010 = kd[CIX(i + 1, j + 2, k + 1)];
		d011 = kd[CIX(i + 1, j + 2, k + 2)];
		d001 = kd[CIX(i + 1, j + 1, k + 2)];

		// 투명도 조절
		glm::vec4 p000(d000, d000, d000, d000);
		glm::vec4 p100(d100, d100, d100, d100);
		glm::vec4 p110(d110, d110, d110, d110);
		glm::vec4 p101(d101, d101, d101, d101);
		glm::vec4 p111(d111, d111, d111, d111);
		glm::vec4 p010(d010, d010, d010, d010);
		glm::vec4 p011(d011, d011, d011, d011);
		glm::vec4 p001(d001, d001, d001, d001);

		// 투명도 고정
		//glm::vec4 p000(d000, d000, d000, 0.03f);
		//glm::vec4 p100(d100, d100, d100, 0.03f);
		//glm::vec4 p110(d110, d110, d110, 0.03f);
		//glm::vec4 p101(d101, d101, d101, 0.03f);
		//glm::vec4 p111(d111, d111, d111, 0.03f);
		//glm::vec4 p010(d010, d010, d010, 0.03f);
		//glm::vec4 p011(d011, d011, d011, 0.03f);
		//glm::vec4 p001(d001, d001, d001, 0.03f);

		glm::vec4 CCIn(1.0f, 0.0f, 0.0f, 1.0f);
		glm::vec4 CCOut(1.0f, 1.0f, 0.0f, 0.2f);
		glm::vec4 CCMid(1.0f, 0.0f, 1.0f, 0.5f);
		glm::vec4 CV(0.0f, 1.0f, 1.0f, 1.0f);

		int localIdx = 36 * dIdx;
		//if (d_draw[dIdx] >= 3 && d_draw[dIdx] <= 48) {
		//	addCubeFaceColorDevice(CCMid, CCMid, CCMid, CCMid, densC, localIdx);
		//	addCubeFaceColorDevice(CCMid, CCMid, CCMid, CCMid, densC, localIdx);
		//	addCubeFaceColorDevice(CCMid, CCMid, CCMid, CCMid, densC, localIdx);
		//	addCubeFaceColorDevice(CCMid, CCMid, CCMid, CCMid, densC, localIdx);
		//	addCubeFaceColorDevice(CCMid, CCMid, CCMid, CCMid, densC, localIdx);
		//	addCubeFaceColorDevice(CCMid, CCMid, CCMid, CCMid, densC, localIdx);
		//}
		//else if (d_draw[dIdx] == 1) {
		//	addCubeFaceColorDevice(CCIn, CCIn, CCIn, CCIn, densC, localIdx);
		//	addCubeFaceColorDevice(CCIn, CCIn, CCIn, CCIn, densC, localIdx);
		//	addCubeFaceColorDevice(CCIn, CCIn, CCIn, CCIn, densC, localIdx);
		//	addCubeFaceColorDevice(CCIn, CCIn, CCIn, CCIn, densC, localIdx);
		//	addCubeFaceColorDevice(CCIn, CCIn, CCIn, CCIn, densC, localIdx);
		//	addCubeFaceColorDevice(CCIn, CCIn, CCIn, CCIn, densC, localIdx);
		//}
		//else if (d_draw[dIdx] == 2) {
		//	addCubeFaceColorDevice(CCOut, CCOut, CCOut, CCOut, densC, localIdx);
		//	addCubeFaceColorDevice(CCOut, CCOut, CCOut, CCOut, densC, localIdx);
		//	addCubeFaceColorDevice(CCOut, CCOut, CCOut, CCOut, densC, localIdx);
		//	addCubeFaceColorDevice(CCOut, CCOut, CCOut, CCOut, densC, localIdx);
		//	addCubeFaceColorDevice(CCOut, CCOut, CCOut, CCOut, densC, localIdx);
		//	addCubeFaceColorDevice(CCOut, CCOut, CCOut, CCOut, densC, localIdx);
		//}
		//else if (d_draw[dIdx] == 49) {
		//	addCubeFaceColorDevice(CV, CV, CV, CV, densC, localIdx);
		//	addCubeFaceColorDevice(CV, CV, CV, CV, densC, localIdx);
		//	addCubeFaceColorDevice(CV, CV, CV, CV, densC, localIdx);
		//	addCubeFaceColorDevice(CV, CV, CV, CV, densC, localIdx);
		//	addCubeFaceColorDevice(CV, CV, CV, CV, densC, localIdx);
		//	addCubeFaceColorDevice(CV, CV, CV, CV, densC, localIdx);
		//}
		//else {
			addCubeFaceColorDevice(p000, p010, p110, p100, densC, localIdx);
			addCubeFaceColorDevice(p001, p011, p111, p101, densC, localIdx);
			addCubeFaceColorDevice(p000, p001, p101, p100, densC, localIdx);
			addCubeFaceColorDevice(p010, p011, p111, p110, densC, localIdx);
			addCubeFaceColorDevice(p000, p010, p011, p001, densC, localIdx);
			addCubeFaceColorDevice(p100, p110, p111, p101, densC, localIdx);
		//}
	}
}

void drawDensity::draw_dens(int N, double* kd) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	cudaGraphicsMapResources(1, &cudaVBODensColor, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_dens_color_buffer, &numByteDensColor, cudaVBODensColor);
	update_dens << <gridDim, blockDim >> > (N, d_dens_color_buffer, kd, CollisionObject::d_drawCollision);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBODensColor, 0);

	glBindBuffer(GL_ARRAY_BUFFER, densityBuffer);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glBindBuffer(GL_ARRAY_BUFFER, densityColorBuffer);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(
		1,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glDrawArrays(GL_TRIANGLES, 0, 36 * N * N * N);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}