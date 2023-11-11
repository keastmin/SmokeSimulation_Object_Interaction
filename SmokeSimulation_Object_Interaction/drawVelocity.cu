#include "drawVelocity.cuh"
#define CIX(i, j, k) ((i) + (N + 2) * (j) + (N + 2) * (N + 2) * (k))
#define DIX(i, j, k) ((i) + (N) * (j) + (N) * (N) * (k))

drawVelocity::drawVelocity(int N, double dx, double dy, double dz) {
	init(N, dx, dy, dz);
}

drawVelocity::~drawVelocity() {
	cudaGraphicsUnregisterResource(cudaVBOVel);
	glDeleteBuffers(1, &velocityBuffer);
	glDeleteBuffers(1, &velocityColorBuffer);
	cudaFree(d_static_vel_buffer);
	cudaFree(d_dynamic_vel_buffer);
}

__global__ void init_vel(int N, glm::vec3* vel, glm::vec3* stvel, glm::vec3* dyvel, double dx, double dy, double dz) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < N && j < N && k < N) {
		int idx = DIX(i, j, k);
		double x, y, z, h;
		h = 1.0f / N;
		x = (i - 0.5) * h + dx;
		y = (j - 0.5) * h + dy;
		z = (k - 0.5) * h + dz;

		glm::vec3 initPos(x, y, z);

		stvel[idx] = initPos;
		dyvel[idx] = initPos;

		vel[2 * idx + 0] = stvel[idx];
		vel[2 * idx + 1] = dyvel[idx];
	}
}

void drawVelocity::init(int N, double dx, double dy, double dz) {
	int size = N * N * N;
	size_t d_size = size * sizeof(glm::vec3);
	cudaMalloc((void**)&d_static_vel_buffer, d_size);
	cudaMalloc((void**)&d_dynamic_vel_buffer, d_size);

	glGenBuffers(1, &velocityBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, velocityBuffer);
	glBufferData(GL_ARRAY_BUFFER, 2 * d_size, NULL, GL_STREAM_DRAW);

	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	cudaGraphicsGLRegisterBuffer(&cudaVBOVel, velocityBuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cudaVBOVel, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_vel_buffer, &numByteVel, cudaVBOVel);
	init_vel<<<gridDim, blockDim >>>(N, d_vel_buffer, d_static_vel_buffer, d_dynamic_vel_buffer, dx, dy, dz);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBOVel, 0);

	d_vel_color_buffer = new glm::vec4[2 * size];
	glm::vec4 init_color(1.0, 1.0, 1.0, 0.3f);
	for (int i = 0; i < 2 * size; i++) {
		d_vel_color_buffer[i] = init_color;
	}

	glGenBuffers(1, &velocityColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, velocityColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, 2 * size * sizeof(glm::vec4), d_vel_color_buffer, GL_STATIC_DRAW);
}

__global__ void update_vel(int N, glm::vec3* vel, glm::vec3* dyvel, double* ku, double* kv, double* kw) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < N && j < N && k < N) {
		int idx = DIX(i, j, k);
		int velIdx = CIX(i + 1, j + 1, k + 1);

		vel[2 * idx + 1].x = dyvel[idx].x + ku[velIdx];
		vel[2 * idx + 1].y = dyvel[idx].y + kv[velIdx];
		vel[2 * idx + 1].z = dyvel[idx].z + kw[velIdx];
	}

}

void drawVelocity::draw_vel(int N, double* ku, double* kv, double* kw) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	cudaGraphicsMapResources(1, &cudaVBOVel, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_vel_buffer, &numByteVel, cudaVBOVel);
	update_vel << <gridDim, blockDim >> > (N, d_vel_buffer, d_static_vel_buffer, ku, kv, kw);
	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1, &cudaVBOVel, 0);

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, velocityBuffer);
	glVertexAttribPointer(
		0,
		3,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glEnableVertexAttribArray(1);
	glBindBuffer(GL_ARRAY_BUFFER, velocityColorBuffer);
	glVertexAttribPointer(
		1,
		4,
		GL_FLOAT,
		GL_FALSE,
		0,
		(void*)0
	);

	glDrawArrays(GL_LINES, 0, 2 * N * N * N);

	glDisableVertexAttribArray(0);
	glDisableVertexAttribArray(1);
}