#include "Bullet.cuh"

#define DIX(i, j, k) ((i) + (N)*(j) + (N)*(N)*(k))
#define CIX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define M_PI 3.141592

Bullet::Bullet(int N, float size, glm::vec3 bInfo[], float vel) : CollisionObject(N, size, bInfo, vel){
	numStacks = 20;
	numSlices = 20;
	sphereNum = 6 * numStacks * numSlices;

	// ��ü ����
	glGenBuffers(1, &spherebuffer);
	glBindBuffer(GL_ARRAY_BUFFER, spherebuffer);
	glBufferData(GL_ARRAY_BUFFER, sphereNum * sizeof(glm::vec3), NULL, GL_STREAM_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBOsphere, spherebuffer, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsUnmapResources(1, &cudaVBOsphere, 0);

	sphereColors = new glm::vec4[sphereNum];
	for (int i = 0; i < sphereNum; ++i) {
		sphereColors[i] = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
	}

	glGenBuffers(1, &sphereColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, sphereColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sphereNum * sizeof(glm::vec4), sphereColors, GL_STATIC_DRAW);

	std::cout << "�Ѿ� ����" << '\n';
}

Bullet::~Bullet() {
	delete sphereColors;
	glDeleteBuffers(1, &spherebuffer);
	glDeleteBuffers(1, &sphereColorBuffer);
	cudaGraphicsUnregisterResource(cudaVBOsphere);
	std::cout << "�Ѿ� �Ҹ�" << '\n';
}

__global__ void inner_collision(int N, glm::vec3 pos, float size, double dx, double dy, double dz, int* d_calc, int* d_draw) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		int dIdx = DIX(i, j, k);
		int cIdx = CIX(i + 1, j + 1, k + 1);
		double h = 1.0 / N;
		double x = (i - 0.5) * h + dx;
		double y = (j - 0.5) * h + dy;
		double z = (k - 0.5) * h + dz;

		glm::vec3 cell_center(x, y, z);

		// ��ü�� �� �߽��� ���� �Ÿ� ���
		float distance = glm::length(cell_center - pos);

		// �浹 ����
		if (distance <= size) {
			d_draw[dIdx] = 1;
			d_calc[cIdx] = 1;
		}
	}
}

__global__ void markSurroundingCells(int N, int* d_draw) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < N && j < N && k < N) {
		int idx = DIX(i, j, k);

		// ���� ���� �浹 ���̸� �ֺ� ���� Ȯ��
		if (d_draw[idx] == 1) {
			// �ֺ� ���� Ȯ���ϰ�, ����ִ� ��(0)���� 2�� ����
			for (int di = -1; di <= 1; di++) {
				for (int dj = -1; dj <= 1; dj++) {
					for (int dk = -1; dk <= 1; dk++) {
						int ni = i + di;
						int nj = j + dj;
						int nk = k + dk;
						if (ni >= 0 && ni < N && nj >= 0 && nj < N && nk >= 0 && nk < N) {
							int nIdx = DIX(ni, nj, nk);
							if (d_draw[nIdx] == 0) {
								d_draw[nIdx] = 2;
							}
						}
					}
				}
			}
		}
	}
}

void Bullet::check_collision(double dx, double dy, double dz) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((_N + blockDim.x - 1) / blockDim.x, (_N + blockDim.y - 1) / blockDim.y, (_N + blockDim.z - 1) / blockDim.z);
	inner_collision << <gridDim, blockDim >> > (_N, _curr_pos, _size, dx, dy, dz, d_calcCollision, d_drawCollision);
	//cudaDeviceSynchronize();
	markSurroundingCells << <gridDim, blockDim >> > (_N, d_drawCollision);
}

__global__ void updateBulletPos(int stacks, int slices, glm::vec3* sphere, glm::vec3 pos, float scale) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < stacks && j < slices) {
		float stackInterval = M_PI / (float)stacks;
		float sliceInterval = 2.0 * M_PI / (float)slices;

		float stackAngle1 = i * stackInterval;
		float stackAngle2 = (i + 1) * stackInterval;

		float sliceAngle1 = j * sliceInterval;
		float sliceAngle2 = (j + 1) * sliceInterval;
		double x = pos.x;
		double y = pos.y;
		double z = pos.z;

		glm::vec3 vertex1 = glm::vec3(
			x + scale * sinf(stackAngle1) * cosf(sliceAngle1),
			y + scale * cosf(stackAngle1),
			z + scale * sinf(stackAngle1) * sinf(sliceAngle1)
		);

		glm::vec3 vertex2 = glm::vec3(
			x + scale * sinf(stackAngle2) * cosf(sliceAngle1),
			y + scale * cosf(stackAngle2),
			z + scale * sinf(stackAngle2) * sinf(sliceAngle1)
		);

		glm::vec3 vertex3 = glm::vec3(
			x + scale * sinf(stackAngle1) * cosf(sliceAngle2),
			y + scale * cosf(stackAngle1),
			z + scale * sinf(stackAngle1) * sinf(sliceAngle2)
		);

		glm::vec3 vertex4 = glm::vec3(
			x + scale * sinf(stackAngle2) * cosf(sliceAngle2),
			y + scale * cosf(stackAngle2),
			z + scale * sinf(stackAngle2) * sinf(sliceAngle2)
		);

		int index = (i * slices + j) * 6;
		sphere[index + 0] = vertex1;
		sphere[index + 1] = vertex2;
		sphere[index + 2] = vertex3;

		sphere[index + 3] = vertex2;
		sphere[index + 4] = vertex4;
		sphere[index + 5] = vertex3;
	}
}

void Bullet::drawBullet(double dx, double dy, double dz) {
	_prev_pos = _curr_pos;
	_curr_pos += _dir * _vel;
	_dir = glm::normalize(_curr_pos - _prev_pos);
	_vel = glm::length(_curr_pos - _prev_pos);

	dim3 blockDim(16, 16);
	dim3 gridDim((numStacks + blockDim.x - 1) / blockDim.x, (numSlices + blockDim.y - 1) / blockDim.y);

	cudaGraphicsMapResources(1, &cudaVBOsphere, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&d_sphere_buffer, &numBytesphere, cudaVBOsphere);
	updateBulletPos << <gridDim, blockDim >> > (numStacks, numSlices, d_sphere_buffer, _curr_pos, _size);
	cudaGraphicsUnmapResources(1, &cudaVBOsphere, 0);

	check_collision(dx, dy, dz);

	//glBindBuffer(GL_ARRAY_BUFFER, spherebuffer);
	//glEnableVertexAttribArray(0);
	//glVertexAttribPointer(
	//	0,
	//	3,
	//	GL_FLOAT,
	//	GL_FALSE,
	//	0,
	//	(void*)0
	//);

	//glBindBuffer(GL_ARRAY_BUFFER, sphereColorBuffer);
	//glEnableVertexAttribArray(1);
	//glVertexAttribPointer(
	//	1,
	//	4,
	//	GL_FLOAT,
	//	GL_FALSE,
	//	0,
	//	(void*)0
	//);
	//glDrawArrays(GL_TRIANGLES, 0, sphereNum);
	//glDisableVertexAttribArray(0);
	//glDisableVertexAttribArray(1);
}