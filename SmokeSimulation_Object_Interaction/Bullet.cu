#include "Bullet.cuh"

#define DIX(i, j, k) ((i) + (N)*(j) + (N)*(N)*(k))
#define CIX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define M_PI 3.141592

Bullet::Bullet(int N, float size, glm::vec3 bInfo[], float vel, int id) : CollisionObject(N, size, bInfo, vel, id){
	numStacks = 20;
	numSlices = 20;
	sphereNum = 6 * numStacks * numSlices;

	// 구체 버퍼
	glGenBuffers(1, &spherebuffer);
	glBindBuffer(GL_ARRAY_BUFFER, spherebuffer);
	glBufferData(GL_ARRAY_BUFFER, sphereNum * sizeof(glm::vec3), NULL, GL_STREAM_DRAW);

	cudaGraphicsGLRegisterBuffer(&cudaVBOsphere, spherebuffer, cudaGraphicsMapFlagsWriteDiscard);

	sphereColors = new glm::vec4[sphereNum];
	for (int i = 0; i < sphereNum; ++i) {
		sphereColors[i] = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
	}

	glGenBuffers(1, &sphereColorBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, sphereColorBuffer);
	glBufferData(GL_ARRAY_BUFFER, sphereNum * sizeof(glm::vec4), sphereColors, GL_STATIC_DRAW);

	std::cout << "총알 생성 ID: " << _ID << '\n';
}

Bullet::~Bullet() {
	delete sphereColors;
	glDeleteBuffers(1, &spherebuffer);
	glDeleteBuffers(1, &sphereColorBuffer);
	cudaGraphicsUnregisterResource(cudaVBOsphere);
	std::cout << "총알 소멸" << '\n';
}

__global__ void inner_collision(int N, int id, glm::vec3 pos, float size, double dx, double dy, double dz, int* d_calc, int* d_draw, int* cID) {
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

		// 구체와 셀 중심점 간의 거리 계산
		float distance = glm::length(cell_center - pos);

		// 충돌 감지
		if (distance <= size) {
			d_draw[dIdx] = 1;
			d_calc[cIdx] = 1;
			cID[cIdx] = id;
		}
	}
}

__global__ void outter_collision(int N, int id, int* d_calc, int* d_draw, int* cID) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	if (i < N && j < N && k < N) {
		int dIdx = DIX(i, j, k);
		int cIdx = CIX(i + 1, j + 1, k + 1);

		// 현재 셀이 충돌 셀이면 주변 셀을 확인
		if (d_draw[dIdx] == 1 && cID[cIdx] == id) {
			// 주변 셀을 확인하고, 비어있는 셀(0)에만 2를 저장
			for (int di = -1; di <= 1; di++) {
				for (int dj = -1; dj <= 1; dj++) {
					for (int dk = -1; dk <= 1; dk++) {
						int ni = i + di;
						int nj = j + dj;
						int nk = k + dk;
						if (ni >= 0 && ni < N && nj >= 0 && nj < N && nk >= 0 && nk < N) {
							int ndIdx = DIX(ni, nj, nk);
							int ncIdx = CIX(ni + 1, nj + 1, nk + 1);
							if (d_draw[ndIdx] == 0) {
								d_draw[ndIdx] = 2;
								d_calc[ncIdx] = 2;
								cID[ncIdx] = id;
							}
						}
					}
				}
			}
		}
	}
}

__global__ void collision_direction(int N, int id, glm::vec3 pos, int* drawIdxVal, int* calcIdxVal, glm::vec3 dir, float vel, double dx, double dy, double dz, int* cID) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i < N && j < N && k < N) {
		int dIdx = DIX(i, j, k);
		int cIdx = CIX(i + 1, j + 1, k + 1);
		double h, x, y, z;
		h = 1.0 / N;
		x = (i - 0.5) * h + dx;
		y = (j - 0.5) * h + dy;
		z = (k - 0.5) * h + dz;

		glm::vec3 cell_center(x, y, z);

		// 진행방향에 존재하는 셀 구하기
		// 구체와 셀 중심점 간의 벡터	
		glm::vec3 cell_to_sphere = cell_center - pos;

		// 벡터 정규화
		float length = glm::length(cell_to_sphere);
		if (length != 0) {
			cell_to_sphere /= length;
		}

		float cos_similarity = glm::dot(cell_to_sphere, dir);

		// 임계값 설정 (예: 0.5)
		float threshold = 0.0;
		if ((cos_similarity > threshold) && drawIdxVal[dIdx] == 2 && cID[cIdx] == id) {
			drawIdxVal[dIdx] = 49;
			calcIdxVal[cIdx] = 49;
		}
	}
}

void Bullet::check_collision(double dx, double dy, double dz) {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((_N + blockDim.x - 1) / blockDim.x, (_N + blockDim.y - 1) / blockDim.y, (_N + blockDim.z - 1) / blockDim.z);
	inner_collision<<<gridDim, blockDim>>>(_N, _ID, _curr_pos, _size, dx, dy, dz, d_calcCollision, d_drawCollision, d_calcID);
	outter_collision<<<gridDim, blockDim>>>(_N, _ID, d_calcCollision, d_drawCollision, d_calcID);
	collision_direction<<<gridDim, blockDim>>>(_N, _ID, _curr_pos, d_drawCollision, d_calcCollision, _dir, _vel, dx, dy, dz, d_calcID);
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