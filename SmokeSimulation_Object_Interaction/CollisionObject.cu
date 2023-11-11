#include "CollisionObject.cuh"

int* CollisionObject::d_calcCollision = nullptr;
int* CollisionObject::d_drawCollision = nullptr;
int* CollisionObject::d_ID = nullptr;

CollisionObject::CollisionObject(int N, float size, glm::vec3 oInfo[], float vel) {
	_N = N;
	_size = size;
	_start_pos = oInfo[0];
	_curr_pos = oInfo[0];
	_prev_pos = oInfo[0];
	_dir = oInfo[1];
	_vel = vel;
}

CollisionObject::~CollisionObject() {

}

void CollisionObject::initialize_memory(int N) {
	cudaMalloc((void**)&d_calcCollision, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
	cudaMalloc((void**)&d_drawCollision, N * N * N * sizeof(int));
	cudaMalloc((void**)&d_ID, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
}

void CollisionObject::finalize_memory() {
	cudaFree(d_calcCollision);
	cudaFree(d_drawCollision);
	cudaFree(d_ID);
}