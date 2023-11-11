#ifndef __BULLET_H__
#define __BULLET_H__

#include "CollisionObject.cuh"

class Bullet : public CollisionObject {
	int numStacks;
	int numSlices;
	int sphereNum;

	GLuint spherebuffer;
	cudaGraphicsResource* cudaVBOsphere;
	size_t numBytesphere;

	GLuint sphereColorBuffer;
	glm::vec4* sphereColors;

	glm::vec3* d_sphere_buffer;
	glm::vec3* d_color_buffer;

public:
	Bullet(int N, float size, glm::vec3 bInfo[], float vel);
	~Bullet();
	void drawBullet(double dx, double dy, double dz);
	void check_collision(double dx, double dy, double dz);
};

__global__ void inner_collision(int N, glm::vec3 pos, float size, double dx, double dy, double dz, int* d_calc, int* d_draw);
__global__ void updateBulletPos(int stacks, int slices, glm::vec3* sphere, glm::vec3 pos, float scale);


#endif __BULLET_H__