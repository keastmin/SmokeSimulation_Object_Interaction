#ifndef __DRAWDENSITY_H__
#define __DRAWDENSITY_H__

#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "CollisionObject.cuh"

class drawDensity {
	// 연기를 그릴 위치 버퍼와 색상 버퍼
	glm::vec3* d_dens_buffer;
	glm::vec4* d_dens_color_buffer;

	GLuint densityBuffer;
	cudaGraphicsResource* cudaVBODens;
	size_t numByteDens;

	GLuint densityColorBuffer;
	cudaGraphicsResource* cudaVBODensColor;
	size_t numByteDensColor;
public:
	drawDensity(int N, double dx, double dy, double dz);
	~drawDensity();
public:
	void init(int N, double dx, double dy, double dz);
	void draw_dens(int N, double* kd);
};

__device__ void addCubeFaceDevice(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3* buffer, int& index);
__device__ void addCubeFaceColorDevice(glm::vec4 p0, glm::vec4 p1, glm::vec4 p2, glm::vec4 p3, glm::vec4* buffer, int& index);
__global__ void init_dens(int N, glm::vec3* dens, double dx, double dy, double dz);
__global__ void init_dens_color(int N, glm::vec4* densC);
__global__ void update_dens(int N, glm::vec4* densC, double* kd, int* d_draw);



#endif __DRAWDENSITY_H__