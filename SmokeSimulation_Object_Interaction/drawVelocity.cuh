#ifndef __DRAWVELOCITY_H__
#define __DRAWVELOCITY_H__

#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class drawVelocity {
	double sourceAlp;

	glm::vec3* d_static_vel_buffer;
	glm::vec3* d_dynamic_vel_buffer;
	glm::vec3* d_vel_buffer;
	glm::vec4* d_vel_color_buffer;

	GLuint velocityBuffer;
	GLuint velocityColorBuffer;
	cudaGraphicsResource* cudaVBOVel;
	size_t numByteVel;

public:
	drawVelocity(int N, double dx, double dy, double dz);
	~drawVelocity();
public:
	void init(int N, double dx, double dy, double dz);
	void draw_vel(int N, double* ku, double* kv, double* kw);
};

__global__ void init_vel(int N, glm::vec3* vel, glm::vec3* stvel, glm::vec3* dyvel, double dx, double dy, double dz);
__global__ void update_vel(int N, glm::vec3* vel, glm::vec3* dyvel, double* ku, double* kv, double* kw);

#endif __DRAWVELOCITY_H__