#ifndef __COLLISIONOBJECT__
#define __COLLISIONOBJECT__

#include <iostream>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

class CollisionObject {
protected:
	int _N;
	glm::vec3 _start_pos;
	glm::vec3 _curr_pos;
	glm::vec3 _prev_pos;
	float _size;

public:
	static int* h_calcCollision;
	static int* h_drawCollision;
	static int* d_calcCollision;
	static int* d_drawCollision;
	glm::vec3 _dir;
	float _vel;

public:
	CollisionObject(int N, float size, glm::vec3 oInfo[], float vel);
	virtual ~CollisionObject();
	virtual void check_collision(double dx, double dy, double dz) = 0;
	inline float getLength() {
		return glm::length(_curr_pos - _start_pos);
	}
	static void initialize_memory(int N);
	static void finalize_memory();
};

#endif __COLLISIONOBJECT__