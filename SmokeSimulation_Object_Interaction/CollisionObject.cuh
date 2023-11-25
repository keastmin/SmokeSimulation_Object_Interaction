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
	float _size;

public:
	static int* d_calcCollision;
	static int* d_drawCollision;
	static int* d_calcID;
	glm::vec3 _curr_pos;
	glm::vec3 _prev_pos;
	glm::vec3 _dir;
	float _vel;
	int _ID;

public:
	CollisionObject(int N, float size, glm::vec3 oInfo[], float vel, int id);
	virtual ~CollisionObject();
	virtual void check_collision(double dx, double dy, double dz) = 0;
	inline float getLength() {
		return glm::length(_curr_pos - _start_pos);
	}
	static void divide_midCell(int N);
	static void initialize_memory(int N);
	static void finalize_memory();
};

#endif __COLLISIONOBJECT__