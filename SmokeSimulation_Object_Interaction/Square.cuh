#ifndef __SQUARE_H__
#define __SQUARE_H__

#include "CollisionObject.cuh"

class Square :  public CollisionObject{
public:
	Square(int N, float size, glm::vec3 bInfo[], float vel, int id);
	~Square();
	void drawSquare();
	void check_collision();
};

#endif __SQUARE_H__