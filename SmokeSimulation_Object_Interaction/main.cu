#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "control.h"
#include "shader.h"
#include "drawVelocity.cuh"
#include "drawDensity.cuh"
#include "FluidSolver3D.cuh"
#include "CollisionObject.cuh"
#include "Bullet.cuh"

// ��ü ����
drawVelocity* _vel;
drawDensity* _den;
CollisionObject* _coll;
std::vector<std::unique_ptr<Bullet>> _bullet;

// 3���� �ε����� 1���� �ε���ó�� ����
#define CIX(i, j, k) ((i) + (N+2)*(j) + (N+2)*(N+2)*(k))
#define DIX(i, j, k) ((i) + (N)*(j) + (N)*(N)*(k))

// ������ ����
GLFWwindow* window;

// �׸��� ũ��
#define SIZE 64

// ������ ������ ����
#define WINDOW_WIDTH 800
#define WINDOW_HEIGHT 800
static int width = WINDOW_WIDTH;
static int height = WINDOW_HEIGHT;

// �ֹ��� ���� GPU �޸� �Ҵ� ����
static double* u, * v, * w, * u_prev, * v_prev, * w_prev;
static double* dens, * dens_prev;

// �ֹ��� ���� ��� ������
static const int N = SIZE;
static double dt = 0.08;
static double diff = 0.0;
static double visc = 0.0;
static double force = 15.0;
static double source = 200.0f;

// �ùķ��̼� ���� ����
static int addforce = 0;
static int mode = 0;
static int simulation_stop = 0;
static int objMode = 0;
static bool boom = false;

// �ùķ��̼� ��ġ
double drawX = -0.5;
double drawY = -0.5;
double drawZ = -0.5;

// �Ѿ��� ũ��� �ӵ� �׸��� ID
float bulletSize = 0.07f;
float bulletVel = 0.1f;
int bulletID = 1;
const int maxObject = 10;
glm::vec3* OBJ_Dir;
float* OBJ_Vel;

// ������ �Ҹ�
void free_data() {
	if (u) cudaFree(u);
	if (v) cudaFree(v);
	if (w) cudaFree(w);
	if (u_prev) cudaFree(u_prev);
	if (v_prev) cudaFree(v_prev);
	if (w_prev) cudaFree(w_prev);
	if (dens) cudaFree(dens);
	if (dens_prev) cudaFree(dens_prev);
	if (_vel) delete _vel;
	if (_den) delete _den;
	if (OBJ_Dir) cudaFree(OBJ_Dir);
	if (OBJ_Vel) cudaFree(OBJ_Vel);

	_coll->finalize_memory();
}

/* --------------------������ �ʱ�ȭ-------------------- */
// ������ �ʱⰪ ���� Ŀ�� �Լ�
__global__ void initArray(double* array, int size) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size) {
		array[i] = 0.0;
	}
}

// �ʱ�ȭ Ŀ�� ���� �Լ�
static void init_data() {
	int size = (N + 2) * (N + 2) * (N + 2);
	size_t d_size = size * sizeof(double);

	cudaMalloc((void**)&u, d_size);
	cudaMalloc((void**)&v, d_size);
	cudaMalloc((void**)&w, d_size);
	cudaMalloc((void**)&u_prev, d_size);
	cudaMalloc((void**)&v_prev, d_size);
	cudaMalloc((void**)&w_prev, d_size);
	cudaMalloc((void**)&dens, d_size);
	cudaMalloc((void**)&dens_prev, d_size);
	cudaMalloc((void**)&OBJ_Dir, maxObject * sizeof(glm::vec3));
	cudaMalloc((void**)&OBJ_Vel, maxObject * sizeof(float));

	int blockSize = 256;
	int numBlocks = (size + blockSize - 1) / blockSize;
	initArray << <numBlocks, blockSize >> > (u, size);
	initArray << <numBlocks, blockSize >> > (v, size);
	initArray << <numBlocks, blockSize >> > (w, size);
	initArray << <numBlocks, blockSize >> > (u_prev, size);
	initArray << <numBlocks, blockSize >> > (v_prev, size);
	initArray << <numBlocks, blockSize >> > (w_prev, size);
	initArray << <numBlocks, blockSize >> > (dens, size);
	initArray << <numBlocks, blockSize >> > (dens_prev, size);

	_coll->initialize_memory(N);
}
/* ---------------------------------------------------- */

// �ùķ��̼� �ʱ�ȭ
void sim_clear() {
	int size = (N + 2) * (N + 2) * (N + 2);
	size_t d_size = size * sizeof(double);
	cudaMemset(u, 0.0, d_size);
	cudaMemset(v, 0.0, d_size);
	cudaMemset(w, 0.0, d_size);
	cudaMemset(dens, 0.0, d_size);
	cudaMemset(u_prev, 0.0, d_size);
	cudaMemset(v_prev, 0.0, d_size);
	cudaMemset(w_prev, 0.0, d_size);
	cudaMemset(dens_prev, 0.0, d_size);
}

/* ------------------�ҽ��� �߰� �Լ�------------------ */
__global__ void setForceAndSource(double* d, double* v, int i1, int j1, int k1, double forceValue, int i2, int j2, int k2, double sourceValue) {
	v[CIX(i1, j1, k1)] = forceValue;
	d[CIX(i2, j2, k2)] = sourceValue;
}

void get_force_source(double* d, double* u, double* v, double* w) {
	int i, j, k, size = (N + 2) * (N + 2) * (N + 2);
	cudaMemset(u, 0, size * sizeof(double));
	cudaMemset(v, 0, size * sizeof(double));
	cudaMemset(w, 0, size * sizeof(double));
	cudaMemset(d, 0, size * sizeof(double));

	double forceValue;
	double sourceValue;

	if (addforce == 1) {
		i = (N + 2) / 2;
		j = 2;
		k = (N + 2) / 2;

		if (i < 1 || i > N || j < 1 || j > N) {
			std::cerr << "���� ���" << '\n';
			return;
		}

		forceValue = force * 3;
		sourceValue = source;
		setForceAndSource << <1, 1 >> > (d, v, i, j, k, forceValue, i, 10, k, sourceValue);
	}
}
/* --------------------------------------------------- */

/* ------------------------�浹 �ܷ� �߰� �Լ�------------------------ */
__global__ void set_collision_force(double* d, double* u, double* v, double* w, double* pu, double* pv, double* pw, int* d_calc, glm::vec3* cd, float* cv, int mx, int* id) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i <= N && j <= N && k <= N) {
		int cIdx = CIX(i, j, k);
		if (d_calc[cIdx] == 1) {
			d[cIdx] = 0.0;
			u[cIdx] = 0.0;
			v[cIdx] = 0.0;
			w[cIdx] = 0.0;
		}
		if (d_calc[cIdx] == 49) {
			int cId = id[cIdx] % mx;
			pu[cIdx] = 150 * cd[cId].x * cv[cId];
			pv[cIdx] = 150 * cd[cId].y * cv[cId];
			pw[cIdx] = 150 * cd[cId].z * cv[cId];
		}
	}
}

void get_collision_force() {
	dim3 blockDim(8, 8, 8);
	dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
	int bSize;
	glm::vec3 _cd[maxObject];
	float _cv[maxObject];
	for (bSize = 0; bSize < _bullet.size(); bSize++) {
		int id = _bullet[bSize]->_ID;
		_cd[id % maxObject] = _bullet[bSize]->_dir;
		_cv[id % maxObject] = _bullet[bSize]->_vel;
	}

	cudaMemcpy(OBJ_Dir, _cd, maxObject * sizeof(glm::vec3), cudaMemcpyHostToDevice);
	cudaMemcpy(OBJ_Vel, _cv, maxObject * sizeof(float), cudaMemcpyHostToDevice);

	set_collision_force<<<gridDim, blockDim>>>(dens, u, v, w, u_prev, v_prev, w_prev, _coll->d_calcCollision, OBJ_Dir, OBJ_Vel, maxObject, _coll->d_calcID);
	cudaDeviceSynchronize();
}
/* ------------------------------------------------------------------ */

// ���� �ܷ� �Լ�
__global__ void exp_force(int N, float size, double dx, double dy, double dz, double* u, double* v, double* w) {
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
	if (i < N && j < N && k < N) {
		int cIdx = CIX(i, j, k);
		double h = 1.0 / N;
		double x = (i - 0.5) * h + dx;
		double y = (j - 0.5) * h + dy;
		double z = (k - 0.5) * h + dz;

		glm::vec3 cell_center(x, y, z);
		glm::vec3 sphere_center(dx + 0.5f, dy + 0.5f, dz + 0.5f);

		// ��ü�� �� �߽��� ���� �Ÿ� ���
		float distance = glm::length(cell_center - sphere_center);

		// �浹 ����
		if (distance <= size) {
			if (x > sphere_center.x && y > sphere_center.y && z > sphere_center.z) {
				u[cIdx] = 500.0; v[cIdx] = 500.0; w[cIdx] = 500.0;
			}
			else if (x <= sphere_center.x && y > sphere_center.y && z > sphere_center.z) {
				u[cIdx] = -500.0; v[cIdx] = 500.0; w[cIdx] = 500.0;
			}
			else if (x > sphere_center.x && y > sphere_center.y && z <= sphere_center.z) {
				u[cIdx] = 500.0; v[cIdx] = 500.0; w[cIdx] = -500.0;
			}
			else if (x <= sphere_center.x && y > sphere_center.y && z <= sphere_center.z) {
				u[cIdx] = -500.0; v[cIdx] = 500.0; w[cIdx] = -500.0;
			}
			else if (x > sphere_center.x && y <= sphere_center.y && z > sphere_center.z) {
				u[cIdx] = 500.0; v[cIdx] = -500.0; w[cIdx] = 500.0;
			}
			else if (x <= sphere_center.x && y <= sphere_center.y && z > sphere_center.z) {
				u[cIdx] = -500.0; v[cIdx] = -500.0; w[cIdx] = 500.0;
			}
			else if (x > sphere_center.x && y <= sphere_center.y && z <= sphere_center.z) {
				u[cIdx] = 500.0; v[cIdx] = -500.0; w[cIdx] = -500.0;
			}
			else if (x <= sphere_center.x && y <= sphere_center.y && z <= sphere_center.z) {
				u[cIdx] = -500.0; v[cIdx] = -500.0; w[cIdx] = -500.0;
			}
		}
	}
}

// �ùķ��̼� ���� �Լ�
void sim_fluid() {
	get_force_source(dens_prev, u_prev, v_prev, w_prev);
	get_collision_force();
	if (boom) {
		dim3 blockDim(8, 8, 8);
		dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);
		exp_force << <gridDim, blockDim >> > (N, 0.17, drawX, drawY, drawZ, u_prev, v_prev, w_prev);
		boom = false;
	}
	_coll->divide_midCell(N);
	vel_step(N, u, v, w, u_prev, v_prev, w_prev, visc, dt, _coll->d_calcCollision);
	dens_step(N, dens, dens_prev, u, v, w, diff, dt, _coll->d_calcCollision);
	cudaDeviceSynchronize();
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_Z && action == GLFW_RELEASE) {
		addforce = (addforce == 0) ? 1 : 0;
		std::cout << "addforce : " << addforce << '\n';
	}

	if (key == GLFW_KEY_1 && action == GLFW_RELEASE) {
		mode = 0;
		std::cout << "mode : " << mode << '\n';
	}

	if (key == GLFW_KEY_2 && action == GLFW_RELEASE) {
		mode = 1;
		std::cout << "mode : " << mode << '\n';
	}

	if (key == GLFW_KEY_SPACE && action == GLFW_RELEASE) {
		simulation_stop = (simulation_stop == 0) ? 1 : 0;
		std::cout << "�ùķ��̼� ��� : " << simulation_stop << '\n';
	}

	if (key == GLFW_KEY_C && action == GLFW_RELEASE) {
		sim_clear();
		std::cout << "�ùķ��̼� �ʱ�ȭ" << '\n';
	}

	if (key == GLFW_KEY_M && action == GLFW_RELEASE) {
		_bullet.clear();
		objMode++;
		objMode = objMode % 3;
		if (objMode == 0) {
			bulletSize = 0.07f;
			bulletVel = 0.1f;
		}
		else if (objMode == 1) {
			bulletSize = 0.1f;
			bulletVel = 0.0f;
			glm::vec3 cameraPos = getCameraPosition();
			glm::vec3 cameraFront = getCameraDirection();
			glm::vec3 _dir(0.0f, 0.0f, 0.0f);
			float t = 2.0f;
			glm::vec3 _pos = cameraPos + t * cameraFront;
			glm::vec3 bInfo[2] = { _pos, _dir };
			_bullet.emplace_back(std::make_unique<Bullet>(N, bulletSize, bInfo, bulletVel, bulletID++));
		}
		else if (objMode == 2) {
			bulletSize = 0.17f;
			bulletVel = 0.0f;
			for (int i = 0; i < 2; i++) {
				glm::vec3 _dir = glm::vec3(0.0f, 0.0f, 0.0f);
				glm::vec3 _pos(drawX + i, 0.0f, 0.0f);
				glm::vec3 bInfo[2] = { _pos, _dir };
				_bullet.emplace_back(std::make_unique<Bullet>(N, bulletSize, bInfo, bulletVel, bulletID++));
			}
		}
		std::cout << "objMode : " << objMode << '\n';
	}
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS && _bullet.size() < maxObject) {
		if (objMode == 0) {
			//glm::vec3 testPos(0, 0, 2);
			//glm::vec3 testDir = testPos - glm::vec3(0,0,3);
			glm::vec3 _pos = getCameraPosition();		// ���� ī�޶� ��ġ
			glm::vec3 _dir = getCameraDirection();		// ���� ī�޶� �ٶ󺸴� ����
			//glm::vec3 _pos = testPos;
			//glm::vec3 _dir = testDir;
			_pos += (_dir * 1.0f);
			glm::vec3 bInfo[2] = { _pos, _dir };

			_bullet.emplace_back(std::make_unique<Bullet>(N, bulletSize, bInfo, bulletVel, bulletID++));
		}
		else if (objMode == 2) {
			dim3 blockDim(8, 8, 8);
			dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y, (N + blockDim.z - 1) / blockDim.z);

			boom = true;

			for (int i = 0; i < _bullet.size(); i++) {
				_bullet[i]->_dir = glm::normalize(_bullet[i]->_curr_pos - glm::vec3(0.0f, 0.0f, 0.0f));
				_bullet[i]->_vel = 0.3f;				
			}
		}
	}
}

int main() {
	// GLFW �ʱ�ȭ
	if (!glfwInit()) {
		std::cerr << "GLFW �ʱ�ȭ ����" << '\n';
		glfwTerminate();
		return -1;
	}
	glfwWindowHint(GLFW_SAMPLES, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 5);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	window = glfwCreateWindow(width, height, "3D Smoke Simulation GPU", NULL, NULL);
	if (window == NULL) {
		std::cerr << "GLFW �ʱ�ȭ ����" << '\n';
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	// GLEW �ʱ�ȭ
	glewExperimental = true;
	if (glewInit() != GLEW_OK) {
		std::cerr << "GLEW �ʱ�ȭ ����" << '\n';
		glfwTerminate();
		return -1;
	}

	// ���� �ʱ�ȭ 
	init_data();
	cudaDeviceSynchronize();

	// Ŭ���� �ʱ�ȭ
	_vel = new drawVelocity(N, drawX, drawY, drawZ);
	_den = new drawDensity(N, drawX, drawY, drawZ);

	// ���̴� �б�
	GLuint programID = LoadShaders("VertexShaderSL.txt", "FragmentShaderSL.txt");
	GLuint MatrixID = glGetUniformLocation(programID, "MVP");

	// ���콺 ����
	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
	glfwPollEvents();
	glfwSetCursorPos(window, width / 2, height / 2);

	GLuint VertexArrayID;
	glGenVertexArrays(1, &VertexArrayID);
	glBindVertexArray(VertexArrayID);

	// ȭ�� ���
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	do {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(programID);

		// ȭ�� �̵�, ��Ʈ�� control.h
		computeMatricesFromInputs(window, width, height);
		glm::mat4 ProjectionMatrix = getProjectionMatrix();
		glm::mat4 ViewMatrix = getViewMatrix();
		glm::mat4 ModelMatrix = glm::mat4(1.0);
		glm::mat4 MVP = ProjectionMatrix * ViewMatrix * ModelMatrix;
		glUniformMatrix4fv(MatrixID, 1, GL_FALSE, &MVP[0][0]);

		if (!simulation_stop) {
			cudaMemset(_coll->d_drawCollision, 0, N * N * N * sizeof(int));
			cudaMemset(_coll->d_calcCollision, 0, (N + 2) * (N + 2) * (N + 2) * sizeof(int));
			cudaMemset(_coll->d_calcID, 0, (N + 2) * (N + 2) * (N + 2) * sizeof(int));

			if (objMode == 0) {
				_bullet.erase(std::remove_if(_bullet.begin(), _bullet.end(),
					[](const std::unique_ptr<Bullet>& b) {
						b->drawBullet(drawX, drawY, drawZ);
						float breakLength = b->getLength();
						return breakLength > 5.0f || breakLength < -5.0f; // �� ������ ���̸� ���Ϳ��� ���ŵ˴ϴ�.
					}), _bullet.end());
			}
			else if (objMode == 1) {
				_bullet[0]->_dir = glm::vec3(0.0f, 0.0f, 0.0f);
				_bullet[0]->_vel = 0.0f;
				glm::vec3 cameraPos = getCameraPosition();
				glm::vec3 cameraFront = getCameraDirection();
				float t = 2.0;
				glm::vec3 _pos = cameraPos + t * cameraFront;
				_bullet[0]->_curr_pos = _pos;
				_bullet[0]->drawBullet(drawX, drawY, drawZ);
			}
			else if (objMode == 2) {
				_bullet.erase(std::remove_if(_bullet.begin(), _bullet.end(),
					[](const std::unique_ptr<Bullet>& b) {
						b->drawBullet(drawX, drawY, drawZ);
						float breakLength = b->getLength();
						return breakLength > 10.0f || breakLength < -10.0f; // �� ������ ���̸� ���Ϳ��� ���ŵ˴ϴ�.
					}), _bullet.end());
			}

			// �ùķ��̼� �ݺ�
			sim_fluid();
		}

		if (mode == 0) {
			glDepthMask(GL_FALSE);
			_den->draw_dens(N, dens);
			glDepthMask(GL_TRUE);
		}
		if (mode == 1) {
			glDepthMask(GL_FALSE);
			_vel->draw_vel(N, u, v, w);
			glDepthMask(GL_TRUE);
		}

		cudaDeviceSynchronize();

		glfwSwapBuffers(window);
		glfwPollEvents();
	} while ((glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && glfwWindowShouldClose(window) == 0));

	// ������ ����
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &VertexArrayID);
	free_data();
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}