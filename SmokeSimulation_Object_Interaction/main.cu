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

// �ùķ��̼� ��ġ
double drawX = -0.5;
double drawY = -0.5;
double drawZ = -0.5;

// �Ѿ��� ũ��� �ӵ�
float bulletSize = 0.07f;
float bulletVel = 0.1f;

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


// �ùķ��̼� ���� �Լ�
void sim_fluid() {
	get_force_source(dens_prev, u_prev, v_prev, w_prev);
	vel_step(N, u, v, w, u_prev, v_prev, w_prev, visc, dt);
	dens_step(N, dens, dens_prev, u, v, w, diff, dt);
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
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
		glm::vec3 testPos(0, 0, 2);
		glm::vec3 testDir = testPos - glm::vec3(0,0,3);
		//glm::vec3 _pos = getCameraPosition();		// ���� ī�޶� ��ġ
		//glm::vec3 _dir = getCameraDirection();		// ���� ī�޶� �ٶ󺸴� ����
		glm::vec3 _pos = testPos;
		glm::vec3 _dir = testDir;
		//_pos += (_dir * 1.0f);
		glm::vec3 bInfo[2] = { _pos, _dir };
		_bullet.emplace_back(std::make_unique<Bullet>(N, bulletSize, bInfo, bulletVel));
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
			cudaMemset(CollisionObject::d_drawCollision, 0, N * N * N * sizeof(int));
			_bullet.erase(std::remove_if(_bullet.begin(), _bullet.end(),
				[](const std::unique_ptr<Bullet>& b) {
					b->drawBullet(drawX, drawY, drawZ);
					float breakLength = b->getLength();
					return breakLength > 5.0f || breakLength < -5.0f; // �� ������ ���̸� ���Ϳ��� ���ŵ˴ϴ�.
				}), _bullet.end());

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