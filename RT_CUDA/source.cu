#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include "kernel.h"
#include <conio.h>
#include <ctime>
#include <math.h>
#include "EasyBMP.hpp"

using namespace EasyBMP;

const int MAX_X = 13;
const int MAX_Y = 13;
const int MIN_Z = -50;
const int MAX_Z = -17;
const int MAX_RADIUS = 5;
const int MIN_RADIUS = 1;
const int PARAMS_FOR_OBJ = 8;
const int PARAMS_FOR_LIGHT = 11;
const int MAX_LIGHT = 3;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


__host__ void generateCursedScene(int objCount, int lightCount, sphere* spheres, bool fullReflection = false) {
	int totalParams = PARAMS_FOR_OBJ * objCount + PARAMS_FOR_LIGHT * lightCount;
	int totalSpheres = objCount + lightCount;
	float* params = (float*)calloc(totalParams, sizeof(float));
	curandGenerator_t gen;
	curandCreateGeneratorHost(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, time(0));
	curandGenerateUniform(gen, params, totalParams);
	int index = 0;
	for (int i = 0; i < totalSpheres; i++) {
		if (i < objCount) {
			index = i * PARAMS_FOR_OBJ;
		}
		else {
			index = objCount * PARAMS_FOR_OBJ + (i - objCount) * PARAMS_FOR_LIGHT;
		}
		vector3 pos(MAX_X * 2 * (params[index] - 0.5), MAX_Y * 2 * (params[index + 1] - 0.5), MIN_Z * params[index + 2] + MAX_Z);
		float radius = MAX_RADIUS * params[index + 3] + MIN_RADIUS;
		vector3 surfaceColor(params[index + 4], params[index + 5], params[index + 6]);
		int reflection = fullReflection ? 1 : roundf(params[index + 7]);
		reflection = (i < objCount) ? reflection : 0;
		vector3 emissionColor = (i < objCount) ? NULL : vector3(params[index + 8] * MAX_LIGHT, params[index + 9] * MAX_LIGHT, params[index + 10] * MAX_LIGHT);
		*(spheres + i) = sphere(pos, radius, surfaceColor, reflection, emissionColor);
	}
}


__host__ void preset_1(sphere* h_spheres, int& objCount, int& lightCount) {
	objCount = 6;
	lightCount = 2;
	h_spheres[0] = sphere(vector3(0.0, -7010, -20), 7000, vector3(0.20, 0.20, 0.20), 1);
	h_spheres[1] = sphere(vector3(0.0, 0, -20), 4, vector3(0.70, 0.52, 0.16), 0.8);
	h_spheres[2] = sphere(vector3(2.0, -1, -10), 2, vector3(0.50, 0.36, 0.76), 0.7);
	h_spheres[3] = sphere(vector3(5.0, 2, -30), 3, vector3(0.55, 0.97, 0.97), 0.5);
	h_spheres[4] = sphere(vector3(-1.5, -1, -40), 1, vector3(0.10, 0.20, 0.30), 0.3);
	h_spheres[5] = sphere(vector3(-5.5, 3, -15), 3, vector3(0.90, 0.90, 0.90), 0.05);
	h_spheres[6] = sphere(vector3(0.0, 5, -30), 70, vector3(0.00, 0.00, 0.00), 0, vector3(1));
	h_spheres[7] = sphere(vector3(0.0, 2, 30), 130, vector3(0.00, 0.00, 0.00), 0, vector3(0.2, 5, 0));
}

__host__ void preset_2(sphere* h_spheres, int& objCount, int& lightCount) {
	objCount = 5;
	lightCount = 1;
	h_spheres[0] = sphere(vector3(0.0, -5003, -20), 5000, vector3(0.20, 0.20, 0.20), 1);
	h_spheres[1] = sphere(vector3(0.0, 0, -20), 4, vector3(1.00, 0.82, 0.36), 1);
	h_spheres[2] = sphere(vector3(5.0, -1, -15), 2, vector3(0.90, 0.26, 0.32), 0.5);
	h_spheres[3] = sphere(vector3(5.0, 0, -25), 3, vector3(0.65, 0.77, 0.97), 0);
	h_spheres[4] = sphere(vector3(-5.5, 0, -15), 3, vector3(1, 1, 1), 1);
	h_spheres[5] = sphere(vector3(0.0, 20, -20), 3, vector3(0), 0, vector3(3));
}


void drawImage(vector3* image, int width, int height) {
	Image img(width, height, "example.bmp");
	for (int i = 0; i < width; ++i) {
		for (int j = 0; j < height; ++j) {
			int r = std::min(float(1), image[j * width + i].x) * 255;
			int g = std::min(float(1), image[j * width + i].y) * 255;
			int b = std::min(float(1), image[j * width + i].z) * 255;
			img.SetPixel(i, j, RGBColor(r, g, b));
		}
	}
	img.Write();
}

int main() {
	int objCount;
	int lightCount;
	int imageWidth;
	int imageHeight;

	printf("Number of spheres :");
	scanf("%d", &objCount);
	printf("Number of light sources :");
	scanf("%d", &lightCount);
	printf("Image Width:");
	scanf("%d", &imageWidth);
	printf("Image Height:");
	scanf("%d", &imageHeight);


	sphere* h_spheres;
	vector3* h_pixels;
	cudaMallocHost((void**)&h_spheres, (objCount + lightCount) * sizeof(sphere));
	cudaMallocHost((void**)&h_pixels, imageHeight * imageWidth * sizeof(vector3));

	generateCursedScene(objCount, lightCount, h_spheres, false);

	//preset_1(h_spheres,objCount,lightCount);
	//preset_2(h_spheres);

	float fov = 30;
	float aspectRatio = imageWidth / float(imageHeight);
	float angle = tan(acos(-1) * 0.5 * fov / 180.0);
	float time = 0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
	sphere* d_spheres;
	vector3* d_pixels;
	cudaMalloc((void**)&d_spheres, sizeof(sphere) * (objCount + lightCount));
	cudaMalloc((void**)&d_pixels, sizeof(vector3) * imageHeight * imageWidth);
	cudaMemcpy(d_spheres, h_spheres, (objCount + lightCount) * sizeof(sphere), cudaMemcpyHostToDevice);

	dim3 dimBlock = dim3(8, 8);
	int yBlocks = imageWidth / dimBlock.y + ((imageWidth % dimBlock.y) == 0 ? 0 : 1);
	int xBlocks = imageHeight / dimBlock.x + ((imageHeight % dimBlock.x) == 0 ? 0 : 1);
	dim3 dimGrid = dim3(xBlocks, yBlocks);


	kernel << <dimGrid, dimBlock >> > (d_spheres, objCount + lightCount, d_pixels, imageWidth, imageHeight, angle, aspectRatio);
	cudaDeviceSynchronize();




	cudaMemcpy(h_pixels, d_pixels, sizeof(vector3) * imageHeight * imageWidth, cudaMemcpyDeviceToHost);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("Elapsed time: %f ms\n", time);
	printf("Now drawing image \n");

	drawImage(h_pixels, imageWidth, imageHeight);
	return 0;
}