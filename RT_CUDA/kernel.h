#include"Sphere.h"

__device__ vector3 trace(const vector3& origin, const vector3& direction, sphere* spheres, const int spheresCount, const int& depth);

__global__ void kernel(sphere* spheres, const int spheresCount, vector3* pixels, int width, int height, float angle, float aspectRatio);