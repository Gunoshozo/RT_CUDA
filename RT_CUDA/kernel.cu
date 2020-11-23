#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include "Sphere.h"
#include "kernel.h"

#define MAX_RECURSION_DEPTH 5

__device__ vector3 trace(const vector3& origin, const vector3& direction, sphere* spheres, const int spheresCount, const int& depth) {
	float tClosest = INFINITY;
	const sphere* sphere = NULL;
	for (unsigned i = 0; i < spheresCount; ++i) {
		float t0 = INFINITY, t1 = INFINITY;
		if (spheres[i].intersect(origin, direction, t0, t1)) {
			if (t0 < 0)
				t0 = t1;
			if (t0 < tClosest) {
				tClosest = t0;
				sphere = &spheres[i];
			}
		}
	}

	if (!sphere) 
		return vector3(1);
	vector3 surfaceColor(0);
	vector3 hitPoint = origin + direction * tClosest;
	vector3 normHitPoint = hitPoint - sphere->center;
	normHitPoint.normalize();

	float bias = 1e-4;
	if (direction.dot(normHitPoint) > 0)
		normHitPoint = -normHitPoint;
	if ((sphere->reflection > 0) && depth < MAX_RECURSION_DEPTH) {
		float faceRatio = -direction.dot(normHitPoint);
		float reflectionCoef = 0.9 * (1 - faceRatio) * (1 - faceRatio) * (1 - faceRatio) + 0.1;
		vector3 reflDirection = direction - normHitPoint * 2 * direction.dot(normHitPoint);
		reflDirection.normalize();
		vector3 reflection = trace(hitPoint + normHitPoint * bias, reflDirection, spheres, spheresCount, depth + 1);
		surfaceColor = (reflection * reflectionCoef) * sphere->surfaceColor;
	}
	else {
		for (unsigned i = 0; i < spheresCount; ++i) {
			if (spheres[i].emissionColor.length() > 0) {
				vector3 doTransmit = 1;
				vector3 lightDirection = spheres[i].center - hitPoint;
				lightDirection.normalize();
				for (unsigned j = 0; j < spheresCount; ++j) {
					if (i != j) {
						float t0, t1;
						if (spheres[j].intersect(hitPoint + normHitPoint * bias, lightDirection, t0, t1)) {
							doTransmit = 0;
							break;
						}
					}
				}
				surfaceColor += sphere->surfaceColor * doTransmit * fmax(float(0), normHitPoint.dot(lightDirection)) * spheres[i].emissionColor;
			}
		}
	}

	return surfaceColor + sphere->emissionColor;
}


__global__ void kernel(sphere* spheres, const int spheresCount, vector3* pixels, int width, int height, float angle, float aspectRatio) {
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	float x = (2 * ((col + 0.5) / width) - 1) * angle * aspectRatio;
	float y = (1 - 2 * ((row + 0.5) / height)) * angle;
	vector3 direction(x, y, -1);
	direction.normalize();
	pixels[row * width + col] = trace(vector3(0), direction, spheres, spheresCount, 0);
}