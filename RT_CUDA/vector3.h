#pragma once
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__device__ __host__ class vector3
{
public:
	float x, y, z;
	__device__ __host__ vector3() : x(0), y(0), z(0) {}
	__device__ __host__ vector3(float x_) : x(x_), y(x_), z(x_) {}
	__device__ __host__ vector3(float x_, float yy, float z_) : x(x_), y(yy), z(z_) {}
	__device__ __host__ vector3& normalize()
	{
		float nor2 = length2();
		if (nor2 > 0) {
			float invNor = 1 / sqrt(nor2);
			x *= invNor, y *= invNor, z *= invNor;
		}
		return *this;
	}
	__device__ __host__ vector3 operator * (const float& f) const {
		return vector3(x * f, y * f, z * f);
	}

	__device__ __host__ vector3 operator* (const vector3& v) const {
		return vector3(x * v.x, y * v.y, z * v.z);
	}

	__device__ __host__ float dot(const vector3& v) const {
		return x * v.x + y * v.y + z * v.z;
	}

	__device__ __host__ vector3 operator- (const vector3& v) const {
		return vector3(x - v.x, y - v.y, z - v.z);
	}

	__device__ __host__ vector3 operator+ (const vector3& v) const {
		return vector3(x + v.x, y + v.y, z + v.z);
	}
	__device__ __host__ vector3& operator+= (const vector3& v) {
		x += v.x, y += v.y, z += v.z;
		return *this;
	}
	__device__ __host__ vector3& operator*= (const vector3& v) {
		x *= v.x, y *= v.y, z *= v.z;
		return *this;
	}
	__device__ __host__ vector3 operator- () const {
		return vector3(-x, -y, -z);
	}
	__device__ __host__ float length2() const {
		return x * x + y * y + z * z;
	}
	__device__ __host__ float length() const {
		return sqrt(length2());
	}

};