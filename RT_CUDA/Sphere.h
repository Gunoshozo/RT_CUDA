#pragma once
#include <cmath>
#include "vector3.h"

__device__ __host__ class sphere
{
public:
	vector3 center;
	float radius, radiusSquared;
	vector3 surfaceColor, emissionColor;
	float  reflection;
	__device__ __host__ sphere();
	__device__ __host__ sphere(
		const vector3& c,
		const float& r,
		const vector3& sc,
		const float& refl = 0,
		const vector3& ec = 0) :
		center(c), radius(r), radiusSquared(r* r), surfaceColor(sc), emissionColor(ec),
		reflection(refl) {}

	__device__ __host__ bool intersect(const vector3& rayOrig, const vector3& rayDir, float& hitPoint1, float& hitPoint2) const
	{
		vector3 rayToCenter = center - rayOrig;
		float projRayToCenter = rayToCenter.dot(rayDir);
		if (projRayToCenter < 0)
			return false;
		float distTo—hord = rayToCenter.dot(rayToCenter) - projRayToCenter * projRayToCenter;
		if (distTo—hord > radiusSquared)
			return false;
		float chordHalf = sqrt(radiusSquared - distTo—hord);
		hitPoint1 = projRayToCenter - chordHalf;
		hitPoint2 = projRayToCenter + chordHalf;
		return true;
	}
};