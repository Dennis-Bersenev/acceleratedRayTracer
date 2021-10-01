#pragma once
#include "ray.cuh"

class camera {
public:

	float d;
	unsigned int height, width;
	vec3 cam_origin, u, v, w;

	__device__ 
	camera(float aspect_ratio) 
	{
		float horiz = 2.f * aspect_ratio;
		lower_left_corner = vec3(-aspect_ratio, -1.f, -1.f);
		horizontal = vec3(horiz, 0.f, 0.f);
		vertical = vec3(0.f, 2.f, 0.f);
		origin = vec3(0.f, 0.f, 0.f);
	}
	__device__ 
	ray get_ray(float u, float v) 
	{ 
		return ray(origin, lower_left_corner + u * horizontal + v * vertical - origin); 
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;

	__host__
	camera(float hfov, unsigned int width, unsigned int height)
	{
		this->width = width;
		this->height = height;
		float a = hfov / 2.f;
		d = ((0.5f * width) / (float)tan(a));
		u = vec3(1, 0, 0);
		v = vec3(0, 1, 0);
		w = vec3(0, 0, 1);
		cam_origin = vec3(0, 0, -d);
	}

	__host__
	camera(float hfov, unsigned int width, unsigned int height, vec3& origin)
	{
		this->width = width;
		this->height = height;
		float a = hfov / 2.f;
		d = width / (float)tan(a);
		u = vec3(1, 0, 0);
		v = vec3(0, 1, 0);
		w = vec3(0, 0, 1);
		cam_origin = vec3(origin.x(), origin.y(), origin.z() - d);
	}

	__host__
	camera(float hfov, unsigned int width, unsigned int height, vec3& origin, vec3& rots)
	{
		this->width = width;
		this->height = height;
		float a = hfov / 2.f;
		d = width / (float)tan(a);
		u = apply_rot(vec3(1, 0, 0), rots);
		v = apply_rot(vec3(0, 1, 0), rots);
		w = apply_rot(vec3(0, 0, 1), rots);
		cam_origin = vec3(origin.x(), origin.y(), origin.z() - d);
	}
};
