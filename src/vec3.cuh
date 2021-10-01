#pragma once
#include <cmath>
#include <stdlib.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef PI
#define PI 3.14159265
#endif

class vec3 {
public:
    __host__ __device__
        vec3() {}

    __host__ __device__
        vec3(double e0, double e1, double e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    __host__ __device__
        inline double x() const { return e[0]; }

    __host__ __device__
        inline double y() const { return e[1]; }

    __host__ __device__
        inline double z() const { return e[2]; }

    __host__ __device__
        inline double r() const { return e[0]; }

    __host__ __device__
        inline double g() const { return e[1]; }

    __host__ __device__
        inline double b() const { return e[2]; }

    __host__ __device__
        inline const vec3& operator+() const { return *this; }

    __host__ __device__
        inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

    __host__ __device__
        inline double operator[](int i) const { return e[i]; }

    __host__ __device__
        inline double& operator[](int i) { return e[i]; }

    __host__ __device__
        inline vec3& operator+=(const vec3& v2);

    __host__ __device__
        inline vec3& operator-=(const vec3& v2);

    __host__ __device__
        inline vec3& operator*=(const vec3& v2);

    __host__ __device__
        inline vec3& operator/=(const vec3& v2);

    __host__ __device__
        inline vec3& operator*=(const double t);

    __host__ __device__
        inline vec3& operator/=(const double t);

    __host__ __device__
        inline double length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }

    __host__ __device__
        inline double squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    __host__ __device__
        inline void make_unit_vector();

    double e[3];
};

__host__
inline std::istream& operator>>(std::istream& is, vec3& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

__host__
inline std::ostream& operator<<(std::ostream& os, const vec3& t) {
    os << "[" << t.e[0] << ", " << t.e[1] << ", " << t.e[2] << "]";
    return os;
}

__host__ __device__
inline void vec3::make_unit_vector() {
    double k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__
inline vec3 operator+(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__
inline vec3 operator-(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__
inline vec3 operator*(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__
inline vec3 operator/(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__
inline vec3 operator*(double t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__
inline vec3 operator/(vec3 v, double t) {
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__
inline vec3 operator*(const vec3& v, double t) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__
inline double dot(const vec3& v1, const vec3& v2) {
    return v1.e[0] * v2.e[0]
        + v1.e[1] * v2.e[1]
        + v1.e[2] * v2.e[2];
}

__host__ __device__
inline vec3 cross(const vec3& v1, const vec3& v2) {
    return vec3(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
        v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
        v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

__host__ __device__
inline vec3& vec3::operator+=(const vec3& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__
inline vec3& vec3::operator*=(const vec3& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__
inline vec3& vec3::operator/=(const vec3& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__
inline vec3& vec3::operator-=(const vec3& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__
inline vec3& vec3::operator*=(const double t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__
inline vec3& vec3::operator/=(const double t) {
    double k = 1.0f / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__
inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

__host__ __device__
inline double determinant(const double col1[], const double col2[])
{
    return col1[0] * col2[1] - col2[0] * col1[1];
}

__host__ __device__
inline double determinant(const vec3& col1, const vec3& col2, const vec3& col3)
{

    return (col1[0] * col2[1] * col3[2] + col2[0] * col3[1] * col1[2] - col3[0] * col1[1] * col2[2] - col3[0] * col2[1] * col1[2]
        - col2[0] * col1[1] * col3[2] - col1[0] * col3[1] * col2[2]);
}

__host__ __device__
inline vec3 rot_about_z(vec3& v, double theta) {
    vec3 a, b, c;
    a = vec3(cos(theta), sin(theta), 0);
    b = vec3(-sin(theta), cos(theta), 0);
    c = vec3(0, 0, 1);
    return (v.x() * a + v.y() * b + v.z() * c);
}

__host__ __device__
inline vec3 rot_about_x(vec3& v, double theta) {
    vec3 a, b, c;
    a = vec3(1, 0, 0);
    b = vec3(0, cos(theta), sin(theta));
    c = vec3(0, -sin(theta), cos(theta));
    return (v.x() * a + v.y() * b + v.z() * c);
}

__host__ __device__
inline vec3 rot_about_y(vec3& v, double theta) {
    vec3 a, b, c;
    a = vec3(cos(theta), 0, -sin(theta));
    b = vec3(0, 1, 0);
    c = vec3(sin(theta), 0, cos(theta));
    return (v.x() * a + v.y() * b + v.z() * c);
}

__host__ __device__
inline vec3 apply_rot(vec3& v, vec3& axis) {
    vec3 x_rot, xy_rot, xyz_rot;
    x_rot = rot_about_x(v, axis.x());
    xy_rot = rot_about_y(x_rot, axis.y());
    xyz_rot =  rot_about_z(xy_rot, axis.z());
    return xyz_rot;
}

