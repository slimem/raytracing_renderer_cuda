#pragma once

#include "common.h"

#ifdef __CUDA_ARCH__
#define USE_INTRINSICS
#endif

// A header only class for 3D vector that is used both by
// the host and the device.
// If it is used by the device, we use intrisics.
class vec3 {
public:
    __host__ __device__ vec3() {
        _v[0] = 0.f;
        _v[1] = 0.f;
        _v[2] = 0.f;
    }
    __host__ __device__ vec3(float v1, float v2, float v3) {
        _v[0] = v1;
        _v[1] = v2;
        _v[2] = v3;
    }
    
    // TODO: Use __saturatef to saturate float values
    __host__ __device__ constexpr bool is_null() const {
        // IEEE-754
        return (
            (_v[0] == 0.f)
            && (_v[1] == 0.f)
            && (_v[2] == 0.f)
            );
    }

    // the following methods just make the code more readable, but they
    // mostly do the same thing
    __host__ __device__ constexpr float u() const { return _v[0]; }
    __host__ __device__ constexpr float v() const { return _v[1]; }
    __host__ __device__ constexpr float w() const { return _v[2]; }
    __host__ __device__ constexpr float x() const { return _v[0]; }
    __host__ __device__ constexpr float y() const { return _v[1]; }
    __host__ __device__ constexpr float z() const { return _v[2]; }
    __host__ __device__ constexpr float r() const { return _v[0]; }
    __host__ __device__ constexpr float g() const { return _v[1]; }
    __host__ __device__ constexpr float b() const { return _v[2]; }

    // +vec3
    __host__ __device__ constexpr const vec3& operator+() const {
        return *this; 
    }
    
    // -vec3 (to negate)
    __host__ __device__ inline vec3 operator-() const {
        return vec3(-_v[0], -_v[1], -_v[2]);
    }

    // vect3[i]
    __host__ __device__ constexpr float operator[](int i) const {
        return _v[i];
    }

    __host__ __device__ constexpr float& operator[](int i) {
        return _v[i];
    };

    __host__ __device__ constexpr vec3& operator+=(const vec3& v) {
#ifdef USE_INTRINSICS
        _v[0] = __fadd_rz(_v[0], v._v[0]);
        _v[1] = __fadd_rz(_v[1], v._v[1]);
        _v[2] = __fadd_rz(_v[2], v._v[2]);
#else
        _v[0] += v._v[0];
        _v[1] += v._v[1];
        _v[2] += v._v[2];
#endif
        return *this;
    }

    __host__ __device__ constexpr vec3& operator-=(const vec3& v) {
#ifdef USE_INTRINSICS
        _v[0] = __fsub_rz(_v[0], v._v[0]);
        _v[1] = __fsub_rz(_v[1], v._v[1]);
        _v[2] = __fsub_rz(_v[2], v._v[2]);
#else
        _v[0] -= v._v[0];
        _v[1] -= v._v[1];
        _v[2] -= v._v[2];
#endif
        return *this;
    }

    __host__ __device__ constexpr vec3& operator*=(const vec3& v) {
#ifdef USE_INTRINSICS
        _v[0] = __fmul_rz(_v[0], v._v[0]);
        _v[1] = __fmul_rz(_v[1], v._v[1]);
        _v[2] = __fmul_rz(_v[2], v._v[2]);
#else
        _v[0] *= v._v[0];
        _v[1] *= v._v[1];
        _v[2] *= v._v[2];
#endif
        return *this;
    }

    __host__ __device__ constexpr vec3& operator/=(const vec3& v) {
#ifdef USE_INTRINSICS
        _v[0] = __fdiv_rz(_v[0], v._v[0]);
        _v[1] = __fdiv_rz(_v[1], v._v[1]);
        _v[2] = __fdiv_rz(_v[2], v._v[2]);
#else
        _v[0] /= v._v[0];
        _v[1] /= v._v[1];
        _v[2] /= v._v[2];
#endif
        return *this;
    }

    __host__ __device__ constexpr vec3& operator*=(const float f) {
#ifdef USE_INTRINSICS
        _v[0] = __fmul_rz(_v[0], f);
        _v[1] = __fmul_rz(_v[1], f);
        _v[2] = __fmul_rz(_v[2], f);
#else
        _v[0] *= f;
        _v[1] *= f;
        _v[2] *= f;
#endif
        return *this;
    }

    __host__ __device__ constexpr vec3& operator/=(const float f) {
#ifdef USE_INTRINSICS
        float u = __fdiv_rz(1.0f, f);
        _v[0] = __fmul_rz(_v[0], u);
        _v[1] = __fmul_rz(_v[1], u);
        _v[2] = __fmul_rz(_v[2], u);
#else
        float u = 1.0f / f;
        _v[0] *= u;
        _v[1] *= u;
        _v[2] *= u;
#endif
        return *this;
    }

    __host__ __device__ constexpr float length() const {
#ifdef USE_INTRINSICS
        return
            __fsqrt_rz(
                __fmul_rz(_v[0], _v[0])
                + __fmul_rz(_v[1], _v[1])
                + __fmul_rz(_v[2], _v[2])    
            );
#else
        return sqrt(
            _v[0] * _v[0] + _v[1] * _v[1] + _v[2] * _v[2]
        );
#endif
    }

    __host__ __device__ constexpr float sq_length() const {
#ifdef USE_INTRINSICS
        return 
            (
                __fmul_rz(_v[0], _v[0])
                + __fmul_rz(_v[1], _v[1])
                + __fmul_rz(_v[2], _v[2])
            );
#else
        return (_v[0] * _v[0] + _v[1] * _v[1] + _v[2] * _v[2]);
#endif
    }

    __host__ __device__ inline vec3 gamma_correct() const {
#ifdef USE_INTRINSICS
        return vec3(
            __fsqrt_rz(_v[0]),
            __fsqrt_rz(_v[1]),
            __fsqrt_rz(_v[2])
        );
#else
        return vec3(sqrt(_v[0]), sqrt(_v[1]), sqrt(_v[2]));
#endif
    }

    __host__ __device__ constexpr void normalize() {
        if (!is_null()) {
            *this /= this->length();
        }
    }

    __host__ __device__ static inline vec3 normalize(vec3 v) {
        if (v.is_null()) {
            return v;
        } else {
            return v / v.length();
        }
    }

    // dot product
    __host__ __device__ static constexpr float dot(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
        return
            (
                __fmul_rz(v1._v[0], v2._v[0])
                + __fmul_rz(v1._v[1], v2._v[1])
                + __fmul_rz(v1._v[2], v2._v[2])
            );
#else
        return v1._v[0] * v2._v[0] + v1._v[1] * v2._v[1] + v1._v[2] * v2._v[2];
#endif
    }

    __host__ __device__ static inline vec3 cross(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fsub_rz(
                    __fmul_rz(v1._v[1], v2._v[2]),
                    __fmul_rz(v1._v[2], v2._v[1])
                ),
                __fmul_rz(
                    __fsub_rz(
                    __fmul_rz(v1._v[0], v2._v[2]),
                    __fmul_rz(v1._v[2], v2._v[0])
                    ),
                    -1
                ),
                __fsub_rz(
                    __fmul_rz(v1._v[0], v2._v[1]),
                    __fmul_rz(v1._v[1], v2._v[0])
                )
            );
#else
        return vec3((v1._v[1] * v2._v[2] - v1._v[2] * v2._v[1]),
            (-(v1._v[0] * v2._v[2] - v1._v[2] * v2._v[0])),
            (v1._v[0] * v2._v[1] - v1._v[1] * v2._v[0]));
#endif
    }

    friend std::ostream& operator<<(std::ostream& os, const vec3& v) {
        os << "(" << v._v[0] << "," << v._v[1] << "," << v._v[2] << ")";
        return os;
    }

    friend std::istream& operator>>(std::istream& is, vec3& v) {
        is >> v._v[0] >> v._v[1] >> v._v[2];
        return is;
    }

    __host__ __device__ friend inline vec3 operator+(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fadd_rz(v1._v[0], v2._v[0]),
                __fadd_rz(v1._v[1], v2._v[1]),
                __fadd_rz(v1._v[2], v2._v[2])
            );
#else
        return vec3(v1._v[0] + v2._v[0], v1._v[1] + v2._v[1], v1._v[2] + v2._v[2]);
#endif
    }

    __host__ __device__ friend inline vec3 operator-(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fsub_rz(v1._v[0], v2._v[0]),
                __fsub_rz(v1._v[1], v2._v[1]),
                __fsub_rz(v1._v[2], v2._v[2])
            );
#else
        return vec3(v1._v[0] - v2._v[0], v1._v[1] - v2._v[1], v1._v[2] - v2._v[2]);
#endif
    }

    __host__ __device__ friend inline vec3 operator*(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fmul_rz(v1._v[0], v2._v[0]),
                __fmul_rz(v1._v[1], v2._v[1]),
                __fmul_rz(v1._v[2], v2._v[2])
            );
#else
        return vec3(v1._v[0] * v2._v[0], v1._v[1] * v2._v[1], v1._v[2] * v2._v[2]);
#endif
    }

    __host__ __device__ friend inline vec3 operator/(const vec3& v1, const vec3& v2) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fdiv_rz(v1._v[0], v2._v[0]),
                __fdiv_rz(v1._v[1], v2._v[1]),
                __fdiv_rz(v1._v[2], v2._v[2])
            );
#else
        return vec3(v1._v[0] / v2._v[0], v1._v[1] / v2._v[1], v1._v[2] / v2._v[2]);
#endif
    }

    __host__ __device__ friend inline vec3 operator*(float t, const vec3& v) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fmul_rz(v._v[0], t),
                __fmul_rz(v._v[1], t),
                __fmul_rz(v._v[2], t)
            );
#else
        return vec3(t * v._v[0], t * v._v[1], t * v._v[2]);
#endif
    }

    __host__ __device__ friend inline vec3 operator*(const vec3& v, float t) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fmul_rz(v._v[0], t),
                __fmul_rz(v._v[1], t),
                __fmul_rz(v._v[2], t)
            );
#else
        return vec3(t * v._v[0], t * v._v[1], t * v._v[2]);
#endif
    }

    __host__ __device__ friend inline vec3 operator/(vec3 v, float t) {
#ifdef USE_INTRINSICS
        return
            vec3(
                __fdiv_rz(v._v[0], t),
                __fdiv_rz(v._v[1], t),
                __fdiv_rz(v._v[2], t)
            );
#else
        return vec3(v._v[0] / t, v._v[1] / t, v._v[2] / t);
#endif
    }

    __device__ static inline vec3 reflect(const vec3& v, const vec3& n) {
        return v - 2.f * dot(v, n) * n;
    }


    __device__ inline vec3 saturate() const {
#ifdef USE_INTRINSICS
        return
            vec3(
                __saturatef(_v[0]),
                __saturatef(_v[1]),
                __saturatef(_v[2])
            );
#else
        return vec3(-1.f, -1.f, -1.f);
#endif
    }

private:
    float _v[3];
};



