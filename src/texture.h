#pragma once

#include "vec3.h"
#include "perlin_noise.h"

enum class noise_type : uint8_t {
    PERLIN,
    TURBULANCE,
    MARBLE,
    UNKNOWN
};

class text {
public:
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const = 0;
};

class constant_texture : public text {
public:
    __device__ constant_texture() {}
    __device__ constant_texture(vec3 col) : _col(col) {}
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const override {
        return _col;
    }
private:
    vec3 _col;
};

class checker_texture : public text {
public:
    __device__ checker_texture() {}
    __device__ checker_texture(const text* t0, const text* t1)
        : _even(t0), _odd(t1) {}
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const override;
private:
    const text* _even = nullptr;
    const text* _odd = nullptr;
};

__device__ inline vec3
checker_texture::value(float u, float v, const vec3& p) const {
    float sines = __sinf(10 * p.x()) * __sinf(10 * p.y()) * __sinf(10 * p.z());
    if (sines < 0.f) {
        return _odd->value(u, v, p);
    } else {
        return _even->value(u, v, p);
    }
}

class noise_texture : public text {
public:
    __device__ noise_texture(noise_type ntype = noise_type::PERLIN, float density = 4.f)
        : _density(density), _ntype(ntype) {
        if (_density <= 0.f) { //avoid division by zero
            _density = 4.f;
        }
    }
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const override {
        if (_ntype == noise_type::PERLIN) {
            return vec3(1, 1, 1) * _noise.noise(p / _density);
        } else if (_ntype == noise_type::TURBULANCE) {
            return vec3(0.7, 0.7, 0.7) * 0.5 * (1 + __sinf(12 * _noise.turbulance(p / _density)));
        } else if (_ntype == noise_type::MARBLE) {
            return vec3(1, 1, 1) * 0.5f * (1 + __sinf((p.z() / _density +  10 * _noise.turbulance(p))));
        } else {
            return vec3(1, 1, 1);
        }
    }
private:
    perlin_noise _noise;
    float _density;
    noise_type _ntype;
};

class wood_texture : public text {
public:
    __device__ wood_texture(
        const vec3& color1,
        const vec3& color2,
        float density = 4.f,
        float hardness = 50.f
    ) : _color1(color1), _color2(color2), _density(density), _hardness(hardness)
    {
        if (_density <= 0.f) { //avoid division by zero
            _density = 4.f;
        }
    }
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const override {
        float n = _hardness * _noise.noise(vec3(p.x(), p.y(), p.z()) / _density);
        // clamp value to [0, 1[
        n -= floorf(n);
        return (_color1 * n) + (_color2 * (1.f - n));
    }
private:
    float _density;
    float _hardness;
    vec3 _color1;
    vec3 _color2;
    perlin_noise _noise;
};