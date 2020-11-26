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
        if (_density <= 0.f) {
            _density = 4.f;
        }
    }
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const override {
        if (_ntype == noise_type::PERLIN) {
            return vec3(1, 1, 1) * _noise.noise(p * _density);
        } else if (_ntype == noise_type::TURBULANCE) {
            //return vec3(0.7, 0.7, 0.7) * 0.5 * (1 + __sinf(12 * _noise.turbulance_noise(p / _density)));
            //return vec3(1.f) * 0.5 * (1 + __sinf(_noise.turbulance_noise(p * _density)));
            return vec3(1.f) * 0.5 * _noise.turbulance_noise(p * _density);
            //return vec3(1.f) * 0.5 * (1 + __sinf(5 * _noise.turbulance_noise(p * _density)));
        } else if (_ntype == noise_type::MARBLE) {
            //return vec3(1) * 0.5f * (1 + __sinf((p.z() * _density + 7 * _noise.turbulance_noise(p))));
            float value = 0.5f * (1 + __sinf((p.z() * _density + 7 * _noise.turbulance_noise(p))));
            //return vec3(0.349, 0.431, 0.498) 
                //(vec3(1,1,1) * 0.5f * (1 + __sinf((p.z() * _density +  7 * _noise.turbulance_noise(p)))));

            vec3 color1(0.925, 0.816, 0.78);
            vec3 color2(0.349/2, 0.431/2, 0.498/2);
            return color1 * value + color2 * (1 - value);
            //return (vec3(0.349, 0.431, 0.498) * value) + (vec3(0) * (1.f - value));
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

class image_texture : public text {
public:
    __device__ image_texture() {}
    __device__ image_texture(float* buffer, int width, int height)
        : _data(buffer), _width(width), _height(height) {}
    __device__ virtual inline vec3 value(float u, float v, const vec3& p) const override {
        //get_sphere_uv(p, u, v);
        int i = u * _width;
        int j = (1 - v) * _height - 0.001;
        if (i < 0) i = 0;
        if (j < 0) j = 0;
        if (i > _width - 1) i = _width - 1;
        if (j > _height - 1) j = _height - 1;

        int index = utils::XY(i, j, _width);
        float r = _data[index * 3 + 0];
        float g = _data[index * 3 + 1];
        float b = _data[index * 3 + 2];

        return vec3(r, g, b);
        /*if (p.x() > 1) {
            return vec3(1, 0, 0);
        }
        if (p.y() > 0.45) {
            return vec3(0, 0, 0);
        }
        if (p.y() < -0.3) {
            return vec3(0, 0, 0);
        }
        return vec3(1, 1, 1);*/
    }
private:
    float* _data;
    int _width = 0;
    int _height = 0;
};