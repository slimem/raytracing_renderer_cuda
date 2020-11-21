#pragma once

#include "vec3.h"

// This perlin noise implementation is inspired by Ken Perlin's from 2002
// You can find it here (in java) https://cs.nyu.edu/~perlin/noise/
// the advantage of this implementation is that it does not use a random
// number generator and everything is initialized on the stack.
class perlin_noise {
public:
    __device__ perlin_noise();
    __device__ float noise(const vec3& p) const;
private:
    __device__ constexpr float fade(float t) const;
    __device__ constexpr float lerp(float t, float a, float b) const;
    __device__ constexpr float grad(uint16_t hash, float x, float y, float z) const;

    // so we wont need to use a curand state
    uint8_t permutation[256] = {
        151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,
        8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,
        35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,
        134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
        55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,
        18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,
        250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,
        189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167,
        43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,
        97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,
        107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
        138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
    };
    uint8_t p[512];
};

__device__
perlin_noise::perlin_noise() {
    for (int i = 0; i < 256; ++i) p[256 + i] = p[i] = permutation[i];
}

__device__ float
perlin_noise::noise(const vec3& point) const {

    float xf = point.x();
    float yf = point.y();
    float zf = point.z();

    uint8_t xi = (int)floorf(point.x()) & 255;
    uint8_t yi = (int)floorf(point.y()) & 255;
    uint8_t zi = (int)floorf(point.z()) & 255;

    // find relative xyz in the cube
    xf -= floorf(xf);
    yf -= floorf(yf);
    zf -= floorf(zf);

    // compute fade curves for xyz
    float u = fade(xf);
    float v = fade(yf);
    float w = fade(zf);

    // hash coordinates of the 8 cube corners
    uint16_t A = p[xi] + yi;
    uint16_t AA = p[A] + zi;
    uint16_t AB = p[A + 1] + zi;
    uint16_t B = p[xi + 1] + yi;
    uint16_t BA = p[B] + zi;
    uint16_t BB = p[B + 1] + zi;

    // add blended results from 8 corners of cube
    float res = lerp
    (
        w,
        lerp(
            v,
            lerp(
                u,
                grad(p[AA], xf, yf, zf),
                grad(p[BA], xf - 1, yf, zf)
            ),
            lerp(
                u,
                grad(p[AB], xf, yf - 1, zf),
                grad(p[BB], xf - 1, yf - 1, zf)
            )
        ),
        lerp(
            v,
            lerp(
                u,
                grad(p[AA + 1], xf, yf, zf - 1),
                grad(p[BA + 1], xf - 1, yf, zf - 1)),
            lerp(
                u,
                grad(p[AB + 1], xf, yf - 1, zf - 1),
                grad(p[BB + 1], xf - 1, yf - 1, zf - 1)
            )
        )
    );
    return (res + 1.0f) / 2.0f;
}

__device__ constexpr float
perlin_noise::fade(float t) const {
    // improved noise fade by Ken Perlin that uses the equation
    // 6t^5 - 15t^4 + 10t^3
    // check the following paper https://mrl.cs.nyu.edu/~perlin/paper445.pdf
    // this is a simplified version that uses less pow(x,y) calls
    // TODO: use cuda intrinsics
    // use __powf
    return t * t * t * (t * (t * 6 - 15) + 10);
}

__device__ constexpr float
perlin_noise::lerp(float t, float a, float b) const {
    // TODO: use cuda intrinsics
    return a + t * (b - a);
}

__device__ constexpr float
perlin_noise::grad(uint16_t hash, float x, float y, float z) const {
    // from https://cs.nyu.edu/~perlin/noise/ (implemented in java)
    uint8_t h = hash & 15;
    // convert lower 4 bits of hash into 12 gradient directions
    float u = h < 8 ? x : y,
        v = h < 4 ? y : h == 12 || h == 14 ? x : z;
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
}