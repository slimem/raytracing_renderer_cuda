#pragma once

#include "hitable_object.h"
#include "utils.h"

// abstract material class
class material {
public:
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const = 0;
};

// create a diffuse material
class lambertian : public material {
public:
    __device__ lambertian(const vec3& a) : _albedo(a) {}
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const override;

private:
    vec3 _albedo;
};

class metal : public material {
public:
    __device__ metal(const vec3& a, float r)
     : _albedo(a) {
        if (r < 1.f) {
            _roughness = r;
        } else {
            _roughness = 1.f;
        }
    }
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const override;

public:
    vec3 _albedo;
    float _roughness = 0.f;
};

__device__ bool
lambertian::scatter(const ray& rin, ray& rout,
    const hit_record& hrec,
    vec3& attenuation,
    curandState* rstate) const {

    // select a random vector at the hit point
    vec3 target = hrec.p() + hrec.n() + utils::random_point_unit_sphere(rstate);
    rout = ray(hrec.p(), target - hrec.p());
    attenuation = _albedo;
    return true;
}

__device__ bool
metal::scatter(const ray& rin, ray& rout,
    const hit_record& hrec,
    vec3& attenuation,
    curandState* rstate) const {

    vec3 reflection = vec3::reflect(vec3::normalize(rin.direction()), hrec.n());
    rout = ray(hrec.p(), reflection + _roughness * utils::random_point_unit_sphere(rstate));
    attenuation = _albedo;

    // check if > 90 degress between the ray and the surface normal (negative dot product)
    float dot = vec3::dot(rout.direction(), hrec.n());
    return (dot > 0.f);
}