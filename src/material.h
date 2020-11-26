#pragma once

#include "hitable_object.h"
#include "utils.h"
#include "texture.h"

// abstract material class
class material {
public:
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const = 0;
    __device__ virtual vec3 emit() const {
        return vec3(0.f, 0.f, 0.f);
    }
};

class emitter : public material {
public:
    __device__ emitter(const vec3& intensity)
     : _intensity(intensity) {}
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const override {
        return false;
    }
    
    __device__ virtual vec3 emit() const override {
        return _intensity;
    }

private:
    vec3 _intensity;
};

// create a diffuse material
class lambertian : public material {
public:
    __device__ lambertian(const text* tex) : _albedo(tex) {}
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const override;

private:
    // albedo is latin for whiteness
    const text* _albedo = nullptr;
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

class dielectric : public material {
public:
    __device__ dielectric(float ri, const vec3& tint) : _ri(ri), _tint(tint) {}
    __device__ virtual bool scatter(const ray& rin, ray& rout,
        const hit_record& hrec,
        vec3& attenuation,
        curandState* rstate) const override;

private:
    float _ri = 0.f;
    vec3 _tint;
};

__device__ bool
lambertian::scatter(const ray& rin, ray& rout,
    const hit_record& hrec,
    vec3& attenuation,
    curandState* rstate) const {

    // select a random vector at the hit point
    vec3 target = hrec.p() + hrec.n() + utils::random_point_unit_sphere(rstate);
    rout = ray(hrec.p(), target - hrec.p(), rin.t());
    attenuation = _albedo->value(hrec.u(), hrec.v(), hrec.p());
    return true;
}

__device__ bool
metal::scatter(const ray& rin, ray& rout,
    const hit_record& hrec,
    vec3& attenuation,
    curandState* rstate) const {

    vec3 reflection = utils::reflect(vec3::normalize(rin.direction()), hrec.n());
    rout = ray(hrec.p(), reflection + _roughness * utils::random_point_unit_sphere(rstate));
    attenuation = _albedo;

    // check if > 90 degress between the ray and the surface normal (negative dot product)
    float dot = vec3::dot(rout.direction(), hrec.n());
    return (dot > 0.f);
}

__device__ bool
dielectric::scatter(const ray& rin, ray& rout,
    const hit_record& hrec,
    vec3& attenuation,
    curandState* rstate) const {

    // normal vector at the hit point. It can
    // point inward or outward, depending on the
    // angle of the ray
    vec3 refraction_normal;
    
    // reflected ray
    vec3 reflected = utils::reflect(rin.direction(), hrec.n());

    float mu;
    float cosine;
    // glass surface absorbs nothing (no attenuation)
    //attenuation = vec3(1.f, 1.f, 1.f);
    attenuation = _tint;
    // if the dot product of the incident ray and the normal vector 
    // is positive, it means that we are inside the material (not outside)
    // thus we need to invert the normal. otherwise, we invert the refraction index
    if (vec3::dot(rin.direction(), hrec.n()) > 0.f) {
        refraction_normal = -hrec.n();
        mu = _ri;
        cosine = vec3::dot(rin.direction(), hrec.n()) / rin.direction().length();
        cosine = __fsqrt_rz(1.f - _ri * _ri * (1 - cosine * cosine));
    } else {
        refraction_normal = hrec.n();
        mu = 1.f / _ri;
        cosine = -vec3::dot(rin.direction(), hrec.n()) / rin.direction().length();
    }

    float reflect_prob;
    vec3 refracted;

    if (utils::refract(rin.direction(), refraction_normal, mu, refracted)) {
        reflect_prob = utils::shlick(cosine, _ri);
    } else {
        // reflect everything
        reflect_prob = 1.f;
    }

    // reflect or refract using reflect_prob
    if (curand_uniform(rstate) < reflect_prob) {
        // reflect
        rout = ray(hrec.p(), reflected);
    } else {
        rout = ray(hrec.p(), refracted);
    }
    return true;
}