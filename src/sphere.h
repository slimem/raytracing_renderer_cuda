#pragma once

#include "hitable_object.h"
#include "material.h"

class sphere : public hitable_object {
public:
    __device__ sphere() {};
    __device__ sphere(vec3 center, float radius, const material* mat, bool inside = false)
        : _c(center), _r(radius), _m(mat), _inside(inside) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) override;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;
    __device__ inline vec3 get_center() const { return _c; }
    __device__ virtual ~sphere() noexcept override;

    __device__ virtual object_type get_object_type() const override {
        return object_type::SPHERE;
    }

private:

    __device__ static constexpr void get_sphere_uv(const vec3& p, float& u, float& v);

    vec3 _c;
    float _r = 0.f;
    const material* _m = nullptr; // change to shaed pointer
    bool _inside;
};

class moving_sphere : public hitable_object {
public:
    __device__ moving_sphere() {};
    __device__ moving_sphere(vec3 center0, vec3 center1,
        float time0, float time1,
        float radius,
        const material* mat)
        : _c0(center0), _c1(center1),
          _t0(time0), _t1(time1),
          _r(radius),
          _m(mat) {}
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) override;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;
    __device__ ~moving_sphere() noexcept override;

    __device__ virtual object_type get_object_type() const override {
        return object_type::MOVING_SPHERE;
    }

    __device__ inline vec3 center(float time) const {
        // interpolation between centers
        return _c0 + ((time - _t0) / (_t1 - _t0)) * (_c1 - _c0);
    }

private:
    vec3 _c0, _c1;
    float _t0, _t1;
    float _r;
    const material* _m = nullptr; // change to a shared ptr
};

__device__ void constexpr
sphere::get_sphere_uv(const vec3& p, float& u, float& v) {
    // how to map point coordinates on a sphere to u v coordinates
    // - p: is a point on the sphere of radius one (centered at origin)
    //      first I had a bug while mapping spheres where I used P as a
    //      global coordinates. P needs to be on the sphere and normalized
    //      with the sphere itself so the size of the sphere does not change
    //      while transforming P coordinates to u v coordinates
    // - u: [0,1] project X and Z value (from -1 to -1) around Y axis
    // - v: [0,1] maps to Y value
    // some value examples:
    // (1   0   0) => (0.5  0.5)
    // (0   1   0) => (0.5  1.0)
    // (0   0   1) => (0.25 0.5)
    // (-1  0   0) => (0    0.5)
    // (0  -1   0) => (0.5  0  )
    // (0   0  -1) => (0.75 0.5)

    float phi = atan2f(p.z(), p.x());
    float theta = asinf(p.y());
    u = 1 - (phi + M_PI) / (2 * M_PI);
    v = (theta + M_PI_2) / M_PI;
}


__device__ bool
sphere::hit(const ray& r, float tmin, float tmax, hit_record& hrec) {
    //printf(" The SPHERE is calling this hit:  ");
    vec3 oc = r.origin() - _c; // A - C
    // 1 - dot((p(​ t) - c)​ ,(p(​ t) - c​)) = R*R
    // 2 - dot((A​ + t*B ​- C)​ ,(A​ + t*B​ - C​)) = R*R (A is origin, B is direction)
    // 3 - t*t*dot(B,​ B)​ + 2*t*dot(B,A​-C​) + dot(A-C,A​-C​) - R*R = 0
    // we solve it as a 2nd degree polynomial with delta = b^2 - 4*a*c
    float a = vec3::dot(r.direction(), r.direction());
    float b2 = vec3::dot(oc, r.direction());
    float c = vec3::dot(oc, oc) - _r * _r;
    float delta = b2 * b2 - a * c;

    if (delta < 0) {
        return false;
    }

    // delta is strictly positive, we have two hit points
    // root is the point of intersection between the ray and the sphere
    // TODO: use __fsqrt_rz for sqrt and profile!!
    float deltaSqrt = sqrt(delta);
    float root = (-b2 - deltaSqrt) / a;

    // pick nearest root
    if ((root < tmin) || (root > tmax)) {
        //if (_inside) return false;
        root = (-b2 + deltaSqrt) / a;
        if ((root < tmin) || (root > tmax)) {
            return false;
        }
    }

    hrec.set_t(root);
    hrec.set_p(r.point_at_parameter(root)); // intersection between ray and sphere
    // Remember: sphere normal is intersection_p - center
    // to get a normalized vector, we devide by its magnitude which is
    // the sphere radius
    vec3 out_normal = (hrec.p() - _c) / _r;
    float u, v; // texture coordinates
    hrec.set_n(out_normal);
    get_sphere_uv(out_normal, u, v);
    hrec.set_u(u);
    hrec.set_v(v);
    //printf(" --- CALCULATED NORMAL: %f,%f,%f\n", hrec.n().x(), hrec.n().y(), hrec.n().z());
    hrec.set_h(hit_object_type::SPHERE);
    hrec.set_m(_m);
    //printf(" DID HIT\n");
    //if (_inside) {
        //return false
        //if (vec3::dot(out_normal, r.direction()) < 1) return false;
        //if (vec3::dot(out_normal, r.direction()) < 1) return false;
        //if (vec3::dot(out_normal, r.direction()) < 0) return false;
    //}
    return true;
}

__device__ bool
sphere::bounding_box(float t0, float t1, AABB& box) const {
    box = AABB(_c - vec3(_r), _c + vec3(_r));
    return true;
}

__device__
sphere::~sphere() noexcept {
    printf("Deleting sphere object at %p\n", this);
    if (_m) {
        printf("--Deleting material object at %p\n", _m);
        delete _m;
    }
}

__device__ bool
moving_sphere::hit(const ray& r, float tmin, float tmax, hit_record& hrec) {
    vec3 oc = r.origin() - center(r.t());
    float a = vec3::dot(r.direction(), r.direction());
    float b = vec3::dot(oc, r.direction());
    float c = vec3::dot(oc, oc) - _r * _r;
    float delta = b * b - a * c;

    if (delta > 0) {
        float t = (-b - sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            hrec.set_n((hrec.p() - center(r.t())) / _r);

            hrec.set_h(hit_object_type::SPHERE);

            hrec.set_m(_m);
            return true;
        }
        // we use the same variable
        t = (-b + sqrt(delta)) / a;
        if (t < tmax && t > tmin) {
            hrec.set_t(t);
            hrec.set_p(r.point_at_parameter(t));
            hrec.set_n((hrec.p() - center(r.t())) / _r);
            hrec.set_h(hit_object_type::SPHERE);
            hrec.set_m(_m);
            return true;
        }
    }

    return false;
}

__device__ bool
moving_sphere::bounding_box(float t0, float t1, AABB& box) const {
    // box at t0
    AABB box0 = AABB(_c0 - vec3(_r), _c0 + vec3(_r));

    // box at t1
    AABB box1 = AABB(_c1 - vec3(_r), _c1 + vec3(_r));

    box = AABB::surrounding_box(box0, box1);
    return true;
}
__device__
moving_sphere::~moving_sphere() noexcept {
    printf("Deleting moving_sphere object at %p\n", this);
    if (_m) {
        printf("--Deleting material object at %p\n", _m);
        delete _m;
    }
}
