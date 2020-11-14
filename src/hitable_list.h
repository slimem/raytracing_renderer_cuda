#pragma once

#include "hitable_object.h"

// Embeds a list of N hitable_objects
class hitable_list : public hitable_object {
public:
    __device__ hitable_list() {};
    __device__ hitable_list(hitable_object** hitabe_objects, uint32_t size)
        : _hitable_objects(hitabe_objects), _size(size) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const override;
    __device__ ~hitable_list() noexcept;

private:
    // a list of dynamically allocated hitable objects
    hitable_object** _hitable_objects;

    // check for overflow in the future, if we get big scenes
    // that hold more than 2^32 objects
    uint32_t _size;
};

// This method checks for hits on all objects between tmin and tmax,
// and updates the hit record accordingly.
__device__ bool
hitable_list::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {
    hit_record tmp_rec;
    bool hit_anything = false;
    float closest = tmax;
    for (uint32_t i = 0; i < _size; i++) {
        if (_hitable_objects[i]->hit(r, tmin, closest, tmp_rec)) {
            hit_anything = true;
            closest = tmp_rec.t();
            hrec = tmp_rec;
        }
    }
    return hit_anything;
}

__device__
hitable_list::~hitable_list() noexcept {
    for (uint32_t i = 0; i < _size; ++i) {
        printf("Deleting object at %p\n", _hitable_objects + i);
        delete* (_hitable_objects + i);
    }
}