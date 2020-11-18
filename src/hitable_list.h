#pragma once

#include "hitable_object.h"

// Embeds a list of N hitable_objects
class hitable_list : public hitable_object {
public:
    __device__ hitable_list() {};
    __device__ hitable_list(hitable_object** hitabe_objects, uint32_t size)
        : _hitable_objects(hitabe_objects), _size(size) {};
    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;
    __device__ ~hitable_list() noexcept override;

    __device__ virtual object_type get_object_type() const override {
        return object_type::HITABLE_LIST;
    }

    __device__ constexpr hitable_object* get_object(uint32_t id);

private:
    // a list of dynamically allocated hitable objects
    hitable_object** _hitable_objects;

    // check for overflow in the future, if we get big scenes
    // that hold more than 2^32 objects
    uint32_t _size;
};

__device__ constexpr hitable_object*
hitable_list::get_object(uint32_t id) {
    for (uint32_t i = 0; i < _size; ++i) {
        if (_hitable_objects[i]->get_id() == id) {
            return _hitable_objects[i];
        }
    }
    return nullptr;
}


// This method checks for hits on all objects between tmin and tmax,
// and updates the hit record accordingly.
__device__ bool
hitable_list::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {
    hit_record tmp_rec;
    bool hit_anything = false;
    float closest = tmax;
    for (uint32_t i = 0; i < _size; i++) {
        if (_hitable_objects[i]->hit(r, tmin, closest, tmp_rec) && (tmp_rec.t() < closest)) {
            hit_anything = true;
            closest = tmp_rec.t();
            hrec = tmp_rec;
        }
    }
    return hit_anything;
}

__device__ bool
hitable_list::bounding_box(float t0, float t1, AABB& box) const {
    if (_size < 1) return false;

    AABB tempbbox; // temporary bounding box
    // if we didnt hit the bounding box of the first object, early exit
    if (!_hitable_objects[0]->bounding_box(t0, t1, tempbbox)) {
        return false;
    } else {
        box = tempbbox;
    }

    // we make the bounding box bigger and bigger with each object
    for (uint32_t i = 1; i < _size; ++i) {
        if (_hitable_objects[i]->bounding_box(t0, t1, tempbbox)) {
            box = AABB::surrounding_box(box, tempbbox);
        } else {
            return false;
        }
    }
    return false;
}

__device__
hitable_list::~hitable_list() noexcept {
    for (uint32_t i = 0; i < _size; ++i) {
        /*switch (_hitable_objects[i]->get_object_type()) {
        case object_type::SPHERE:
            delete ((sphere*)_hitable_objects[i]);
            break;
        case object_type::MOVING_SPHERE:
            delete ((moving_sphere*)_hitable_objects[i]);
            break;
        default:
        {
            printf("Warning! deleting unknown object type\n");
            delete (_hitable_objects[i]);
            break;
        }
        }*/
        delete* (_hitable_objects + i);
    }
}