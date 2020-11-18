#pragma once

#include "hitable_object.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// class for bounding volume hierarchy
// a bvh node will check if a ray hits it or not
class bvh_node : public hitable_object {
public:
    __device__ bvh_node() {}
    __device__ bvh_node(hitable_object** hlist, int n, float time0, float time1, curandState* rstate);

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) const override;

    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;

    __device__ virtual object_type get_object_type() const override{
        return object_type::BOUNDING_VOLUME_HIERARCHY;
    }

    __device__ virtual ~bvh_node() noexcept;

private:
    hitable_object* _left = nullptr;
    hitable_object* _right = nullptr;
    AABB _box;
};

struct box_compare {
    __device__ box_compare(int axis) : _axis(axis) {}
    __device__ bool operator()(hitable_object* ah, hitable_object* bh) const {
        return true;

        printf("Accessing %p and %p ", ah, bh);
        printf("COMPARING AH = %d and BH = %d\n", static_cast<int>(ah->get_object_type()), static_cast<int>(bh->get_object_type()));
        AABB left_box, right_box;
        if (!ah->bounding_box(0.f, 0.f, left_box)
            || !bh->bounding_box(0.f, 0.f, right_box)) {
            printf("Error: No bounding box in bvh_node constructor\n");
            return false;
        }

        float left_min, right_min;
        if (_axis == 0) {
            left_min = left_box.min().x();
            right_min = right_box.min().x();
        } else if (_axis == 1) {
            left_min = left_box.min().y();
            right_min = right_box.min().y();
        } else if (_axis == 2) {
            left_min = left_box.min().z();
            right_min = right_box.min().z();
        } else {
            printf("Error: Unsupported comparison mode\n");
            return false;
        }

        if ((left_min - right_min) < 0.f) {
            return false;
        } else {
            return true;
        }
        return false;
    }
    int _axis = 0;
};

__device__ bool box_x_compare(hitable_object* a, hitable_object* b) {
    AABB left_box, right_box;
    hitable_object* ah = a;
    hitable_object* bh = b;
    if (!ah->bounding_box(0.f, 0.f, left_box)
        || !bh->bounding_box(0.f, 0.f, right_box)) {
        printf("Error: No bounding box in bvh_node constructor\n");
    }
    if (left_box.min().x() - right_box.min().x() < 0.f) {
        return false;
    } else {
        return true;
    }
}

__device__ bool box_y_compare(hitable_object* a, hitable_object* b) {
    AABB left_box, right_box;
    hitable_object* ah = a;
    hitable_object* bh = b;
    if (!ah->bounding_box(0.f, 0.f, left_box) || !bh->bounding_box(0.f, 0.f, right_box)) {
        printf("Error: No bounding box in bvh_node constructor\n");
    }
    if (left_box.min().y() - right_box.min().y() < 0.f) {
        return false;
    } else {
        return true;
    }
}

__device__ bool box_z_compare(hitable_object* a, hitable_object* b) {
    AABB left_box, right_box;
    hitable_object* ah = a;
    hitable_object* bh = b;
    if (!ah->bounding_box(0.f, 0.f, left_box) || !bh->bounding_box(0.f, 0.f, right_box)) {
        printf("Error: No bounding box in bvh_node constructor\n");
    }
    if (left_box.min().z() - right_box.min().z() < 0.f) {
        return false;
    } else {
        return true;
    }
}

__device__ bool compare(int a, int b) {
    return a < b;
}
__device__
bvh_node::bvh_node(
    hitable_object** hlist,
    int n,
    float time0,
    float time1,
    curandState* rstate) {

    /*device_vector<int> devVect;
    for (int i = 0; i < 6; ++i) {
        int rand = curand(rstate) % 30;
        devVect.emplace_back(rand);
        printf("%d, ", devVect[devVect.size() - 1]);
    }
    printf("\nWILL START SORTING NOW %d\n", devVect[0]);
    utils::seq_qsort<device_vector<int>, int>(devVect, 0, devVect.size() - 1, compare);
    //auto function = [&](int a, int b) {return a >= b;};
    //utils::seq_qsort<device_vector<int>, int>(devVect, 0, devVect.size()-1, [&](int a, int b) {return a >= b; });
    printf("SORTED: \n");
    for (int i = 0; i < 6; ++i) {
        printf("%d, ", devVect[i]);
    }
    printf("\n");

    device_vector<hitable_object*> hv;
    for (int i = 0; i < n; ++i) {
        printf("%s ", hitable_list::obj_type_str(hlist[i]->get_object_type()));
        hv.emplace_back(hlist[i]);
    }
    printf("\n");
    utils::seq_qsort<device_vector<hitable_object*>, hitable_object*>(hv, 0, n - 1, box_x_compare);
    for (int i = 0; i < n; ++i) {
        printf("%s ", hitable_list::obj_type_str(hv[i]->get_object_type()));
        //hv.emplace_back(hlist[i]);
    }
    printf("\n");

    utils::seq_qsort<hitable_object**, hitable_object*>(hlist, 0, n, box_x_compare);

    for (int i = 0; i < n; ++i) {
        printf("%s ", hitable_list::obj_type_str(hlist[i]->get_object_type()));
        //hv.emplace_back(hlist[i]);
    }

    printf("=====================\n");*/
    
    // chose a random axis
    int axis = curand(rstate) % 3;
    if (axis == 0) {
        //thrust::sort(objects.begin(), objects.end(), box_x_compare);
        utils::seq_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_x_compare);
    } else if (axis == 1) {
        utils::seq_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_y_compare);
    } else {
        utils::seq_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_z_compare);
    }

    // if one element, left and right are the same
    if (n == 1) {
        _left = _right = hlist[0];
    } else if (n == 2) {
        _left = hlist[0];
        _right = hlist[1];
    } else {
        _left = new bvh_node(hlist, n / 2, time0, time1, rstate);
        _right = new bvh_node(hlist + n / 2, n - n / 2, time0, time1, rstate);
    }

    AABB left_box, right_box;
    if (!_left->bounding_box(time0, time1, left_box)
        || !_right->bounding_box(time0, time1, right_box)) {
        printf("Error: No bounding box in bvh_node constructor\n");
    }
    _box = AABB::surrounding_box(left_box, right_box);
}


__device__ bool
bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {
    
    if (_box.hit(r, tmin, tmax)) {
        hit_record left_hrec, right_hrec;
        bool hit_left = _left->hit(r, tmin, tmax, left_hrec);
        bool hit_right = _right->hit(r, tmin, tmax, right_hrec);

        if (hit_left && hit_right) {
            // chose the closest
            if (left_hrec.t() < right_hrec.t()) {
                hrec = left_hrec;
            } else {
                hrec = right_hrec;
            }
            return true;
        } else if (hit_left) {
            hrec = left_hrec;
            return true;
        } else if (hit_right) {
            hrec = right_hrec;
            return true;
        } else {
            return false;
        }
    }
    return true;
}

__device__ bool
bvh_node::bounding_box(float t0, float t1, AABB& box) const {
    box = _box;
    return true;
}

__device__
bvh_node::~bvh_node() noexcept {
    printf("Deleting BVH_NODE object at %p\n", this);
}

