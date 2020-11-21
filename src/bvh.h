#pragma once

#include "hitable_object.h"
#include <thrust/sort.h>
#include <thrust/device_vector.h>

// class for bounding volume hierarchy
// a bvh node will check if a ray hits it or not
class bvh_node : public hitable_object {
public:
    __device__ bvh_node() {}
    __device__ bvh_node(hitable_object** hlist, int n, float time0, float time1, curandState* rstate, int level);

    __device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& hrec) override;
    __device__ bool dfs(const ray& r, float tmin, float tmax, hit_record& hrec);

    __device__ virtual bool bounding_box(float t0, float t1, AABB& box) const override;

    __device__ virtual object_type get_object_type() const override{
        return object_type::BOUNDING_VOLUME_HIERARCHY;
    }

    __device__ constexpr hitable_object* left() const { return _left; }
    __device__ constexpr hitable_object* right() const { return _right; }

    __device__ static void display_tree(bvh_node* root, int level);
    __device__ inline bool is_lowest_bvh() {
        // _left and _right should not be null
        return (_left->is_leaf() || _right->is_leaf());
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

__device__
bvh_node::bvh_node(
    hitable_object** hlist,
    int n,
    float time0,
    float time1,
    curandState* rstate, 
    int level) {
    
    // chose a random axis
    int axis = curand(rstate) % 3;
    if (axis == 0) {
        thrust::sort(hlist, hlist+n-1, box_compare(0));
        //utils::it_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_x_compare);
    } else if (axis == 1) {
        thrust::sort(hlist, hlist + n - 1, box_compare(1));
        //utils::it_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_y_compare);
    } else {
        thrust::sort(hlist, hlist + n - 1, box_compare(2));
        //utils::it_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_z_compare);
    }

    // if one element, left and right are the same
    if (n == 1) {
        _left = _right = hlist[0];
    } else if (n == 2) {
        _left = hlist[0];
        _right = hlist[1];
    } else {
        _left = new bvh_node(hlist, n / 2, time0, time1, rstate, level + 1);
        _left->set_id((level + 1) * 10);
        _right = new bvh_node(hlist + n / 2, n - n / 2, time0, time1, rstate, level + 1);
        _right->set_id((level + 1) * 11) ;
    }

    AABB left_box, right_box;
    if (!_left->bounding_box(time0, time1, left_box)
        || !_right->bounding_box(time0, time1, right_box)) {
        printf("Error: No bounding box in bvh_node constructor\n");
    }
    _box = AABB::surrounding_box(left_box, right_box);
}

__device__ bool
bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& hrec) {
    //return true;
    return _box.hit(r, tmin, tmax);
}

__device__ bool
bvh_node::dfs(const ray& r, float tmin, float tmax, hit_record& hrec) {
    if (!_box.hit(r, tmin, tmax)) return false;
    hitable_object* stack[STACK_SIZE];
    hitable_object** stack_ptr = stack;
    *stack_ptr = NULL; //stack bottom
    stack_ptr++;
    *stack_ptr = this;
    stack_ptr++;
    hit_record temp_rec;
    float closest = tmax;
    bool hit_anything = false;
   

    while (*--stack_ptr != NULL) {
        hitable_object* node = *stack_ptr;


        if (!node->is_leaf()) {
            if (node->hit(r, tmin, tmax, temp_rec)) {
                *stack_ptr++ = static_cast<bvh_node*>(node)->_left;
                *stack_ptr++ = static_cast<bvh_node*>(node)->_right;
            }
        } else {
            // leaf node; check if intersects
            if (node->hit(r, tmin, closest, temp_rec) && (temp_rec.t() < closest)) {

                hit_anything = true;
                closest = temp_rec.t();
                hrec = temp_rec;
            }
        }
    }
    return hit_anything;
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

__device__ void
bvh_node::display_tree(bvh_node* root, int level) {
    printf("%*c|_", level, ' ');
    printf("(%d) %s\n", root->get_id(), hitable_object::obj_type_str(root->get_object_type()));
    if (root->_left) {
        if (root->_left->get_object_type() == object_type::BOUNDING_VOLUME_HIERARCHY) {
            display_tree(static_cast<bvh_node*>(root->_left), level + 2);
        } else {
            printf("  %*c|_", level, ' ');
            printf("(%d) LEFT %s (%.2f,%.2f,%.2f)\n", root->_left->get_id(), hitable_object::obj_type_str(root->_left->get_object_type()),
                static_cast<sphere*>(root->_left)->get_center().x(),
                static_cast<sphere*>(root->_left)->get_center().y(),
                static_cast<sphere*>(root->_left)->get_center().z());
        }
    }
    if (root->_right) {
        if (root->_right->get_object_type() == object_type::BOUNDING_VOLUME_HIERARCHY) {
            display_tree(static_cast<bvh_node*>(root->_right), level + 2);
        } else {
            printf("  %*c|_", level, ' ');
            printf("(%d) RIGHT %s (%.2f,%.2f,%.2f)\n", root->_right->get_id(), hitable_object::obj_type_str(root->_right->get_object_type()),
                static_cast<sphere*>(root->_left)->get_center().x(),
                static_cast<sphere*>(root->_left)->get_center().y(),
                static_cast<sphere*>(root->_left)->get_center().z());
        }
    }
}
