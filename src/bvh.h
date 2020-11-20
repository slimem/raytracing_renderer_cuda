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
    __device__ const hitable_object* get_this();
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
    curandState* rstate, 
    int level) {
    
    // chose a random axis
    int axis = curand(rstate) % 3;
    if (axis == 0) {
        thrust::sort(hlist, hlist+n-1, box_x_compare);
        //utils::it_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_x_compare);
    } else if (axis == 1) {
        thrust::sort(hlist, hlist + n - 1, box_y_compare);
        //utils::it_qsort<hitable_object**, hitable_object*>(hlist, 0, n - 1, box_y_compare);
    } else {
        thrust::sort(hlist, hlist + n - 1, box_z_compare);
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
bvh_node::dfs(const ray& r, float tmin, float tmax, hit_record& hrec) {
    
    // allocate traversal stack from thread-local memory and push NULL
    // to indicate that there are no nodes left

    // a list of node pointers
    //node_ptr stack[64];
    // this double node pointer will point to the start of the stack
    //node_ptr* stack_ptr = stack;
    //*stack_ptr++ = NULL;

    hitable_object* stack[32];
    hitable_object** stack_ptr = stack;
    *stack_ptr = NULL; //stack bottom
    stack_ptr++;

    // traverse starting from the root
    hitable_object* node = this;
    hit_record l_rec;
    hit_record r_rec;

    float closest = tmax;


    //bool verdict = false;
    bool hit_something = false;
    do {

        /*if (!node->is_leaf() || static_cast<bvh_node*>(node)->is_lowest_bvh()) {
            // we reached our desired destination, check here
            
        }*/

        /*printf("DFS: current node is (%d) %s LEAF?\n", node->get_id(),
            hitable_object::obj_type_str(node->get_object_type()),
            node->is_leaf());*/

        /*if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
            printf("Will SEGFAULT NOW BECAUSE I SAY IT WILL DO\n");
        }*/

        if (node->is_leaf()) break;

        hitable_object* left = static_cast<bvh_node*>(node)->_left;
        hitable_object* right = static_cast<bvh_node*>(node)->_right;

        /*if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
            //printf("Will SEGFAULT AGAIN NOW BECAUSE I SAY IT WILL DO\n");

            printf("Current LEFT node is (%d) %s LEAF? %d\n", left->get_id(),
                hitable_object::obj_type_str(left->get_object_type()),
                left->is_leaf());
            printf("Current RIGHT node is (%d) %s LEAF? %d\n", right->get_id(),
                hitable_object::obj_type_str(right->get_object_type()),
                right->is_leaf());
        }*/

        bool left_hit = left->hit(r, tmin, closest, l_rec);
        if (left_hit && left->is_leaf()) {
            if (l_rec.t() < closest) {
                closest = l_rec.t();
            }
        }
        bool right_hit = left->hit(r, tmin, closest, r_rec);
        /*if (right_hit) {
            if (r_rec.t() < closest) {
                closest = r_rec.t();
            }
        }*/

        if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
           // printf("Will SEGFAULT AGAIN NOW BECAUSE I SAY IT WILL DO\n");
        }

        //bool hit_something = left_hit || right_hit;

        // we hit an object, not a bvh
        if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
            // we are sure that children are leafs
            if (left_hit && right_hit) {
                // we chose the closest
                if (l_rec.t() < r_rec.t()) {
                    // chose left
                    hrec = l_rec;
                    hit_something = true;
                } else {
                    hrec = r_rec;
                    hit_something = true;
                }
            } else if (left_hit) {
                hrec = l_rec;
                hit_something = true;
            } else if (right_hit) {
                hrec = r_rec;
                hit_something = true;
            }
            // exit loop
            if (hit_something) {
                //printf("Hit a leaf object, exiting..\n");
                return true;
            }

            // we are at the bottom_level + 1; nothing to hit
            // so we pop from the stack
            //printf("SADLY WE DIDNT HIT LEAF, POPPING\n");
            node = *--stack_ptr;

        } else {

            //printf("We are still in a BVH node\n");

            bool traverse_left = left_hit && !(left->is_leaf());
            bool traverse_right = right_hit && !(right->is_leaf());

            // we update our stack
            if (!traverse_left && !traverse_right) {
                //printf("POPPING:\n");
                node = *--stack_ptr; //pop node
                /*printf("POPPING NODE AND WE ARE POINTING TO (%d) %s LEAF?\n", node->get_id(),
                    hitable_object::obj_type_str(node->get_object_type()),
                    node->is_leaf());*/
            } else {
                node = (traverse_left) ? left : right;
                
                if (traverse_left && traverse_right) {
                    // pushing to stack
                    *stack_ptr++ = right;
                    /*printf("PUSHING NODE TO STACK (%d) %s LEAF?\n", right->get_id(),
                        hitable_object::obj_type_str(right->get_object_type()),
                        right->is_leaf());
                    printf("TRAVERSING WITH NODE (%d) %s LEAF?\n", node->get_id(),
                        hitable_object::obj_type_str(node->get_object_type()),
                        node->is_leaf());
                    printf(" SEARCHING IN BOTH NODES\n");*/
                } /*else {
                    printf(" ONLY SEARCHING IN ONE NODE\n");
                }*/
            }
        }
        //printf("----------------------------------\n");
        //printf("At this point everything is OK\n");
        /*if (node != NULL) {
            printf("CURRENT NODE IS NOT NULL, WILL CONTINUE\n");
        }*/
        
        //printf("%s -> ", hitable_object::obj_type_str(node->get_object_type()));
        //node = static_cast<bvh_node*>(node)->_left;
    } while (node != NULL);
    
    return false;
    //printf("Finished DFS...HIT NOTHING\n");
    /*for (int i = 0; i < 64; ++i) {
        printf("DFS: %p\n", *(stack_ptr + i));
    }*/
}

/*__device__ bool
bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& hrec) const {*/
    //printf(" The DFS is calling this hit:  ");
    //if (_box.hit(r, tmin, tmax)) {
        //printf(" DID HIT\n");
        // do DFS on the tree

        /*
        hit_record left_hrec, right_hrec;
        bool hit_left = _left->hit(r, tmin, tmax, left_hrec);
        bool hit_right = _right->hit(r, tmin, tmax, right_hrec);

        // check if right and left are child nodes

        if (hit_left && hit_right) {
            // chose the closest
            if (left_hrec.t() < right_hrec.t()) {
                // chose left
                if (_left->get_object_type() != object_type::BOUNDING_VOLUME_HIERARCHY) {
                    hrec = left_hrec;
                    return true;
                } else {
                    return false;
                }

            } else {
                if (_right->get_object_type() != object_type::BOUNDING_VOLUME_HIERARCHY) {
                    hrec = right_hrec;
                    return true;
                } else {
                    return false;
                }
            }
            
        } else if (hit_left) {
            if (_left->get_object_type() != object_type::BOUNDING_VOLUME_HIERARCHY) {
                hrec = left_hrec;
                return true;
            } else {
                return false;
            }
        } else if (hit_right) {
            if (_right->get_object_type() != object_type::BOUNDING_VOLUME_HIERARCHY) {
                hrec = right_hrec;
                return true;
            }
        } else {
            return false;
        }*/
       // return true;
   // }
    //printf(" DID NOT HIT\n");
   // return false;
//}
__device__ const hitable_object*
bvh_node::get_this() {
    return this;
}

#if 0
__device__ bool
bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& hrec) {
    if (_box.hit(r, t_min, t_max)) {


        hitable_object* stack[32];
        hitable_object** stack_ptr = stack;
        *stack_ptr = NULL; //stack bottom
        stack_ptr++;

        // traverse starting from the root
        hitable_object* node = this;
        hit_record l_rec;
        hit_record r_rec;



        //bool verdict = false;
        bool hit_something = false;
        do {

            /*if (!node->is_leaf() || static_cast<bvh_node*>(node)->is_lowest_bvh()) {
                // we reached our desired destination, check here

            }*/

            /*printf("DFS: current node is (%d) %s LEAF?\n", node->get_id(),
                hitable_object::obj_type_str(node->get_object_type()),
                node->is_leaf());*/

                /*if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
                    printf("Will SEGFAULT NOW BECAUSE I SAY IT WILL DO\n");
                }*/

            if (node->is_leaf()) break;

            hitable_object* left = static_cast<bvh_node*>(node)->_left;
            hitable_object* right = static_cast<bvh_node*>(node)->_right;


            bool left_hit = left->hit(r, t_min, t_max, l_rec);
            bool right_hit = left->hit(r, t_min, t_max, r_rec);
            /*if (right_hit) {
                if (r_rec.t() < closest) {
                    closest = r_rec.t();
                }
            }*/

            if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
                // printf("Will SEGFAULT AGAIN NOW BECAUSE I SAY IT WILL DO\n");
            }

            //bool hit_something = left_hit || right_hit;

            // we hit an object, not a bvh
            if (static_cast<bvh_node*>(node)->is_lowest_bvh()) {
                // we are sure that children are leafs
                if (left_hit && right_hit) {
                    // we chose the closest
                    if (l_rec.t() < r_rec.t()) {
                        // chose left
                        hrec = l_rec;
                        hit_something = true;
                    } else {
                        hrec = r_rec;
                        hit_something = true;
                    }
                } else if (left_hit) {
                    hrec = l_rec;
                    hit_something = true;
                } else if (right_hit) {
                    hrec = r_rec;
                    hit_something = true;
                }
                // exit loop
                if (hit_something) {
                    //printf("Hit a leaf object, exiting..\n");
                    return true;
                }

                node = *--stack_ptr;

            } else {


                bool traverse_left = left_hit && !(left->is_leaf());
                bool traverse_right = right_hit && !(right->is_leaf());

                // we update our stack
                if (!traverse_left && !traverse_right) {

                    node = *--stack_ptr; //pop node

                } else {
                    node = (traverse_left) ? left : right;

                    if (traverse_left && traverse_right) {

                        *stack_ptr++ = right;

                    }
                }
            }

        } while (node != NULL);

        return false;

    } else return false;
}
#endif

__device__ bool
bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec)  {
    if (_box.hit(r, t_min, t_max)) {
        hit_record left_rec, right_rec;
        bool hit_left = _left->hit(r, t_min, t_max, left_rec);
        bool hit_right = _right->hit(r, t_min, t_max, right_rec);
        if (hit_left && hit_right) {
            if (left_rec.t() < right_rec.t())
                rec = left_rec;
            else
                rec = right_rec;
            return true;
        } else if (hit_left) {
            rec = left_rec;
            return true;
        } else if (hit_right) {
            rec = right_rec;
            return true;
        } else
            return false;
    } else return false;
}









/*__device__ bool
bvh_node::hit(const ray& r, float tmin, float tmax, hit_record& hrec) {
    
    if (_box.hit(r, tmin, tmax)) {
        hit_record left_hrec, right_hrec;
        bool hit_left = _left->hit(r, tmin, tmax, left_hrec);
        bool hit_right = _right->hit(r, tmin, tmax, right_hrec);

        // check if right and left are child nodes

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
}*/



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
