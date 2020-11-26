#pragma once

#include "vec3.h"

class utils {
public:

    __host__ __device__ static constexpr int XY(int x, int y, int width = WIDTH) {
#ifdef __CUDA_ARCH__
        // __fmaf_rz(x, y, z) returns x * y + z
        return __fmaf_rz(y, width, x);
#else
        return y * width + x;
#endif
    }

    __device__ static constexpr float fmin(float a, float b) {
        return a < b ? a : b;
    }

    __device__ static constexpr float fmax(float a, float b) {
        return a > b ? a : b;
    }

    template <typename T>
    __host__ __device__ static void inline swap(T& a, T& b) {
        T c(a);
        a = b;
        b = c;
    }

    __device__ static vec3 random_point_unit_sphere(curandState* rstate);
    __device__ static vec3 random_point_unit_disk(curandState* rstate);

    __device__ static inline vec3 reflect(const vec3& v, const vec3& n);

    __device__ static inline bool refract(
        const vec3& v, const vec3& n,
        float ni_nt,
        vec3& refracted
    );

    //compute specular reflection coefficient R using Schlick's model
    __device__ static constexpr float shlick(float cosine, float refid);

    // A recursive qsort
    template<typename T, typename V>
    __device__ static void rec_qsort(T& v, int low, int high, bool(*f)(V, V));

    // An iterative qsort
    template<typename T, typename V>
    __device__ static void it_qsort(T& v, int low, int high, bool(*f)(V, V));

    template<typename T, typename V>
    __device__ static size_t seq_partition(T& v, int low, int high, bool(*f)(V, V));

    template<typename T>
    __host__ static void unpack_rgb_to_rgba(T* rgba, const T* rgb, const size_t n);
};

__device__ vec3
utils::random_point_unit_sphere(curandState* rstate) {
    vec3 point;
    do {
        // grab a random point and center it in
        // the unit circle
        // the random value is generated using the random state
        // of the pixel calling the function
        point = 2.f * vec3(
            curand_uniform(rstate),
            curand_uniform(rstate),
            curand_uniform(rstate)
        ) - vec3(1.f, 1.f, 1.f);

    } while (point.sq_length() >= 1.f);
    return point;
}

__device__ vec3
utils::random_point_unit_disk(curandState* rstate) {
    vec3 point;
    do {
        point = 2.f * vec3(
            curand_uniform(rstate),
            curand_uniform(rstate),
            0
        ) - vec3(1.f, 1.f, 0.f);

    } while (vec3::dot(point, point) >= 1.f);
    return point;
}

__device__ inline vec3
utils::reflect(const vec3& v, const vec3& n) {
    // r = i - 2 ( i * n ) * n
    return v - 2.f * vec3::dot(v, n) * n;
}


// refract a vector using snells's law:
// i is the incident
// t is the refracted
// μ is n1/n2
// n is the normal vector
// t = μ * (i - (n.i)n) - n * sqrt(1 - μ*μ*(1 - (n.i)^2)
// the discriminant 1 - μ*μ*(1 - (n.i)^2 should be positive
__device__ inline bool
utils::refract(const vec3& v, const vec3& n,
    float mu,
    vec3& refracted) {
    //mu = n1/n2
    // n is already normalized
    vec3 i = vec3::normalize(v);
    float in = vec3::dot(i, n);
    float delta = 1.f - mu * mu * (1 - in * in);
    if (delta > 0) { // there is refraction
        refracted = mu * (i - n * in) - n * sqrt(delta);
        return true;
    } else {
        return false;
    }
}

__device__ constexpr float
utils::shlick(float cosine, float ref_id) {
#ifdef __CUDA_ARCH__
    float r0 = 
        __fdiv_rz(
        (1.f - ref_id),
        (1.f + ref_id)
        );
    r0 = __fmul_rz(r0, r0);
    return r0
        + __fmul_rz(
        (1.f - r0),
        __powf(1.f - cosine, 5.f)
        );
#else
    float r0 = (1.f - ref_id) / (1.f + ref_id);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5.f);
#endif
}

template<typename T>
class device_vector
{
private:
    T* m_begin;
    T* m_end;

    size_t _capacity;
    size_t _length;
    __device__ void expand() {
        _capacity *= 2;
        size_t tempLength = (m_end - m_begin);
        T* tempBegin = new T[_capacity];

        memcpy(tempBegin, m_begin, tempLength * sizeof(T));
        delete[] m_begin;
        m_begin = tempBegin;
        m_end = m_begin + tempLength;
        _length = static_cast<size_t>(m_end - m_begin);
    }
public:
    __device__  explicit device_vector() : _length(0), _capacity(16) {
        m_begin = new T[_capacity];
        m_end = m_begin;
    }
    __device__ T& operator[] (unsigned int index) {
        return *(m_begin + index);//*(begin+index)
    }
    __device__ T* begin() {
        return m_begin;
    }
    __device__ T* end() {
        return m_end;
    }
    __device__ ~device_vector()
    {
        delete[] m_begin;
        m_begin = nullptr;
    }

    __device__ void emplace_back(T t) {

        if ((m_end - m_begin) >= _capacity) {
            expand();
        }

        new (m_end) T(t);
        m_end++;
        _length++;
    }
    __device__ T pop_back() {
        T endElement = (*m_end);
        delete m_end;
        m_end--;
        return endElement;
    }

    __device__ size_t size() {
        return _length;
    }

    __device__ bool empty() {
        return _length != 0;
    }
};

template<typename T, typename V>
__device__ size_t
utils::seq_partition(T& v, int low, int high, bool(*f)(V,V)) {
    V pivot = v[high]; // pivot
    //index of smaller element
    int i = (low - 1);

    for (int j = low; j <= high - 1; ++j) {
        if (f(v[j], pivot)) {
            ++i;
            swap(v[i], v[j]);
        }
    }

    swap(v[i + 1], v[high]);
    return (i + 1);
}

template<typename T, typename V>
__device__ void
utils::rec_qsort(T& v, int low, int high, bool(*f)(V,V)){
    if (low < high) {
        int pi = seq_partition<T, V>(v, low, high, f);

        rec_qsort<T,V>(v, low, pi - 1, f);
        rec_qsort<T,V>(v, pi + 1, high, f);
    }
}

template<typename T, typename V>
__device__ void
utils::it_qsort(T& v, int low, int high, bool(*f)(V, V)) {

    int top = -1;
    int stack[high - low + 1];
    // prepare initial values
    stack[++top] = low;
    stack[++top] = high;

    // pop from the stack
    while (top >= 0) {
        //pop high and low
        high = stack[top--];
        low = stack[top--];

        // set pivot at its correct position
        int pi = seq_partition<T, V>(v, low, high, f);

        // if there are elements on the left side of the pivot push left
        // side to stack
        if (pi - 1 > low) {
            stack[++top] = low;
            stack[++top] = pi - 1;
        }
        // if there are elements on the right side of the pivot push right
        // side to stack
        if (pi + 1 < high) {
            stack[++top] = pi + 1;
            stack[++top] = high;
        }
    }
}

template<typename T>
__host__ void
utils::unpack_rgb_to_rgba(T* rgba, const T* rgb, const size_t size) {
    if (size == 0) {
        return;
    }
    for (int i = size; --i; rgba += 4, rgb += 3) {
        *(uint32_t*)(void*)rgba = *(const uint32_t*)(const void*)rgb;
    }
    for (int j = 0; j < 3; ++j) {
        rgba[j] = rgb[j];
    }
}