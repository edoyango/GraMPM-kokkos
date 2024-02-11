#ifndef GRAMPM_EXTRA
#define GRAMPM_EXTRA

#include "Kokkos_Core_fwd.hpp"
#include "Kokkos_Macros.hpp"
#include <Kokkos_Core.hpp>
#include <string>

template<typename T>
struct box {
    T min[3], max[3];
    KOKKOS_INLINE_FUNCTION
    box(const T minx_, const T miny_, const T minz_, const T maxx_, const T maxy_, const T maxz_)
        : min {minx_, miny_, minz_}
        , max {maxx_, maxy_, maxz_}
    {}
    KOKKOS_INLINE_FUNCTION
    box()
    {}
};

template<typename T>
T range(const box<T> &box_, const int d) {return box_.max[d]-box_.min[d];}

template<typename T>
KOKKOS_INLINE_FUNCTION
bool no_overlap(const box<T> &that, const box<T> &other) {
    // <= and >= because end is open interval
    return that.min[0] >= other.max[0] ||
           that.min[1] >= other.max[1] ||
           that.min[2] >= other.max[2] ||
           that.max[0] <= other.min[0] ||
           that.max[1] <= other.min[1] ||
           that.max[2] <= other.min[2];
}

template<typename T>
KOKKOS_INLINE_FUNCTION
box<T> find_overlapping_box(const box<T> &that, const box<T> &other) {
    return box<T>(
        Kokkos::max(that.min[0], other.min[0]),
        Kokkos::max(that.min[1], other.min[1]),
        Kokkos::max(that.min[2], other.min[2]),
        Kokkos::min(that.max[0], other.max[0]),
        Kokkos::min(that.max[1], other.max[1]),
        Kokkos::min(that.max[2], other.max[2])
    );
}

template<typename T>
box<T> translate_origin(const box<T> &that, const T x, const T y, const T z) {
    return box<T>(
        that.min[0] - x,
        that.min[1] - y,
        that.min[2] - z,
        that.max[0] - x,
        that.max[1] - y,
        that.max[2] - z
    );
}

template<typename T>
box<T> translate_origin(const box<T> &that, const T x[3]) {
    return box<T>(
        that.min[0] - x[0],
        that.min[1] - x[1],
        that.min[2] - x[2],
        that.max[0] - x[0],
        that.max[1] - x[1],
        that.max[2] - x[2]
    );
}

template<typename T>
KOKKOS_INLINE_FUNCTION
box<T> extend(const box<T> &that, const T buffer) {
    return box<T>(
        that.min[0] - buffer,
        that.min[1] - buffer,
        that.min[2] - buffer,
        that.max[0] + buffer,
        that.max[1] + buffer,
        that.max[2] + buffer
    );
}

template<typename T>
bool contains_point(const box<T> &that, const T x[3]) {
    return x[0] >= that.min[0] &&
           x[1] >= that.min[1] && 
           x[2] >= that.min[2] &&
           x[0] <  that.max[0] &&
           x[1] <  that.max[1] &&
           x[2] <  that.max[2];
}

template<typename T>
KOKKOS_INLINE_FUNCTION
bool contains_point(const box<T> &that, const T x, const T y, const T z) {
    return x >= that.min[0] &&
           y >= that.min[1] && 
           z >= that.min[2] &&
           x <  that.max[0] &&
           y <  that.max[1] &&
           z <  that.max[2];
}

template<typename T, typename F>
box<F> idx2coords(const box<T> &that, const F dcell, const F mingridx, const F mingridy, const F mingridz) {
    return box<F>(
        mingridx + that.min[0]*dcell,
        mingridy + that.min[1]*dcell,
        mingridz + that.min[2]*dcell,
        mingridx + that.max[0]*dcell,
        mingridy + that.max[1]*dcell,
        mingridz + that.max[2]*dcell
    );
}

template<typename T, typename F>
box<F> idx2coords(const box<T> &that, const F dcell, const F mingrid[3]) {
    return box<F>(
        mingrid[0] + that.min[0]*dcell,
        mingrid[1] + that.min[1]*dcell,
        mingrid[2] + that.min[2]*dcell,
        mingrid[0] + that.max[0]*dcell,
        mingrid[1] + that.max[1]*dcell,
        mingrid[2] + that.max[2]*dcell
    );
}

template<typename T>
KOKKOS_INLINE_FUNCTION
T distance_from_face(const box<T> &that, const T x, const T y, const T z) {
    T tmp = Kokkos::max(
        {T(0), that.min[0] - x, x - that.max[0]}
    );
    T dr = tmp*tmp;
    tmp = Kokkos::max(
        {T(0), that.min[1] - y, y - that.max[1]}
    );
    dr += tmp*tmp;
    tmp = Kokkos::max(
        {T(0), that.min[2] - z, z - that.max[2]}
    );
    dr += tmp*tmp;
    return dr;
}
#endif