#ifndef GRAMPM_EXTRA
#define GRAMPM_EXTRA

#include <Kokkos_Core.hpp>

// class containing grid indices or min/max coordinates of a box on a grid
template<typename T>
struct box {
    T start[3], end[3];

    box(T startx, T starty, T startz, T endx, T endy, T endz)
        : start {startx, starty, startz}
        , end {endx, endy, endz}
    {}

    box(T start_[3], T end_[3])
        : box(start_[0], start_[1], start_[2], end_[0], end_[1], end_[2])
    {}

    box() {}

};

template<typename T>
T range(const box<T> &box_, const int d) {
    return box_.end[d]-box_.start[d];
}

template<typename T>
bool no_overlap(const box<T> &that, const box<T> &other) {
    // <= and >= because end is open interval
    return that.start[0] >= other.end[0] ||
        that.start[1] >= other.end[1] ||
        that.start[2] >= other.end[2] ||
        that.end[0] <= other.start[0] ||
        that.end[1] <= other.start[1] ||
        that.end[2] <= other.start[2];
}

template<typename T>
box<T> find_overlapping_box(const box<T> &that, const box<T> &other) {
    return box<T>(
        Kokkos::max(that.start[0], other.start[0]),
        Kokkos::max(that.start[1], other.start[1]),
        Kokkos::max(that.start[2], other.start[2]),
        Kokkos::min(that.end[0], other.end[0]),
        Kokkos::min(that.end[1], other.end[1]),
        Kokkos::min(that.end[2], other.end[2])
    );
}

template<typename T>
box<T> translate_origin(const box<T> &that, const T idx[3]) {
    return box<T>(
        that.start[0] - idx[0],
        that.start[1] - idx[1],
        that.start[2] - idx[2],
        that.end[0] - idx[0],
        that.end[1] - idx[1],
        that.end[2] - idx[2]
    );
}

template<typename T>
bool contains_point(const box<T> &that, const T idx[3]) {
    return idx[0] >= that.start[0] &&
        idx[1] >= that.start[1] && 
        idx[2] >= that.start[2] &&
        idx[0] < that.end[0] &&
        idx[1] < that.end[1] &&
        idx[2] < that.end[2];
}

template<typename T>
KOKKOS_INLINE_FUNCTION
bool contains_point(const box<T> &that, const T idxx, const T idxy, const T idxz) {
    return idxx >= that.start[0] &&
        idxy >= that.start[1] && 
        idxz >= that.start[2] &&
        idxx < that.end[0] &&
        idxy < that.end[1] &&
        idxz < that.end[2];
}

template<typename T, typename F>
box<F> idx2coords(const box<T> &that, const F dcell, const F mingrid[3]) {
    return box<F>(
        mingrid[0] + that.start[0]*dcell,
        mingrid[1] + that.start[1]*dcell,
        mingrid[2] + that.start[2]*dcell,
        mingrid[0] + that.end[0]*dcell,
        mingrid[1] + that.end[1]*dcell,
        mingrid[2] + that.end[2]*dcell
    );
}
#endif