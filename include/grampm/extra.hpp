#ifndef GRAMPM_EXTRA
#define GRAMPM_EXTRA

#include <Kokkos_Core.hpp>
#include <type_traits>

// class containing grid indices or min/max coordinates of a box on a grid
template<typename T>
struct box {
    T start[3], end[3];
    box(T start_[3], T end_[3])
        : start {start_[0], start_[1], start_[2]}
        , end {end_[0], end_[1], end_[2]}
    {}

    box(T startx, T starty, T startz, T endx, T endy, T endz)
        : start {startx, starty, startz}
        , end {endx, endy, endz}
    {}

    box() {}

    T range(const int d) const {return end[d]-start[d];}

    bool no_overlap_with(const box<T> &other) const {
        // <= and >= because end is open interval
        return start[0] >= other.end[0] ||
            start[1] >= other.end[1] ||
            start[2] >= other.end[2] ||
            end[0] <= other.start[0] ||
            end[1] <= other.start[1] ||
            end[2] <= other.start[2];
    }

    box<T> find_overlapping_box(const box<T> &other) const {
        return box<T>(
            Kokkos::max(start[0], other.start[0]),
            Kokkos::max(start[1], other.start[1]),
            Kokkos::max(start[2], other.start[2]),
            Kokkos::min(end[0], other.end[0]),
            Kokkos::min(end[1], other.end[1]),
            Kokkos::min(end[2], other.end[2])
        );
    }

    box<T> translate_origin_using(const T idx[3]) const {
        return box<T>(
            start[0] - idx[0],
            start[1] - idx[1],
            start[2] - idx[2],
            end[0] - idx[0],
            end[1] - idx[1],
            end[2] - idx[2]
        );
    }

    KOKKOS_INLINE_FUNCTION
    bool contains_point(const T idx[3]) const {
        return idx[0] >= start[0] &&
            idx[1] >= start[1] && 
            idx[2] >= start[2] &&
            idx[0] < end[0] &&
            idx[1] < end[1] &&
            idx[2] < end[2];
    }

    KOKKOS_INLINE_FUNCTION
    bool contains_point(const T idxx, const T idxy, const T idxz) const {
        return idxx >= start[0] &&
            idxy >= start[1] && 
            idxz >= start[2] &&
            idxx < end[0] &&
            idxy < end[1] &&
            idxz < end[2];
    }

    template<typename F>
    box<F> idx2coords(const F dcell, const F mingrid[3]) {
        return box<F>(
            mingrid[0] + start[0]*dcell,
            mingrid[1] + start[1]*dcell,
            mingrid[2] + start[2]*dcell,
            mingrid[0] + end[0]*dcell,
            mingrid[1] + end[1]*dcell,
            mingrid[2] + end[2]*dcell
        );
    }
};
#endif