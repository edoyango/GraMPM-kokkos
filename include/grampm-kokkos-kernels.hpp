#ifndef GRAMPM_KOKKOS_KERNELS
#define GRAMPM_KOKKOS_KERNELS

#include <Kokkos_Core.hpp>
#include <cmath>

// fast (?) sign function from SO https://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template <typename T>
KOKKOS_INLINE_FUNCTION
static int sgn_kokkos(T val) {
    return (T(0.) < val) - (val < T(0.));
}

// it would've been nice to implement kernels via class inheritance and virtual functions, but Kokkos make it difficult
// https://kokkos.github.io/kokkos-core-wiki/usecases/VirtualFunctions.html
namespace GraMPM {
    namespace accelerated {
        namespace kernels {

#ifdef KERNEL_LINEAR_BSPLINE
            template<typename F>
            struct kernel {

                // change me (radius)
                const F radius = 1., dcell;

                kernel(const F dc)
                    : dcell {dc}
                {
                }

                KOKKOS_INLINE_FUNCTION
                void operator()(const F &dx, const F&dy, const F &dz, F &w, F &dwdx, F &dwdy, F &dwdz) const {
                    const F qx = Kokkos::abs(dx)/dcell, qy = Kokkos::abs(dy)/dcell, qz = Kokkos::abs(dz)/dcell;
                    const F w1x = w1(qx), w1y = w1(qy), w1z = w1(qz);
                    w = w1x*w1y*w1z;
                    dwdx = w1y*w1z*dw1dq(qx)*dqdr(dx);
                    dwdy = w1x*w1z*dw1dq(qy)*dqdr(dy);
                    dwdz = w1x*w1y*dw1dq(qz)*dqdr(dz);
                }
                
                // overriding 
                protected:
                    // change me
                    KOKKOS_INLINE_FUNCTION
                    F w1(const F &q) const {
                        return Kokkos::max(0., 1.-q);
                    }

                    // change me
                    KOKKOS_INLINE_FUNCTION
                    F dw1dq (const F &q) const {
                        return -1.;
                    }
                    KOKKOS_INLINE_FUNCTION
                    F dqdr(const F &dr) const {
                        return sgn_kokkos(dr)/dcell;
                    }
            };
#endif

#ifdef KERNEL_CUBIC_BSPLINE
            template<typename F>
            struct kernel {

                const F radius = 2., dcell;

                kernel(const F dc)
                    : dcell {dc}
                {
                }

                KOKKOS_INLINE_FUNCTION
                void operator()(const F &dx, const F&dy, const F &dz, F &w, F &dwdx, F &dwdy, F &dwdz) const {
                    const F qx = Kokkos::abs(dx)/dcell, qy = Kokkos::abs(dy)/dcell, qz = Kokkos::abs(dz)/dcell;
                    const F w1x = w1(qx), w1y = w1(qy), w1z = w1(qz);
                    w = w1x*w1y*w1z;
                    dwdx = w1y*w1z*dw1dq(qx)*dqdr(dx);
                    dwdy = w1x*w1z*dw1dq(qy)*dqdr(dy);
                    dwdz = w1x*w1y*dw1dq(qz)*dqdr(dz);
                }
                
                // overriding 
                protected:
                    KOKKOS_INLINE_FUNCTION
                    F w1(const F &q) const {
                        const F dim2q {Kokkos::max(0., 2.-q)}, dim1q {Kokkos::max(0., 1.-q)};
                        return 2./3.*(0.25*dim2q*dim2q*dim2q-dim1q*dim1q*dim1q);
                    }

                    KOKKOS_INLINE_FUNCTION
                    F dw1dq (const F &q) const {
                        const F dim2q {Kokkos::max(0., 2.-q)}, dim1q {Kokkos::max(0., 1.-q)};
                        return -2.*(0.25*dim2q*dim2q-dim1q*dim1q);
                    }
                    
                    KOKKOS_INLINE_FUNCTION
                    F dqdr(const F &dr) const {
                        return sgn_kokkos(dr)/dcell;
                    }
            };
#endif
        }
    }
}
#endif