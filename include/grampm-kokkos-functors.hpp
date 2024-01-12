#ifndef GRAMPM_KOKKOS_FUNCTORS
#define GRAMPM_KOKKOS_FUNCTORS

#include <Kokkos_Core.hpp>

namespace GraMPM {
    namespace accelerated {
        namespace functors {
            
            template<typename F>
            struct map_gidx {
                const F dcell, minx, miny, minz;
                const int nx, ny, nz;
                const Kokkos::View<F*[3]> x;
                Kokkos::View<int*> gidx;
                
                map_gidx(F dcell_, F minx_, F miny_, F minz_, int nx_, int ny_, int nz_, Kokkos::View<F*[3]> x_, 
                    Kokkos::View<int*> gidx_)
                    : dcell {dcell_}
                    , minx {minx_}
                    , miny {miny_}
                    , minz {minz_}
                    , nx {nx_}
                    , ny {ny_}
                    , nz {nz_}
                    , x {x_}
                    , gidx {gidx_}
                {
                }

                KOKKOS_INLINE_FUNCTION 
                void operator()(const int i) const {
                    gidx(i) = static_cast<int>((x(i, 0)-minx)/dcell)*ny*nz +
                    static_cast<int>((x(i, 1)-miny)/dcell)*nz +
                    static_cast<int>((x(i, 2)-minz)/dcell);
                }
            };
        }
    }
}
#endif