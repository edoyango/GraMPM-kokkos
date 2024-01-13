#ifndef GRAMPM_KOKKOS_FUNCTORS
#define GRAMPM_KOKKOS_FUNCTORS

#include <Kokkos_Core.hpp>
#include <grampm-kokkos-kernels.hpp>
#include <stdio.h>

namespace GraMPM {
    namespace accelerated {
        namespace functors {
            
            template<typename F>
            struct map_gidx {
                const F dcell, minx, miny, minz;
                const int nx, ny, nz;
                // NB const views means the data within the views can be changed, but not the container
                const Kokkos::View<F*[3]> x;
                const Kokkos::View<int*> gidx;
                
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

            template<typename F>
            struct find_neighbour_nodes {

                const F dcell, minx, miny, minz;
                const int nx, ny, nz, npp;
                const Kokkos::View<F*[3]> x;
                const Kokkos::View<int*> gidx;
                const Kokkos::View<int*> pg;
                const Kokkos::View<F*> w;
                const Kokkos::View<F*[3]> dwdx;
                const kernels::kernel<F> knl;

                find_neighbour_nodes(F dcell_, F minx_, F miny_, F minz_, int nx_, int ny_, int nz_, int r, 
                    Kokkos::View<F*[3]> x_, Kokkos::View<int*> gidx_, Kokkos::View<int*> pg_, Kokkos::View<F*> w_, 
                    Kokkos::View<F*[3]> dwdx_, kernels::kernel<F> knl_)
                    : dcell {dcell_}
                    , minx {minx_} 
                    , miny {miny_}
                    , minz {minz_}
                    , nx {nx_}
                    , ny {ny_}
                    , nz {nz_}
                    , npp {r*r*r*8}
                    , x {x_}
                    , gidx {gidx_}
                    , pg {pg_}
                    , w {w_}
                    , dwdx {dwdx_}
                    , knl {knl_}
                {
                }
                
                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    const int idx = gidx(i), idxx = idx/(ny*nz), idxy = (idx % (ny*nz))/nz, idxz = idx-idxx*ny*nz-idxy*nz;
                    int j = i*npp;
                    for (int jdxx = idxx+1-knl.radius; jdxx <= idxx+knl.radius; ++jdxx) {
                        for (int jdxy = idxy+1-knl.radius; jdxy <= idxy+knl.radius; ++jdxy) {
                            for (int jdxz = idxz+1-knl.radius; jdxz <= idxz+knl.radius; ++jdxz) {
                                pg(j) = jdxx*ny*nz + jdxy*nz + jdxz;
                                const F dx = x(i, 0) - (jdxx*dcell+minx), dy = x(i, 1) - (jdxy*dcell+miny), dz = x(i, 2) - (jdxz*dcell+minz);
                                knl(dx, dy, dz, w(j), dwdx(j, 0), dwdx(j, 1), dwdx(j, 2));
                                j++;
                            }
                        }
                    }
                }
            };
        }
    }
}
#endif