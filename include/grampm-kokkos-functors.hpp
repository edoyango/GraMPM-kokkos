#ifndef GRAMPM_KOKKOS_FUNCTORS
#define GRAMPM_KOKKOS_FUNCTORS

#include <Kokkos_Core.hpp>
#include <grampm-kokkos-kernels.hpp>
#include <stdio.h>

namespace GraMPM {
    namespace accelerated {
        namespace functors {

            template<typename F>
            struct zero_3d_view {
                const Kokkos::View<F*[3]> data;
                zero_3d_view(Kokkos::View<F*[3]> data_) : data {data_} {}
                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    data(i, 0) = 0.;
                    data(i, 1) = 0.;
                    data(i, 2) = 0.;
                }
            };
            
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

            template<typename F, typename kernel>
            struct find_neighbour_nodes {

                const F dcell, minx, miny, minz;
                const int nx, ny, nz, npp;
                const Kokkos::View<F*[3]> x;
                const Kokkos::View<int*> gidx;
                const Kokkos::View<int*> pg;
                const Kokkos::View<F*> w;
                const Kokkos::View<F*[3]> dwdx;
                const kernel knl;

                find_neighbour_nodes(F dcell_, F minx_, F miny_, F minz_, int nx_, int ny_, int nz_, int r, 
                    Kokkos::View<F*[3]> x_, Kokkos::View<int*> gidx_, Kokkos::View<int*> pg_, Kokkos::View<F*> w_, 
                    Kokkos::View<F*[3]> dwdx_, kernel knl_)
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

            template<typename F>
            struct map_p2g_mass {
                int npp;
                const Kokkos::View<F*> p_mass;
                const Kokkos::View<F*, Kokkos::MemoryTraits<Kokkos::Atomic>> g_mass;
                const Kokkos::View<int*> pg;
                const Kokkos::View<F*> w;
                map_p2g_mass(int npp_, Kokkos::View<F*> p_mass_, Kokkos::View<F*> g_mass_, Kokkos::View<int*> pg_, 
                    Kokkos::View<F*> w_)
                    : npp {npp_}
                    , p_mass {p_mass_}
                    , g_mass {g_mass_}
                    , pg {pg_}
                    , w {w_}
                {
                }
                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    const int jstart = i*npp;
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        // Kokkos::atomic_add(&g_mass(idx), p_mass(i)*w(j));
                        g_mass(idx) += p_mass(i)*w(j);
                    }
                }
            };

            template<typename F>
            struct map_p2g_momentum {
                int npp;
                const Kokkos::View<F*> p_mass;
                const Kokkos::View<F*[3]> p_v;
                const Kokkos::View<F*[3], Kokkos::MemoryTraits<Kokkos::Atomic>> g_momentum;
                const Kokkos::View<int*> pg;
                const Kokkos::View<F*> w;
                map_p2g_momentum(int npp_, Kokkos::View<F*> p_mass_, Kokkos::View<F*[3]> p_v_, Kokkos::View<F*[3]> g_momentum_, Kokkos::View<int*> pg_, 
                    Kokkos::View<F*> w_)
                    : npp {npp_}
                    , p_mass {p_mass_}
                    , p_v {p_v_}
                    , g_momentum {g_momentum_}
                    , pg {pg_}
                    , w {w_}
                {
                }
                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    const int jstart = i*npp;
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        // Kokkos::atomic_add(&g_mass(idx), p_mass(i)*w(j));
                        g_momentum(idx, 0) += p_mass(i)*p_v(i, 0)*w(j);
                        g_momentum(idx, 1) += p_mass(i)*p_v(i, 1)*w(j);
                        g_momentum(idx, 2) += p_mass(i)*p_v(i, 2)*w(j);
                    }
                }
            };

            template<typename F>
            struct map_p2g_force {
                int npp;
                F bfx, bfy, bfz;
                const Kokkos::View<F*> p_mass, p_rho;
                const Kokkos::View<F*[6]> p_sigma;
                const Kokkos::View<F*[3], Kokkos::MemoryTraits<Kokkos::Atomic>> g_force;
                const Kokkos::View<int*> pg;
                const Kokkos::View<F*> w;
                const Kokkos::View<F*[3]> dwdx;
                map_p2g_force(int npp_, Kokkos::View<F*> p_mass_, Kokkos::View<F*> p_rho_, 
                    Kokkos::View<F*[6]> p_sigma_, Kokkos::View<F*[3]> g_force_, Kokkos::View<int*> pg_, 
                    Kokkos::View<F*> w_, Kokkos::View<F*[3]> dwdx_, F bfx_, F bfy_, F bfz_)
                    : npp {npp_}
                    , bfx {bfx_}
                    , bfy {bfy_}
                    , bfz {bfz_}
                    , p_mass {p_mass_}
                    , p_rho {p_rho_}
                    , p_sigma {p_sigma_}
                    , g_force {g_force_}
                    , pg {pg_}
                    , w {w_}
                    , dwdx {dwdx_}
                {
                }
                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    const int jstart = i*npp;
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        g_force(idx, 0) += -p_mass(i)/p_rho(i)*(
                            p_sigma(i, 0)*dwdx(j, 0) +
                            p_sigma(i, 3)*dwdx(j, 1) +
                            p_sigma(i, 4)*dwdx(j, 2)
                        ) + bfx*p_mass(i)*w(j);
                        g_force(idx, 1) += -p_mass(i)/p_rho(i)*(
                            p_sigma(i, 3)*dwdx(j, 0) +
                            p_sigma(i, 1)*dwdx(j, 1) +
                            p_sigma(i, 5)*dwdx(j, 2)
                        ) + bfy*p_mass(i)*w(j);
                        g_force(idx, 2) += -p_mass(i)/p_rho(i)*(
                            p_sigma(i, 4)*dwdx(j, 0) +
                            p_sigma(i, 5)*dwdx(j, 1) +
                            p_sigma(i, 2)*dwdx(j, 2)
                        ) + bfz*p_mass(i)*w(j);
                    }
                }
            };
        }
    }
}
#endif