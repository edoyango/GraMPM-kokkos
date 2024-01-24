#ifndef GRAMPM_KOKKOS_FUNCTORS
#define GRAMPM_KOKKOS_FUNCTORS

#include <Kokkos_Core.hpp>
#include <grampm/accelerated/kernels.hpp>
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
                    data(i, 0) = F(0.);
                    data(i, 1) = F(0.);
                    data(i, 2) = F(0.);
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
                const int npp;
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
                    const F massi = p_mass(i);
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        // Kokkos::atomic_add(&g_mass(idx), p_mass(i)*w(j));
                        g_mass(idx) += massi*w(j);
                    }
                }
            };

            template<typename F>
            struct map_p2g_momentum {
                const int npp;
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
                    const F massi = p_mass(i), vxi = p_v(i, 0), vyi = p_v(i, 1), vzi = p_v(i, 2);
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        const F massiwj = massi*w(j);
                        // Kokkos::atomic_add(&g_mass(idx), p_mass(i)*w(j));
                        g_momentum(idx, 0) += vxi*massiwj;
                        g_momentum(idx, 1) += vyi*massiwj;
                        g_momentum(idx, 2) += vzi*massiwj;
                    }
                }
            };

            template<typename F>
            struct map_p2g_force {
                const int npp;
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
                    const F massi = p_mass(i), vi = massi/p_rho(i), sigixx = p_sigma(i, 0), sigiyy = p_sigma(i, 1), 
                        sigizz = p_sigma(i, 2), sigixy = p_sigma(i, 3), sigixz = p_sigma(i, 4), sigiyz = p_sigma(i, 5);
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        const F massiwj = p_mass(i)*w(j), dwdxj = dwdx(j, 0), dwdyj = dwdx(j, 1), dwdzj = dwdx(j, 2);
                        g_force(idx, 0) += -vi*(
                            sigixx*dwdxj +
                            sigixy*dwdyj +
                            sigixz*dwdzj
                        ) + bfx*massiwj;
                        g_force(idx, 1) += -vi*(
                            sigixy*dwdxj +
                            sigiyy*dwdyj +
                            sigiyz*dwdzj
                        ) + bfy*massiwj;
                        g_force(idx, 2) += -vi*(
                            sigixz*dwdxj +
                            sigiyz*dwdyj +
                            sigizz*dwdzj
                        ) + bfz*massiwj;
                    }
                }
            };

            template<typename F>
            struct map_g2p_acceleration {
                const int npp;
                const Kokkos::View<F*[3]> p_a, g_force, p_dxdt, g_momentum;
                const Kokkos::View<F*> g_mass, w;
                const Kokkos::View<int*> pg;
                map_g2p_acceleration(int npp_, Kokkos::View<F*[3]> p_a_, Kokkos::View<F*[3]> g_force_, 
                    Kokkos::View<F*[3]> p_dxdt_, Kokkos::View<F*[3]> g_momentum_, Kokkos::View<F*> g_mass_, 
                    Kokkos::View<F*> w_, Kokkos::View<int*> pg_)
                    : npp {npp_}
                    , p_a {p_a_}
                    , g_force {g_force_}
                    , p_dxdt {p_dxdt_}
                    , g_momentum {g_momentum_}
                    , g_mass {g_mass_}
                    , w {w_}
                    , pg {pg_}
                {}

                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
<<<<<<< HEAD
                    F axi = F(0.);
                    F ayi = F(0.);
                    F azi = F(0.);
                    F dxdti = F(0.);
                    F dydti = F(0.);
                    F dzdti = F(0.);
=======
                    p_a(i, 0) = F(0.);
                    p_a(i, 1) = F(0.);
                    p_a(i, 2) = F(0.);
                    p_dxdt(i, 0) = F(0.);
                    p_dxdt(i, 1) = F(0.);
                    p_dxdt(i, 2) = F(0.);
>>>>>>> 9b1dc176a5b46032ad768ce26708ceaf8dab11bc
                    const int jstart = i*npp;
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        const F wjmassj = w(j)/g_mass(idx);
                        axi += g_force(idx, 0)*wjmassj;
                        ayi += g_force(idx, 1)*wjmassj;
                        azi += g_force(idx, 2)*wjmassj;
                        dxdti += g_momentum(idx, 0)*wjmassj;
                        dydti += g_momentum(idx, 1)*wjmassj;
                        dzdti += g_momentum(idx, 2)*wjmassj;
                    }
                    p_a(i, 0) = axi;
                    p_a(i, 1) = ayi;
                    p_a(i, 2) = azi;
                    p_dxdt(i, 0) = dxdti;
                    p_dxdt(i, 1) = dydti;
                    p_dxdt(i, 2) = dzdti;
                }
            };

            template<typename F>
            struct map_g2p_strainrate {
                const int npp;
                const Kokkos::View<F*[6]> p_strainrate;
                const Kokkos::View<F*[3]> p_spinrate, g_momentum, dwdx;
                const Kokkos::View<F*> g_mass;
                const Kokkos::View<int*> pg;

                map_g2p_strainrate(int npp_, Kokkos::View<F*[6]> p_strainrate_, 
                    Kokkos::View<F*[3]> p_spinrate_, Kokkos::View<F*[3]> g_momentum_,
                    Kokkos::View<F*[3]> dwdx_, Kokkos::View<F*> g_mass_, Kokkos::View<int*> pg_)
                    : npp {npp_}
                    , p_strainrate {p_strainrate_}
                    , p_spinrate {p_spinrate_}
                    , g_momentum {g_momentum_}
                    , dwdx {dwdx_}
                    , g_mass {g_mass_}
                    , pg {pg_}
                {}

                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
<<<<<<< HEAD
                    F strainratexxi = 0.;
                    F strainrateyyi = 0.;
                    F strainratezzi = 0.;
                    F strainratexyi = 0.;
                    F strainratexzi = 0.;
                    F strainrateyzi = 0.;
                    F spinratexyi = 0.;
                    F spinratexzi = 0.;
                    F spinrateyzi = 0.;
                    const int jstart = i*npp;
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        const F massj = g_mass(idx), dwdxjmassj = dwdx(j, 0)/massj, dwdyjmassj = dwdx(j, 1)/massj, 
                            dwdzjmassj = dwdx(j, 2)/massj, momentumxj = g_momentum(idx, 0), 
                            momentumyj = g_momentum(idx, 1), momentumzj = g_momentum(idx, 2);
                        strainratexxi += momentumxj*dwdxjmassj;
                        strainrateyyi += momentumyj*dwdyjmassj;
                        strainratezzi += momentumzj*dwdzjmassj;
                        strainratexyi += F(0.5)*(momentumyj*dwdxjmassj + momentumxj*dwdyjmassj);
                        strainratexzi += F(0.5)*(momentumzj*dwdxjmassj + momentumxj*dwdzjmassj);
                        strainrateyzi += F(0.5)*(momentumzj*dwdyjmassj + momentumyj*dwdzjmassj);
                        spinratexyi += F(0.5)*(momentumxj*dwdyjmassj - momentumyj*dwdxjmassj);
                        spinratexzi += F(0.5)*(momentumxj*dwdzjmassj - momentumzj*dwdxjmassj);
                        spinrateyzi += F(0.5)*(momentumyj*dwdzjmassj - momentumzj*dwdyjmassj);
=======
                    p_strainrate(i, 0) = F(0.);
                    p_strainrate(i, 1) = F(0.);
                    p_strainrate(i, 2) = F(0.);
                    p_strainrate(i, 3) = F(0.);
                    p_strainrate(i, 4) = F(0.);
                    p_strainrate(i, 5) = F(0.);
                    p_spinrate(i, 0) = F(0.);
                    p_spinrate(i, 1) = F(0.);
                    p_spinrate(i, 2) = F(0.);
                    const int jstart = i*npp;
                    for (int j = jstart; j < jstart + npp; ++j) {
                        const int idx = pg(j);
                        p_strainrate(i, 0) += g_momentum(idx, 0)/g_mass(idx)*dwdx(j, 0);
                        p_strainrate(i, 1) += g_momentum(idx, 1)/g_mass(idx)*dwdx(j, 1);
                        p_strainrate(i, 2) += g_momentum(idx, 2)/g_mass(idx)*dwdx(j, 2);
                        p_strainrate(i, 3) += F(0.5)*(g_momentum(idx, 1)/g_mass(idx)*dwdx(j, 0) +
                            g_momentum(idx, 0)/g_mass(idx)*dwdx(j, 1));
                        p_strainrate(i, 4) += F(0.5)*(g_momentum(idx, 2)/g_mass(idx)*dwdx(j, 0) +
                            g_momentum(idx, 0)/g_mass(idx)*dwdx(j, 2));
                        p_strainrate(i, 5) += F(0.5)*(g_momentum(idx, 2)/g_mass(idx)*dwdx(j, 1) +
                            g_momentum(idx, 1)/g_mass(idx)*dwdx(j, 2));
                        p_spinrate(i, 0) += F(0.5)*(dwdx(j, 1)*g_momentum(idx, 0)/g_mass(idx) -
                            dwdx(j, 0)*g_momentum(idx, 1)/g_mass(idx));
                        p_spinrate(i, 1) += F(0.5)*(dwdx(j, 2)*g_momentum(idx, 0)/g_mass(idx) - 
                            dwdx(j, 0)*g_momentum(idx, 2)/g_mass(idx));
                        p_spinrate(i, 2) += F(0.5)*(dwdx(j, 2)*g_momentum(idx, 1)/g_mass(idx) - 
                            dwdx(j, 1)*g_momentum(idx, 2)/g_mass(idx));
>>>>>>> 9b1dc176a5b46032ad768ce26708ceaf8dab11bc
                    }
                    p_strainrate(i, 0) = strainratexxi;
                    p_strainrate(i, 1) = strainrateyyi;
                    p_strainrate(i, 2) = strainratezzi;
                    p_strainrate(i, 3) = strainratexyi;
                    p_strainrate(i, 4) = strainratexzi;
                    p_strainrate(i, 5) = strainrateyzi;
                    p_spinrate(i, 0) = spinratexyi;
                    p_spinrate(i, 1) = spinratexzi;
                    p_spinrate(i, 2) = spinrateyzi;
                }
            };

            template<typename F>
            struct update_data {
                double dt;
                const Kokkos::View<F*[3]> data, update;

                update_data(Kokkos::View<F*[3]> data_, Kokkos::View<F*[3]> update_)
                    : data {data_}
                    , update {update_}
                {}

                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    data(i, 0) += dt*update(i, 0);
                    data(i, 1) += dt*update(i, 1);
                    data(i, 2) += dt*update(i, 2);
                }
            };

            template<typename F>
            struct update_density {
                double dt;
                const Kokkos::View<F*> p_density;
                const Kokkos::View<F*[6]> p_strainrate;

                update_density(Kokkos::View<F*> p_density_, Kokkos::View<F*[6]> p_strainrate_)
                    : p_density {p_density_}
                    , p_strainrate {p_strainrate_}
                {}

                KOKKOS_INLINE_FUNCTION
                void operator()(const int i) const {
                    p_density(i) /= F(1.) + dt*(p_strainrate(i, 0)+p_strainrate(i, 1)+p_strainrate(i, 2));
                }
            };
        }
    }
}
#endif