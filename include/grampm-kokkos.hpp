#ifndef GRAMPM_KOKKOS
#define GRAMPM_KOKKOS

#include <Kokkos_Core.hpp>
#include <grampm.hpp>

/*============================================================================================================*/

constexpr size_t dims {3}, voigt_tens_elems {6}, spin_tens_elems {3};

template<typename F> using spatial_view_type = Kokkos::View<F*[dims]>;
template<typename F> using scalar_view_type = Kokkos::View<F*>;
template<typename F> using cauchytensor_view_type = Kokkos::View<F*[voigt_tens_elems]>;
template<typename F> using spintensor_view_type = Kokkos::View<F*[spin_tens_elems]>;
using intscalar_view_type = Kokkos::View<int*>;

/*============================================================================================================*/

namespace GraMPM {

    namespace Kokkos {
        template<typename F>
        class MPM_system {

            protected:
                size_t m_p_size;

                // device views
                spatial_view_type<F> d_p_x, d_p_v, d_p_a, d_p_dxdt;
                cauchytensor_view_type<F> d_p_sigma, d_p_strainrate;
                spintensor_view_type<F> d_p_spinrate;
                scalar_view_type<F> d_p_mass, d_p_rho;

                intscalar_view_type d_p_grid_idx;

                typename spatial_view_type<F>::HostMirror h_p_x, h_p_v, h_p_a, h_p_dxdt;
                typename cauchytensor_view_type<F>::HostMirror h_p_sigma, h_p_strainrate;
                typename spintensor_view_type<F>::HostMirror h_p_spinrate;
                typename scalar_view_type<F>::HostMirror h_p_mass, h_p_rho;

                typename intscalar_view_type::HostMirror h_p_grid_idx;

            public:
                // vector of particles
                MPM_system(std::vector<particle<F>> &pv)
                : m_p_size {pv.size()}
                , d_p_x("Particles' 3D positions", m_p_size)
                , d_p_v("Particles' 3D velocity", m_p_size)
                , d_p_a("Particles' 3D accelerations", m_p_size)
                , d_p_dxdt("Particles' 3D dxdt", m_p_size)
                , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
                , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
                , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
                , d_p_mass("Particles' mass", m_p_size)
                , d_p_rho("Particles' mass", m_p_size)
                , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
                , h_p_x {create_mirror_view(d_p_x)}
                , h_p_v {create_mirror_view(d_p_v)}
                , h_p_a {create_mirror_view(d_p_a)}
                , h_p_dxdt{create_mirror_view(d_p_dxdt)}
                , h_p_sigma{create_mirror_view(d_p_sigma)}
                , h_p_strainrate{create_mirror_view(d_p_strainrate)}
                , h_p_spinrate{create_mirror_view(d_p_spinrate)}
                , h_p_mass{create_mirror_view(d_p_mass)}
                , h_p_rho{create_mirror_view(d_p_rho)}
                , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
                {
                    for (int i = 0; i < m_p_size; ++i) {
                        for (int d = 0; d < dims; ++d) {
                            h_p_x(i, d) = pv[i].x[d];
                            h_p_v(i, d) = pv[i].v[d];
                            h_p_a(i, d) = pv[i].a[d];
                            h_p_dxdt(i, d) = pv[i].dxdt[d];
                        }
                        for (int d = 0; d < voigt_tens_elems; ++d) {
                            h_p_sigma(i, d) = pv[i].sigma[d];
                            h_p_strainrate(i, d) = pv[i].strainrate[d];
                        }
                        for (int d = 0; d < spin_tens_elems; ++d) {
                            h_p_spinrate(i, d) = pv[i].spinrate[d];
                        }
                        h_p_mass(i) = pv[i].mass;
                        h_p_rho(i) = pv[i].rho;
                    }
                }

                MPM_system(const size_t n)
                : m_p_size {n}
                , d_p_x("Particles' 3D positions", m_p_size)
                , d_p_v("Particles' 3D velocity", m_p_size)
                , d_p_a("Particles' 3D accelerations", m_p_size)
                , d_p_dxdt("Particles' 3D dxdt", m_p_size)
                , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
                , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
                , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
                , d_p_mass("Particles' mass", m_p_size)
                , d_p_rho("Particles' mass", m_p_size)
                , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
                , h_p_x {create_mirror_view(d_p_x)}
                , h_p_v {create_mirror_view(d_p_v)}
                , h_p_a {create_mirror_view(d_p_a)}
                , h_p_dxdt{create_mirror_view(d_p_dxdt)}
                , h_p_sigma{create_mirror_view(d_p_sigma)}
                , h_p_strainrate{create_mirror_view(d_p_strainrate)}
                , h_p_spinrate{create_mirror_view(d_p_spinrate)}
                , h_p_mass{create_mirror_view(d_p_mass)}
                , h_p_rho{create_mirror_view(d_p_rho)}
                , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
                {
                }

                void h2d() {
                    deep_copy(d_p_x, h_p_x);
                    deep_copy(d_p_v, h_p_v);
                    deep_copy(d_p_a, h_p_a);
                    deep_copy(d_p_dxdt, h_p_dxdt);
                    deep_copy(d_p_sigma, h_p_sigma);
                    deep_copy(d_p_strainrate, h_p_strainrate);
                    deep_copy(d_p_spinrate, h_p_spinrate);
                    deep_copy(d_p_mass, h_p_mass);
                    deep_copy(d_p_rho, h_p_rho);
                }

                void d2h() {
                    deep_copy(h_p_x, d_p_x);
                    deep_copy(h_p_v, d_p_v);
                    deep_copy(h_p_a, d_p_a);
                    deep_copy(h_p_dxdt, d_p_dxdt);
                    deep_copy(h_p_sigma, d_p_sigma);
                    deep_copy(h_p_strainrate, d_p_strainrate);
                    deep_copy(h_p_spinrate, d_p_spinrate);
                    deep_copy(h_p_mass, d_p_mass);
                    deep_copy(h_p_rho, d_p_rho);
                }

                size_t p_size() const {return m_p_size;}
                F& p_x(const size_t i) {return h_p_x(i, 0);}
                F& p_y(const size_t i) {return h_p_x(i, 1);}
                F& p_z(const size_t i) {return h_p_x(i, 2);}
                F& p_vx(const size_t i) {return h_p_v(i, 0);}
                F& p_vy(const size_t i) {return h_p_v(i, 1);}
                F& p_vz(const size_t i) {return h_p_v(i, 2);}
                F& p_ax(const size_t i) {return h_p_a(i, 0);}
                F& p_ay(const size_t i) {return h_p_a(i, 1);}
                F& p_az(const size_t i) {return h_p_a(i, 2);}
                F& p_dxdt(const size_t i) {return h_p_dxdt(i, 0);}
                F& p_dydt(const size_t i) {return h_p_dxdt(i, 1);}
                F& p_dzdt(const size_t i) {return h_p_dxdt(i, 2);}
                F& p_mass(const size_t i) {return h_p_mass(i);}
                F& p_rho(const size_t i) {return h_p_rho(i);}
                F& p_sigmaxx(const size_t i) {return h_p_sigma(i, 0);}
                F& p_sigmayy(const size_t i) {return h_p_sigma(i, 1);}
                F& p_sigmazz(const size_t i) {return h_p_sigma(i, 2);}
                F& p_sigmaxy(const size_t i) {return h_p_sigma(i, 3);}
                F& p_sigmaxz(const size_t i) {return h_p_sigma(i, 4);}
                F& p_sigmayz(const size_t i) {return h_p_sigma(i, 5);}
                F& p_strainratexx(const size_t i) {return h_p_strainrate(i, 0);}
                F& p_strainrateyy(const size_t i) {return h_p_strainrate(i, 1);}
                F& p_strainratezz(const size_t i) {return h_p_strainrate(i, 2);}
                F& p_strainratexy(const size_t i) {return h_p_strainrate(i, 3);}
                F& p_strainratexz(const size_t i) {return h_p_strainrate(i, 4);}
                F& p_strainrateyz(const size_t i) {return h_p_strainrate(i, 5);}
                F& p_spinratexy(const size_t i) {return h_p_spinrate(i, 0);}
                F& p_spinratexz(const size_t i) {return h_p_spinrate(i, 1);}
                F& p_spinrateyz(const size_t i) {return h_p_spinrate(i, 2);}

                particle<F> p_at(const size_t i) {
                    return particle<F>(h_p_x(i, 0), h_p_x(i, 1), h_p_x(i, 2), h_p_v(i, 0), h_p_v(i, 1), h_p_v(i, 2), 
                        h_p_mass(i), h_p_rho(i), h_p_sigma(i, 0), h_p_sigma(i, 1), h_p_sigma(i, 2), h_p_sigma(i, 3),
                        h_p_sigma(i, 4), h_p_sigma(i, 5), h_p_a(i, 0), h_p_a(i, 1), h_p_a(i, 2), h_p_dxdt(i, 0), 
                        h_p_dxdt(i, 1), h_p_dxdt(i, 2), h_p_strainrate(i, 0), h_p_strainrate(i, 0), 
                        h_p_strainrate(i, 0), h_p_strainrate(i, 0), h_p_strainrate(i, 0), h_p_strainrate(i, 0), 
                        h_p_spinrate(i, 0), h_p_spinrate(i, 1), h_p_spinrate(i, 2));
                }
        };
    }
}
#endif