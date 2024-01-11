#ifndef GRAMPM_KOKKOS
#define GRAMPM_KOKKOS

#include <Kokkos_Core.hpp>
#include <grampm.hpp>
#include <array>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <grampm-kokkos-kernels.hpp>

/*============================================================================================================*/

constexpr size_t dims {3}, voigt_tens_elems {6}, spin_tens_elems {3};

template<typename F> using spatial_view_type = Kokkos::View<F*[dims]>;
template<typename F> using scalar_view_type = Kokkos::View<F*>;
template<typename F> using cauchytensor_view_type = Kokkos::View<F*[voigt_tens_elems]>;
template<typename F> using spintensor_view_type = Kokkos::View<F*[spin_tens_elems]>;
using intscalar_view_type = Kokkos::View<int*>;

/*============================================================================================================*/

static size_t count_line(std::string fname) {
    size_t n = 0;
    std::string tmp;
    std::ifstream ifs(fname);
    while (std::getline(ifs, tmp)) n++;
    return n;
}

namespace GraMPM {

    namespace accelerated {

        template<typename F>
        class MPM_system {

            protected:
                const size_t m_p_size;
                const F m_g_cell_size;
                const std::array<F, dims> m_mingrid, m_maxgrid;
                const std::array<size_t, dims> m_ngrid;
                const size_t m_g_size;

                // device views
                spatial_view_type<F> d_p_x, d_p_v, d_p_a, d_p_dxdt, d_g_momentum, d_g_force;
                cauchytensor_view_type<F> d_p_sigma, d_p_strainrate;
                spintensor_view_type<F> d_p_spinrate;
                scalar_view_type<F> d_p_mass, d_p_rho, d_g_mass;

                intscalar_view_type d_p_grid_idx;

                typename spatial_view_type<F>::HostMirror h_p_x, h_p_v, h_p_a, h_p_dxdt, h_g_momentum, h_g_force;
                typename cauchytensor_view_type<F>::HostMirror h_p_sigma, h_p_strainrate;
                typename spintensor_view_type<F>::HostMirror h_p_spinrate;
                typename scalar_view_type<F>::HostMirror h_p_mass, h_p_rho, h_g_mass;

                typename intscalar_view_type::HostMirror h_p_grid_idx;

                kernels::kernel<F> knl;

            public:
                // vector of particles
                MPM_system(std::vector<particle<F>> &pv, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
                    : m_p_size {pv.size()}
                    , m_g_cell_size {dcell}
                    , m_mingrid {mingrid}
                    , m_maxgrid {maxgrid}
                    , m_ngrid {
                        static_cast<size_t>(std::ceil((maxgrid[0]-mingrid[0])/dcell))+1,
                        static_cast<size_t>(std::ceil((maxgrid[1]-mingrid[1])/dcell))+1,
                        static_cast<size_t>(std::ceil((maxgrid[2]-mingrid[2])/dcell))+1
                    }
                    , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
                    , d_p_x("Particles' 3D positions", m_p_size)
                    , d_p_v("Particles' 3D velocity", m_p_size)
                    , d_p_a("Particles' 3D accelerations", m_p_size)
                    , d_p_dxdt("Particles' 3D dxdt", m_p_size)
                    , d_g_momentum("Grid cells' 3D momentum", m_g_size)
                    , d_g_force("Grid cells' 3D force", m_g_size)
                    , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
                    , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
                    , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
                    , d_p_mass("Particles' mass", m_p_size)
                    , d_p_rho("Particles' mass", m_p_size)
                    , d_g_mass("Grid cells' mass", m_g_size)
                    , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
                    , h_p_x {create_mirror_view(d_p_x)}
                    , h_p_v {create_mirror_view(d_p_v)}
                    , h_p_a {create_mirror_view(d_p_a)}
                    , h_p_dxdt{create_mirror_view(d_p_dxdt)}
                    , h_g_momentum{create_mirror_view(d_g_momentum)}
                    , h_g_force{create_mirror_view(d_g_force)}
                    , h_p_sigma{create_mirror_view(d_p_sigma)}
                    , h_p_strainrate{create_mirror_view(d_p_strainrate)}
                    , h_p_spinrate{create_mirror_view(d_p_spinrate)}
                    , h_p_mass{create_mirror_view(d_p_mass)}
                    , h_p_rho{create_mirror_view(d_p_rho)}
                    , h_g_mass{create_mirror_view(d_g_mass)}
                    , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
                    , knl(dcell)
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

                MPM_system(const size_t n, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
                    : m_p_size {n}
                    , m_g_cell_size {dcell}
                    , m_mingrid {mingrid}
                    , m_maxgrid {maxgrid}
                    , m_ngrid{
                        static_cast<size_t>(std::ceil((maxgrid[0]-mingrid[0])/dcell))+1,
                        static_cast<size_t>(std::ceil((maxgrid[1]-mingrid[1])/dcell))+1,
                        static_cast<size_t>(std::ceil((maxgrid[2]-mingrid[2])/dcell))+1
                    }
                    , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
                    , d_p_x("Particles' 3D positions", m_p_size)
                    , d_p_v("Particles' 3D velocity", m_p_size)
                    , d_p_a("Particles' 3D accelerations", m_p_size)
                    , d_p_dxdt("Particles' 3D dxdt", m_p_size)
                    , d_g_momentum("Grid cells' 3D momentum", m_g_size)
                    , d_g_force("Grid cells' 3D force", m_g_size)
                    , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
                    , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
                    , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
                    , d_p_mass("Particles' mass", m_p_size)
                    , d_p_rho("Particles' mass", m_p_size)
                    , d_g_mass("Grid cells' mass", m_g_size)
                    , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
                    , h_p_x {create_mirror_view(d_p_x)}
                    , h_p_v {create_mirror_view(d_p_v)}
                    , h_p_a {create_mirror_view(d_p_a)}
                    , h_p_dxdt{create_mirror_view(d_p_dxdt)}
                    , h_g_momentum{create_mirror_view(d_g_momentum)}
                    , h_g_force{create_mirror_view(d_g_force)}
                    , h_p_sigma{create_mirror_view(d_p_sigma)}
                    , h_p_strainrate{create_mirror_view(d_p_strainrate)}
                    , h_p_spinrate{create_mirror_view(d_p_spinrate)}
                    , h_p_mass{create_mirror_view(d_p_mass)}
                    , h_p_rho{create_mirror_view(d_p_rho)}
                    , h_g_mass{create_mirror_view(d_g_mass)}
                    , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
                    , knl(dcell)
                    {
                    }

                MPM_system(std::string fname, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
                    : m_p_size {count_line(fname)}
                    , m_g_cell_size {dcell}
                    , m_mingrid {mingrid}
                    , m_maxgrid {maxgrid}
                    , m_ngrid{
                        static_cast<size_t>(std::ceil((maxgrid[0]-mingrid[0])/dcell))+1,
                        static_cast<size_t>(std::ceil((maxgrid[1]-mingrid[1])/dcell))+1,
                        static_cast<size_t>(std::ceil((maxgrid[2]-mingrid[2])/dcell))+1
                    }
                    , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
                    , knl(dcell)
                {
                    std::ifstream file(fname);
                    std::string line, header;
                    // pull out header
                    std::getline(file, header);
                    size_t i = 0;
                    while (std::getline(file, line)) {
                        std::istringstream iss(line);
                        GraMPM::particle<F> p;
                        iss >> p.x[0] >> p.x[1] >> p.x[2] >> p.v[0] >> p.v[1] >> p.v[2] >> p.mass >> p.rho >> p.sigma[0] >> p.sigma[1] >> 
                            p.sigma[2] >> p.sigma[3] >> p.sigma[4] >> p.sigma[5] >> p.a[0] >> p.a[1] >> p.a[2] >> p.dxdt[0] >> p.dxdt[1] >>
                            p.dxdt[2] >> p.strainrate[0] >> p.strainrate[1] >> p.strainrate[2] >> p.strainrate[3] >> p.strainrate[4] >>
                            p.strainrate[5] >> p.spinrate[0] >> p.spinrate[1] >> p.spinrate[2];
                        for (int d = 0; d < dims; ++d) {
                            h_p_x(i, d) = p.x[d];
                            h_p_v(i, d) = p.v[d];
                            h_p_a(i, d) = p.a[d];
                            h_p_dxdt(i, d) = p.dxdt[d];
                        }
                        for (int d = 0; d < voigt_tens_elems; ++d) {
                            h_p_sigma(i, d) = p.sigma[d];
                            h_p_strainrate(i, d) = p.strainrate[d];
                        }
                        for (int d = 0; d < spin_tens_elems; ++d) {
                            h_p_spinrate(i, d) = p.spinrate[d];
                        }
                        h_p_mass(i) = p.mass;
                        h_p_rho(i) = p.rho;
                        i++;
                    }
                }

                void h2d() {
                    deep_copy(d_p_x, h_p_x);
                    deep_copy(d_p_v, h_p_v);
                    deep_copy(d_p_a, h_p_a);
                    deep_copy(d_p_dxdt, h_p_dxdt);
                    deep_copy(d_g_momentum, h_g_momentum);
                    deep_copy(d_g_force, h_g_force);
                    deep_copy(d_p_sigma, h_p_sigma);
                    deep_copy(d_p_strainrate, h_p_strainrate);
                    deep_copy(d_p_spinrate, h_p_spinrate);
                    deep_copy(d_p_mass, h_p_mass);
                    deep_copy(d_p_rho, h_p_rho);
                    deep_copy(d_g_mass, h_g_mass);
                }

                void d2h() {
                    deep_copy(h_p_x, d_p_x);
                    deep_copy(h_p_v, d_p_v);
                    deep_copy(h_p_a, d_p_a);
                    deep_copy(h_p_dxdt, d_p_dxdt);
                    deep_copy(h_g_momentum, d_g_momentum);
                    deep_copy(h_g_force, d_g_force);
                    deep_copy(h_p_sigma, d_p_sigma);
                    deep_copy(h_p_strainrate, d_p_strainrate);
                    deep_copy(h_p_spinrate, d_p_spinrate);
                    deep_copy(h_p_mass, d_p_mass);
                    deep_copy(h_p_rho, d_p_rho);
                    deep_copy(h_g_mass, d_g_mass);
                }

                size_t p_size() const {return m_p_size;}
                F g_cell_size() const {return m_g_cell_size;}
                std::array<F, dims> g_mingrid() const {return m_mingrid;}
                std::array<F, dims> g_maxgrid() const {return m_maxgrid;}
                std::array<size_t, dims> g_ngrid() const { return m_ngrid;}
                size_t g_size() const {return m_g_size;}
                F g_mingridx() const {return m_mingrid[0];}
                F g_mingridy() const {return m_mingrid[1];}
                F g_mingridz() const {return m_mingrid[2];}
                F g_maxgridx() const {return m_maxgrid[0];}
                F g_maxgridy() const {return m_maxgrid[1];}
                F g_maxgridz() const {return m_maxgrid[2];}
                size_t g_ngridx() const {return m_ngrid[0];}
                size_t g_ngridy() const {return m_ngrid[1];}
                size_t g_ngridz() const {return m_ngrid[2];}
                F& g_mass(size_t i) const {return h_g_mass(i);}
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
                F& g_momentumx(const size_t i) {return h_g_momentum(i, 0);}
                F& g_momentumx(const size_t i, const size_t j, const size_t k) {return h_g_momentum(calc_idx(i, j, k), 0);}
                F& g_momentumy(const size_t i) {return h_g_momentum(i, 1);}
                F& g_momentumy(const size_t i, const size_t j, const size_t k) {return h_g_momentum(calc_idx(i, j, k), 1);}
                F& g_momentumz(const size_t i) {return h_g_momentum(i, 2);}
                F& g_momentumz(const size_t i, const size_t j, const size_t k) {return h_g_momentum(calc_idx(i, j, k), 2);}
                F& g_forcex(const size_t i) {return h_g_force(i, 0);}
                F& g_forcex(const size_t i, const size_t j, const size_t k) {return h_g_force(calc_idx(i, j, k), 0);}
                F& g_forcey(const size_t i) {return h_g_force(i, 1);}
                F& g_forcey(const size_t i, const size_t j, const size_t k) {return h_g_force(calc_idx(i, j, k), 1);}
                F& g_forcez(const size_t i) {return h_g_force(i, 2);}
                F& g_forcez(const size_t i, const size_t j, const size_t k) {return h_g_force(calc_idx(i, j, k), 2);}
                F& p_mass(const size_t i) {return h_p_mass(i);}
                F& p_rho(const size_t i) {return h_p_rho(i);}
                F& g_mass(const size_t i) {return h_g_mass(i);}
                F& g_mass(const size_t i, const size_t j, const size_t k) {return h_g_mass(calc_idx(i, j, k));}
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
                        h_p_dxdt(i, 1), h_p_dxdt(i, 2), h_p_strainrate(i, 0), h_p_strainrate(i, 1), 
                        h_p_strainrate(i, 2), h_p_strainrate(i, 3), h_p_strainrate(i, 4), h_p_strainrate(i, 5), 
                        h_p_spinrate(i, 0), h_p_spinrate(i, 1), h_p_spinrate(i, 2));
                }

                void d_zero_grid() {
                    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_momentum, 0.);
                    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_force, 0.);
                    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_mass, 0.);
                }
                void h_zero_grid() {
                    for (size_t i = 0; i < m_g_size; ++i) {
                        h_g_momentum(i, 0) = 0.;
                        h_g_momentum(i, 1) = 0.;
                        h_g_momentum(i, 2) = 0.;
                        h_g_force(i, 0) = 0.;
                        h_g_force(i, 1) = 0.;
                        h_g_force(i, 2) = 0.;
                        h_g_mass(i) = 0.;
                    }
                }

                size_t calc_idx(const size_t i, const size_t j, const size_t k) {
                    return i*m_ngrid[1]*m_ngrid[2] + j*m_ngrid[2] + k;
                }

                void p_save_to_file(const std::string &prefix, const int &timestep) const {

                    // convert timestep number to string (width of 7 chars, for up to 9,999,999,999 timesteps)
                    std::string str_timestep = std::to_string(timestep);
                    str_timestep = std::string(7-str_timestep.length(), '0') + str_timestep;

                    std::string fname {prefix + str_timestep};

                    std::ofstream outfile(fname);

                    const int i_width = 7, f_width = 12, f_precision=10;

                    outfile << std::setw(i_width) << "id" << ' '
                            << std::setw(f_width) << "x" << ' '
                            << std::setw(f_width) << "y" << ' '
                            << std::setw(f_width) << "z" << ' '
                            << std::setw(f_width) << "vx" << ' '
                            << std::setw(f_width) << "vy" << ' '
                            << std::setw(f_width) << "vz" << ' '
                            << std::setw(f_width) << "mass" << ' '
                            << std::setw(f_width) << "rho" << ' '
                            << std::setw(f_width) << "sigmaxx" << ' '
                            << std::setw(f_width) << "sigmayy" << ' '
                            << std::setw(f_width) << "sigmazz" << ' '
                            << std::setw(f_width) << "sigmaxy" << ' '
                            << std::setw(f_width) << "sigmaxz" << ' '
                            << std::setw(f_width) << "sigmayz" << ' '
                            << std::setw(f_width) << "ax" << ' '
                            << std::setw(f_width) << "ay" << ' '
                            << std::setw(f_width) << "az" << ' '
                            << std::setw(f_width) << "dxdt" << ' '
                            << std::setw(f_width) << "dydt" << ' '
                            << std::setw(f_width) << "dzdt" << ' '
                            << std::setw(f_width) << "strainratexx" << ' '
                            << std::setw(f_width) << "strainrateyy" << ' '
                            << std::setw(f_width) << "strainratezz" << ' '
                            << std::setw(f_width) << "strainratexy" << ' '
                            << std::setw(f_width) << "strainratexz" << ' '
                            << std::setw(f_width) << "strainrateyz" << ' '
                            << std::setw(f_width) << "spinratexy" << ' '
                            << std::setw(f_width) << "spinratexz" << ' '
                            << std::setw(f_width) << "spinrateyz" << ' '
                            << '\n';

                    for (size_t i = 0; i < m_p_size; ++i) {
                        outfile << std::setw(i_width) << i << ' ' << std::setprecision(f_precision)
                                << std::setw(f_width) << std::fixed << h_p_x(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_x(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_x(i, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_p_v(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_v(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_v(i, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_p_mass(i) << ' '
                                << std::setw(f_width) << std::fixed << h_p_rho(i) << ' '
                                << std::setw(f_width) << std::fixed << h_p_sigma(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_sigma(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_sigma(i, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_p_sigma(i, 3) << ' '
                                << std::setw(f_width) << std::fixed << h_p_sigma(i, 4) << ' '
                                << std::setw(f_width) << std::fixed << h_p_sigma(i, 5) << ' '
                                << std::setw(f_width) << std::fixed << h_p_a(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_a(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_a(i, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_p_dxdt(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_dxdt(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_dxdt(i, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_p_strainrate(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_strainrate(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_strainrate(i, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_p_strainrate(i, 3) << ' '
                                << std::setw(f_width) << std::fixed << h_p_strainrate(i, 4) << ' '
                                << std::setw(f_width) << std::fixed << h_p_strainrate(i, 5) << ' '
                                << std::setw(f_width) << std::fixed << h_p_spinrate(i, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_p_spinrate(i, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_p_spinrate(i, 2) << ' '
                                << '\n';
                    }
                }
        };
    }
}
#endif