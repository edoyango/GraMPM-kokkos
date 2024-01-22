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
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/functors.hpp>

/*============================================================================================================*/

constexpr size_t dims {3}, voigt_tens_elems {6}, spin_tens_elems {3};

template<typename F> using spatial_view_type = Kokkos::View<F*[dims]>;
template<typename F> using scalar_view_type = Kokkos::View<F*>;
template<typename F> using cauchytensor_view_type = Kokkos::View<F*[voigt_tens_elems]>;
template<typename F> using spintensor_view_type = Kokkos::View<F*[spin_tens_elems]>;
using intscalar_view_type = Kokkos::View<int*>;

/*============================================================================================================*/

namespace GraMPM {

    namespace accelerated {

        template<typename F>
        struct empty_boundary_func {
            int itimestep;
            F dt;
            const double ngridx, ngridy, ngridz;
            const Kokkos::View<F*[3]> data;
            empty_boundary_func(Kokkos::View<F*[3]> data_, F ngridx_, F ngridy_, F ngridz_)
                : data {data_} 
                , ngridx {ngridx_}
                , ngridy {ngridy_}
                , ngridz {ngridz_}
            {};
            KOKKOS_INLINE_FUNCTION
            void operator()(const int i, const int j, const int k) const {
            }
        };

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary = empty_boundary_func<F>, typename force_boundary = empty_boundary_func<F>>
        class MPM_system {

            public:
                const kernel knl;

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

                const int pg_npp;
                intscalar_view_type d_pg_nn;
                scalar_view_type<F> d_pg_w;
                spatial_view_type<F> d_pg_dwdx;

                typename spatial_view_type<F>::HostMirror h_p_x, h_p_v, h_p_a, h_p_dxdt, h_g_momentum, h_g_force;
                typename cauchytensor_view_type<F>::HostMirror h_p_sigma, h_p_strainrate;
                typename spintensor_view_type<F>::HostMirror h_p_spinrate;
                typename scalar_view_type<F>::HostMirror h_p_mass, h_p_rho, h_g_mass;

                typename intscalar_view_type::HostMirror h_p_grid_idx;

                typename intscalar_view_type::HostMirror h_pg_nn;
                typename scalar_view_type<F>::HostMirror h_pg_w;
                typename spatial_view_type<F>::HostMirror h_pg_dwdx;

                momentum_boundary f_momentum_boundary;
                force_boundary f_force_boundary;

                std::array<F, dims> m_body_force;

                const functors::map_gidx<F> f_map_gidx;
                const functors::find_neighbour_nodes<F, kernel> f_find_neighbour_nodes;
                const functors::map_p2g_mass<F> f_map_p2g_mass;
                const functors::map_p2g_momentum<F> f_map_p2g_momentum;
                functors::map_p2g_force<F> f_map_p2g_force;
                const functors::map_g2p_acceleration<F> f_map_g2p_acceleration;
                const functors::map_g2p_strainrate<F> f_map_g2p_strainrate;
                functors::update_data<F> f_g_update_momentum, f_p_update_velocity, f_p_update_position;
                functors::update_density<F> f_p_update_density;

            public:
                stress_update f_stress_update;
                
                // Constructors
                // vector of particles
                MPM_system(std::vector<particle<F>> &pv, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell);
                // initialize size - fill data later
                MPM_system(const size_t n, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell);
                // from file
                MPM_system(std::string fname, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell);
                // transfer data to device
                void h2d();
                // transfer data to host
                void d2h();

                // getters/setters
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
                int p_grid_idx(const size_t i) {return h_p_grid_idx(i);}
                template<typename I>
                std::array<I, dims> unravel_idx(const I &idx) const;
                std::array<int, dims> p_grid_idx_unravelled(const int i) {return unravel_idx<int>(h_p_grid_idx(i));}

                particle<F> p_at(const size_t i) {
                    return particle<F>(h_p_x(i, 0), h_p_x(i, 1), h_p_x(i, 2), h_p_v(i, 0), h_p_v(i, 1), h_p_v(i, 2), 
                        h_p_mass(i), h_p_rho(i), h_p_sigma(i, 0), h_p_sigma(i, 1), h_p_sigma(i, 2), h_p_sigma(i, 3),
                        h_p_sigma(i, 4), h_p_sigma(i, 5), h_p_a(i, 0), h_p_a(i, 1), h_p_a(i, 2), h_p_dxdt(i, 0), 
                        h_p_dxdt(i, 1), h_p_dxdt(i, 2), h_p_strainrate(i, 0), h_p_strainrate(i, 1), 
                        h_p_strainrate(i, 2), h_p_strainrate(i, 3), h_p_strainrate(i, 4), h_p_strainrate(i, 5), 
                        h_p_spinrate(i, 0), h_p_spinrate(i, 1), h_p_spinrate(i, 2));
                }

                int pg_nn(const size_t i) {return h_pg_nn(i);}
                int pg_nn(const size_t i, const size_t j) {return h_pg_nn(i*pg_npp+j);}
                F pg_w(const size_t i) {return h_pg_w(i);}
                F pg_w(const size_t i, const size_t j) {return h_pg_w(i*pg_npp+j);}
                F pg_dwdx(const size_t i) {return h_pg_dwdx(i, 0);}
                F pg_dwdx(const size_t i, const size_t j) {return h_pg_dwdx(i*pg_npp+j, 0);}
                F pg_dwdy(const size_t i) {return h_pg_dwdx(i, 1);}
                F pg_dwdy(const size_t i, const size_t j) {return h_pg_dwdx(i*pg_npp+j, 1);}
                F pg_dwdz(const size_t i) {return h_pg_dwdx(i, 2);}
                F pg_dwdz(const size_t i, const size_t j) {return h_pg_dwdx(i*pg_npp+j, 2);}

                std::array<F, dims>& body_force() {return m_body_force;}
                F& body_forcex() {return m_body_force[0];}
                F& body_forcey() {return m_body_force[1];}
                F& body_forcez() {return m_body_force[2];}

                void d_zero_grid();
                void h_zero_grid();

                size_t calc_idx(const size_t i, const size_t j, const size_t k) {
                    return i*m_ngrid[1]*m_ngrid[2] + j*m_ngrid[2] + k;
                }

                void p_save_to_file(const std::string &prefix, const int &timestep) const;

                void g_save_to_file(const std::string &prefix, const int &timestep) const;

                void update_particle_to_cell_map() {
                    Kokkos::parallel_for("map particles to grid", m_p_size, f_map_gidx);
                }

                void find_neighbour_nodes() {
                    Kokkos::parallel_for("find neighbour nodes", m_p_size, f_find_neighbour_nodes);
                }

                void map_p2g_mass() {
                    Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_mass, 0.);
                    Kokkos::parallel_for("map particle mass to grid", m_p_size, f_map_p2g_mass);
                }

                void map_p2g_momentum() {
                    Kokkos::parallel_for("zero grid momentum", m_g_size, functors::zero_3d_view<F>(d_g_momentum));
                    Kokkos::parallel_for("map particle momentum to grid", m_p_size, f_map_p2g_momentum);
                }

                void map_p2g_force() {
                    // update body force
                    f_map_p2g_force.bfx = m_body_force[0];
                    f_map_p2g_force.bfy = m_body_force[1];
                    f_map_p2g_force.bfz = m_body_force[2];

                    Kokkos::parallel_for("zero grid force", m_g_size, functors::zero_3d_view<F>(d_g_force));
                    Kokkos::parallel_for("map particle force to grid", m_p_size, f_map_p2g_force);
                }

                void map_g2p_acceleration() {
                    Kokkos::parallel_for("map grid force/momentum to particles", m_p_size, f_map_g2p_acceleration);
                }

                void map_g2p_strainrate() {
                    Kokkos::parallel_for("map grid force/momentum to particles' strainrate", m_p_size, 
                        f_map_g2p_strainrate);
                }

                void g_apply_momentum_boundary_conditions(const int itimestep, const F dt) {
                    f_momentum_boundary.itimestep = itimestep;
                    f_momentum_boundary.dt = dt;
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({0, 0, 0}, {m_ngrid[0], m_ngrid[1], m_ngrid[2]});
                    Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::Rank<3>>>(
                        "apply grid momentum boundary conditions", 
                        exec_policy,
                        f_momentum_boundary
                    );
                }

                void g_apply_force_boundary_conditions(const int itimestep, const F dt) {
                    f_force_boundary.itimestep = itimestep;
                    f_force_boundary.dt = dt;
                    Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({0, 0, 0}, {m_ngrid[0], m_ngrid[1], m_ngrid[2]});
                    Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::Rank<3>>>(
                        "apply grid force boundary conditions", 
                        exec_policy,
                        f_force_boundary
                    );
                }

                void p_update_stress(const F &dt) {
                    f_stress_update.dt = dt;
                    Kokkos::parallel_for("update particles' stress", m_p_size, f_stress_update);
                }

                void g_update_momentum(const F &dt) {
                    f_g_update_momentum.dt = dt;
                    Kokkos::parallel_for("update grid momentum", m_g_size, f_g_update_momentum);
                }

                void p_update_velocity(const F &dt) {
                    f_p_update_velocity.dt = dt;
                    Kokkos::parallel_for("update particles' velocity", m_p_size, f_p_update_velocity);
                }

                void p_update_position(const F &dt) {
                    f_p_update_position.dt = dt;
                    Kokkos::parallel_for("update particles' position", m_p_size, f_p_update_position);
                }

                void p_update_density(const F &dt) {
                    f_p_update_density.dt = dt;
                    Kokkos::parallel_for("update particles' density", m_p_size, f_p_update_density);
                }

        };
    }
}

#include <grampm/accelerated/core-constructors.ipp>
#include <grampm/accelerated/core-helpers.ipp>
#endif