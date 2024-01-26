#ifndef GRAMPM_KOKKOS
#define GRAMPM_KOKKOS

#include <Kokkos_Core.hpp>
#include <array>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/functors.hpp>

/*============================================================================================================*/

constexpr int dims {3}, voigt_tens_elems {6}, spin_tens_elems {3};

template<typename F> using spatial_view_type = Kokkos::View<F*[dims]>;
template<typename F> using scalar_view_type = Kokkos::View<F*>;
template<typename F> using cauchytensor_view_type = Kokkos::View<F*[voigt_tens_elems]>;
template<typename F> using spintensor_view_type = Kokkos::View<F*[spin_tens_elems]>;
using intscalar_view_type = Kokkos::View<int*>;

/*============================================================================================================*/

namespace GraMPM {

    template<typename F>
    struct particle {
        std::array<F, 3> x, v, a, dxdt;
        std::array<F, 6> sigma, strainrate;
        std::array<F, 3> spinrate;
        F mass, rho;
        particle(const F inx, const F iny, const F inz, const F invx, const F invy, const F invz, 
            const F inmass, const F inrho, const F insigmaxx, const F insigmayy, const F insigmazz, 
            const F insigmaxy, const F insigmaxz, const F insigmayz);
        particle(const F inx, const F iny, const F inz, const F invx, const F invy, const F invz, const F inmass,
            const F inrho, const F insigmaxx, const F insigmayy, const F insigmazz, const F insigmaxy, 
            const F insigmaxz, const F insigmayz, const F inax, const F inay, const F inaz, const F indxdt, 
            const F indydt, const F indzdt, const F instrainratexx, const F instrainrateyy, const F instrainratezz, 
            const F instrainratexy, const F instrainratexz, const F instrainrateyz, const F inspinratexy,
            const F inspinratexz, const F inspinrateyz);
        particle(const std::array<F, 3> inx, const std::array<F, 3> inv, const F inmass, const F inrho, 
            const std::array<F, 6> insigma, const std::array<F, 3> ina, const std::array<F, 3> indxdt, 
            const std::array<F, 6> instrainrate, const std::array<F, 3> inspinrate);
        particle();
    };

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
                const int m_p_size;
                const F m_g_cell_size;
                const std::array<F, dims> m_mingrid, m_maxgrid;
                const std::array<int, dims> m_ngrid;
                const int m_g_size;

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
                MPM_system(const int n, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell);
                // from file
                MPM_system(std::string fname);
                // transfer data to device
                void h2d();
                // transfer data to host
                void d2h();

                // getters/setters
                int p_size() const {return m_p_size;}
                F g_cell_size() const {return m_g_cell_size;}
                std::array<F, dims> g_mingrid() const {return m_mingrid;}
                std::array<F, dims> g_maxgrid() const {return m_maxgrid;}
                std::array<int, dims> g_ngrid() const { return m_ngrid;}
                int g_size() const {return m_g_size;}
                F g_mingridx() const {return m_mingrid[0];}
                F g_mingridy() const {return m_mingrid[1];}
                F g_mingridz() const {return m_mingrid[2];}
                F g_maxgridx() const {return m_maxgrid[0];}
                F g_maxgridy() const {return m_maxgrid[1];}
                F g_maxgridz() const {return m_maxgrid[2];}
                int g_ngridx() const {return m_ngrid[0];}
                int g_ngridy() const {return m_ngrid[1];}
                int g_ngridz() const {return m_ngrid[2];}
                F& g_mass(int i) const {return h_g_mass(i);}
                F& p_x(const int i) {return h_p_x(i, 0);}
                F& p_y(const int i) {return h_p_x(i, 1);}
                F& p_z(const int i) {return h_p_x(i, 2);}
                F& p_vx(const int i) {return h_p_v(i, 0);}
                F& p_vy(const int i) {return h_p_v(i, 1);}
                F& p_vz(const int i) {return h_p_v(i, 2);}
                F& p_ax(const int i) {return h_p_a(i, 0);}
                F& p_ay(const int i) {return h_p_a(i, 1);}
                F& p_az(const int i) {return h_p_a(i, 2);}
                F& p_dxdt(const int i) {return h_p_dxdt(i, 0);}
                F& p_dydt(const int i) {return h_p_dxdt(i, 1);}
                F& p_dzdt(const int i) {return h_p_dxdt(i, 2);}
                F& g_momentumx(const int i) {return h_g_momentum(i, 0);}
                F& g_momentumx(const int i, const int j, const int k) {return h_g_momentum(calc_idx(i, j, k), 0);}
                F& g_momentumy(const int i) {return h_g_momentum(i, 1);}
                F& g_momentumy(const int i, const int j, const int k) {return h_g_momentum(calc_idx(i, j, k), 1);}
                F& g_momentumz(const int i) {return h_g_momentum(i, 2);}
                F& g_momentumz(const int i, const int j, const int k) {return h_g_momentum(calc_idx(i, j, k), 2);}
                F& g_forcex(const int i) {return h_g_force(i, 0);}
                F& g_forcex(const int i, const int j, const int k) {return h_g_force(calc_idx(i, j, k), 0);}
                F& g_forcey(const int i) {return h_g_force(i, 1);}
                F& g_forcey(const int i, const int j, const int k) {return h_g_force(calc_idx(i, j, k), 1);}
                F& g_forcez(const int i) {return h_g_force(i, 2);}
                F& g_forcez(const int i, const int j, const int k) {return h_g_force(calc_idx(i, j, k), 2);}
                F& p_mass(const int i) {return h_p_mass(i);}
                F& p_rho(const int i) {return h_p_rho(i);}
                F& g_mass(const int i) {return h_g_mass(i);}
                F& g_mass(const int i, const int j, const int k) {return h_g_mass(calc_idx(i, j, k));}
                F& p_sigmaxx(const int i) {return h_p_sigma(i, 0);}
                F& p_sigmayy(const int i) {return h_p_sigma(i, 1);}
                F& p_sigmazz(const int i) {return h_p_sigma(i, 2);}
                F& p_sigmaxy(const int i) {return h_p_sigma(i, 3);}
                F& p_sigmaxz(const int i) {return h_p_sigma(i, 4);}
                F& p_sigmayz(const int i) {return h_p_sigma(i, 5);}
                F& p_strainratexx(const int i) {return h_p_strainrate(i, 0);}
                F& p_strainrateyy(const int i) {return h_p_strainrate(i, 1);}
                F& p_strainratezz(const int i) {return h_p_strainrate(i, 2);}
                F& p_strainratexy(const int i) {return h_p_strainrate(i, 3);}
                F& p_strainratexz(const int i) {return h_p_strainrate(i, 4);}
                F& p_strainrateyz(const int i) {return h_p_strainrate(i, 5);}
                F& p_spinratexy(const int i) {return h_p_spinrate(i, 0);}
                F& p_spinratexz(const int i) {return h_p_spinrate(i, 1);}
                F& p_spinrateyz(const int i) {return h_p_spinrate(i, 2);}
                int p_grid_idx(const int i) {return h_p_grid_idx(i);}
                template<typename I>
                std::array<I, dims> unravel_idx(const I &idx) const;
                std::array<int, dims> p_grid_idx_unravelled(const int i) {return unravel_idx<int>(h_p_grid_idx(i));}

                particle<F> p_at(const int i) {
                    return particle<F>(h_p_x(i, 0), h_p_x(i, 1), h_p_x(i, 2), h_p_v(i, 0), h_p_v(i, 1), h_p_v(i, 2), 
                        h_p_mass(i), h_p_rho(i), h_p_sigma(i, 0), h_p_sigma(i, 1), h_p_sigma(i, 2), h_p_sigma(i, 3),
                        h_p_sigma(i, 4), h_p_sigma(i, 5), h_p_a(i, 0), h_p_a(i, 1), h_p_a(i, 2), h_p_dxdt(i, 0), 
                        h_p_dxdt(i, 1), h_p_dxdt(i, 2), h_p_strainrate(i, 0), h_p_strainrate(i, 1), 
                        h_p_strainrate(i, 2), h_p_strainrate(i, 3), h_p_strainrate(i, 4), h_p_strainrate(i, 5), 
                        h_p_spinrate(i, 0), h_p_spinrate(i, 1), h_p_spinrate(i, 2));
                }

                int pg_nn(const int i) {return h_pg_nn(i);}
                int pg_nn(const int i, const int j) {return h_pg_nn(i*pg_npp+j);}
                F pg_w(const int i) {return h_pg_w(i);}
                F pg_w(const int i, const int j) {return h_pg_w(i*pg_npp+j);}
                F pg_dwdx(const int i) {return h_pg_dwdx(i, 0);}
                F pg_dwdx(const int i, const int j) {return h_pg_dwdx(i*pg_npp+j, 0);}
                F pg_dwdy(const int i) {return h_pg_dwdx(i, 1);}
                F pg_dwdy(const int i, const int j) {return h_pg_dwdx(i*pg_npp+j, 1);}
                F pg_dwdz(const int i) {return h_pg_dwdx(i, 2);}
                F pg_dwdz(const int i, const int j) {return h_pg_dwdx(i*pg_npp+j, 2);}

                std::array<F, dims>& body_force() {return m_body_force;}
                F& body_forcex() {return m_body_force[0];}
                F& body_forcey() {return m_body_force[1];}
                F& body_forcez() {return m_body_force[2];}

                void d_zero_grid();
                void h_zero_grid();

                int calc_idx(const int i, const int j, const int k) const;

                void save_to_h5(const std::string &prefix, const int &timestep) const;
                void save_to_h5_async(const std::string &prefix, const int &timestep) const;
                
                // operations
                void update_particle_to_cell_map();
                void find_neighbour_nodes();
                void map_p2g_mass();
                void map_p2g_momentum();
                void map_p2g_force();
                void map_g2p_acceleration();
                void map_g2p_strainrate();
                void g_apply_momentum_boundary_conditions(const int itimestep, const F dt);
                void g_apply_force_boundary_conditions(const int itimestep, const F dt);
                void p_update_stress(const F &dt);
                void g_update_momentum(const F &dt);
                void p_update_velocity(const F &dt);
                void p_update_position(const F &dt);
                void p_update_density(const F &dt);

        };
    }
}

#include <grampm/accelerated/core-constructors.ipp>
#include <grampm/accelerated/core-helpers.ipp>
#include <grampm/accelerated/core-operations.ipp>
#endif