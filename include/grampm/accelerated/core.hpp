#ifndef GRAMPM_KOKKOS
#define GRAMPM_KOKKOS

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <array>
#include <Kokkos_StdAlgorithms.hpp>
#include <string>
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/functors.hpp>
#include <grampm/extra.hpp>

/*============================================================================================================*/

constexpr int dims {3}, voigt_tens_elems {6}, spin_tens_elems {3};

template<typename F> using spatial_view_type = Kokkos::DualView<F*[dims]>;
template<typename F> using scalar_view_type = Kokkos::DualView<F*>;
template<typename F> using cauchytensor_view_type = Kokkos::DualView<F*[voigt_tens_elems]>;
template<typename F> using spintensor_view_type = Kokkos::DualView<F*[spin_tens_elems]>;
using intscalar_view_type = Kokkos::DualView<int*>;

/*============================================================================================================*/

namespace GraMPM {

    template<typename F>
    struct particle {
        std::array<F, 3> x, v, a, dxdt;
        std::array<F, 6> sigma, strainrate;
        std::array<F, 3> spinrate;
        F mass, rho;
        particle(const F inx, const F iny, const F inz, const F inmass,  const F inrho, const F invx = F(0.), 
            const F invy = F(0.), const F invz = F(0.), const F insigmaxx = F(0.), const F insigmayy = F(0.), 
            const F insigmazz = F(0.), const F insigmaxy = F(0.), const F insigmaxz = F(0.), const F insigmayz = F(0.), 
            const F inax = F(0.), const F inay = F(0.), const F inaz = F(0.), const F indxdt = F(0.), 
            const F indydt = F(0.), const F indzdt = F(0.), const F instrainratexx = F(0.), 
            const F instrainrateyy = F(0.), const F instrainratezz = F(0.), const F instrainratexy = F(0.), 
            const F instrainratexz = F(0.), const F instrainrateyz = F(0.), const F inspinratexy = F(0.),
            const F inspinratexz = F(0.), const F inspinrateyz = F(0.));
        particle(const std::array<F, 3> inx, const std::array<F, 3> inv, const F inmass, const F inrho, 
            const std::array<F, 6> insigma = std::array<F, 6>{F(0.), F(0.), F(0.), F(0.), F(0.), F(0.)}, 
            const std::array<F, 3> ina = std::array<F, 6>{F(0.), F(0.), F(0.), F(0.), F(0.), F(0.)}, 
            const std::array<F, 3> indxdt = std::array<F, 3>{F(0.), F(0.), F(0.)}, 
            const std::array<F, 6> instrainrate = std::array<F, 6>{F(0.), F(0.), F(0.), F(0.), F(0.), F(0.)}, 
            const std::array<F, 3> inspinrate = std::array<F, 3>{F(0.), F(0.), F(0.)});
        particle();
    };

    namespace accelerated {

        template<typename F>
        struct empty_boundary_func {
            int itimestep;
            F dt;
            const double ngrid[dims];
            const Kokkos::View<F*[3]> data;
            empty_boundary_func(Kokkos::View<F*[3]> data_, F ngridx_, F ngridy_, F ngridz_)
                : data {data_} 
                , ngrid {ngridx_, ngridy_, ngridz_}
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
                const box<F> m_g_extents;
                const std::array<int, dims> m_ngrid;
                const int m_g_size;
                int procid, numprocs, m_p_size_global;

                // device views
                spatial_view_type<F> m_p_x, m_p_v, m_p_a, m_p_dxdt, m_g_momentum, m_g_force;
                cauchytensor_view_type<F> m_p_sigma, m_p_strainrate;
                spintensor_view_type<F> m_p_spinrate;
                scalar_view_type<F> m_p_mass, m_p_rho, m_g_mass;

                intscalar_view_type m_p_grid_idx;

                const int pg_npp;
                intscalar_view_type m_pg_nn;
                scalar_view_type<F> m_pg_w;
                spatial_view_type<F> m_pg_dwdx;

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

#ifdef GRAMPM_MPI
                const Kokkos::DualView<box<int>*> m_ORB_extents, m_ORB_send_halo, m_ORB_recv_halo;
                int n_ORB_neighbours;
                Kokkos::DualView<int*> m_ORB_neighbours;
#endif

            public:
                stress_update f_stress_update;
                
                // Constructors
                // initialize size - fill data later
                MPM_system(const int n, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell);
                // vector of particles
                MPM_system(std::vector<particle<F>> &pv, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell);
                // from file
                MPM_system(std::string fname);
                // transfer data to device
                void h2d();
                // transfer data to host
                void d2h();

                // getters/setters
                int p_size() const;
                F g_cell_size() const;
                std::array<F, dims> g_mingrid() const;
                std::array<F, dims> g_maxgrid() const;
                std::array<int, dims> g_ngrid() const;
                int g_size() const;
                F g_mingridx() const;
                F g_mingridy() const;
                F g_mingridz() const;
                F g_maxgridx() const;
                F g_maxgridy() const;
                F g_maxgridz() const;
                int g_ngridx() const;
                int g_ngridy() const;
                int g_ngridz() const;
                F& p_x(const int i);
                F& p_y(const int i);
                F& p_z(const int i);
                F& p_vx(const int i);
                F& p_vy(const int i);
                F& p_vz(const int i);
                F& p_ax(const int i);
                F& p_ay(const int i);
                F& p_az(const int i);
                F& p_dxdt(const int i);
                F& p_dydt(const int i);
                F& p_dzdt(const int i);
                F& p_mass(const int i);
                F& p_rho(const int i);
                F& p_sigmaxx(const int i);
                F& p_sigmayy(const int i);
                F& p_sigmazz(const int i);
                F& p_sigmaxy(const int i);
                F& p_sigmaxz(const int i);
                F& p_sigmayz(const int i);
                F& p_strainratexx(const int i);
                F& p_strainrateyy(const int i);
                F& p_strainratezz(const int i);
                F& p_strainratexy(const int i);
                F& p_strainratexz(const int i);
                F& p_strainrateyz(const int i);
                F& p_spinratexy(const int i);
                F& p_spinratexz(const int i);
                F& p_spinrateyz(const int i);
                int p_grid_idx(const int i) const;
                std::array<int, dims> p_grid_idx_unravelled(const int i) const;
                F& g_momentumx(const int i);
                F& g_momentumx(const int i, const int j, const int k);
                F& g_momentumy(const int i);
                F& g_momentumy(const int i, const int j, const int k);
                F& g_momentumz(const int i);
                F& g_momentumz(const int i, const int j, const int k);
                F& g_forcex(const int i);
                F& g_forcex(const int i, const int j, const int k);
                F& g_forcey(const int i);
                F& g_forcey(const int i, const int j, const int k);
                F& g_forcez(const int i);
                F& g_forcez(const int i, const int j, const int k);
                F& g_mass(const int i);
                F& g_mass(const int i, const int j, const int k);

                particle<F> p_at(const int i) const;

                int pg_nn(const int i) const;
                int pg_nn(const int i, const int j) const;
                F pg_w(const int i) const;
                F pg_w(const int i, const int j) const;
                F pg_dwdx(const int i) const;
                F pg_dwdx(const int i, const int j) const;
                F pg_dwdy(const int i) const;
                F pg_dwdy(const int i, const int j) const;
                F pg_dwdz(const int i) const;
                F pg_dwdz(const int i, const int j) const;

                std::array<F, dims>& body_force();
                F& body_forcex();
                F& body_forcey();
                F& body_forcez();

                void d_zero_grid();
                void h_zero_grid();

                template<typename I> std::array<I, dims> unravel_idx(const I &idx) const;
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

#ifdef GRAMPM_MPI
                void ORB_determine_boundaries();
                int ORB_min_idxx(const int i) const;
                int ORB_min_idxy(const int i) const;
                int ORB_min_idxz(const int i) const;
                int ORB_max_idxx(const int i) const;
                int ORB_max_idxy(const int i) const;
                int ORB_max_idxz(const int i) const;
                int ORB_n_neighbours() const;
                int ORB_neighbour(const int i) const;
                int ORB_send_halo_minx(const int neighbour) const;
                int ORB_send_halo_miny(const int neighbour) const;
                int ORB_send_halo_minz(const int neighbour) const;
                int ORB_send_halo_maxx(const int neighbour) const;
                int ORB_send_halo_maxy(const int neighbour) const;
                int ORB_send_halo_maxz(const int neighbour) const;
                int ORB_recv_halo_minx(const int neighbour) const;
                int ORB_recv_halo_miny(const int neighbour) const;
                int ORB_recv_halo_minz(const int neighbour) const;
                int ORB_recv_halo_maxx(const int neighbour) const;
                int ORB_recv_halo_maxy(const int neighbour) const;
                int ORB_recv_halo_maxz(const int neighbour) const;
#endif

        };
    }
}

#include <grampm/accelerated/core-getters-setters.ipp>
#include <grampm/accelerated/core-constructors.ipp>
#include <grampm/accelerated/core-helpers.ipp>
#include <grampm/accelerated/core-operations.ipp>
#ifdef GRAMPM_MPI
#include <grampm/accelerated/mpi.ipp>
#endif
#endif