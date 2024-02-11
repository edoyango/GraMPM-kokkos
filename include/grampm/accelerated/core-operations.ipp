#ifndef GRAMPM_ACCELERATED_CORE_OPERATIONS
#define GRAMPM_ACCELERATED_CORE_OPERATIONS

#include <Kokkos_Core.hpp>

namespace GraMPM {
    namespace accelerated {

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::update_particle_to_cell_map() {
            Kokkos::parallel_for(
                "map particles to grid", 
                m_p_size, 
                functors::map_gidx<F>(
                    m_g_cell_size,
                    m_g_extents.min,
                    m_ngrid.data(),
                    m_p_x.d_view,
                    m_p_grid_idx.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::find_neighbour_nodes() {
            Kokkos::parallel_for(
                "find neighbour nodes", 
                m_p_size, 
                functors::find_neighbour_nodes<F, K>(
                    m_g_cell_size,
                    m_g_extents.min,
                    m_ngrid.data(),
                    static_cast<int>(knl.radius),
                    m_p_x.d_view,
                    m_p_grid_idx.d_view,
                    m_pg_nn.d_view,
                    m_pg_w.d_view,
                    m_pg_dwdx.d_view,
                    knl
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::map_p2g_mass() {
            Kokkos::deep_copy(m_g_mass.d_view, F(0.));
            Kokkos::parallel_for(
                "map particle mass to grid", 
                m_p_size, 
                functors::map_p2g_mass<F>(
                    pg_npp,
                    m_p_mass.d_view,
                    m_g_mass.d_view, 
                    m_pg_nn.d_view, 
                    m_pg_w.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::map_p2g_momentum() {
            Kokkos::deep_copy(m_g_momentum.d_view, F(0.));
            Kokkos::parallel_for(
                "map particle momentum to grid", 
                m_p_size, 
                functors::map_p2g_momentum<F>(
                    pg_npp,
                    m_p_mass.d_view,
                    m_p_v.d_view,
                    m_g_momentum.d_view,
                    m_pg_nn.d_view,
                    m_pg_w.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::map_p2g_force() {
            Kokkos::deep_copy(m_g_force.d_view, F(0.));
            Kokkos::parallel_for(
                "map particle force to grid", 
                m_p_size, 
                functors::map_p2g_force<F>(
                    pg_npp,
                    m_p_mass.d_view,
                    m_p_rho.d_view,
                    m_p_sigma.d_view,
                    m_g_force.d_view,
                    m_pg_nn.d_view,
                    m_pg_w.d_view,
                    m_pg_dwdx.d_view,
                    m_body_force[0],
                    m_body_force[1],
                    m_body_force[2]
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::map_g2p_acceleration() {
            Kokkos::parallel_for(
                "map grid force/momentum to particles", 
                m_p_size, 
                functors::map_g2p_acceleration<F>(
                    pg_npp,
                    m_p_a.d_view,
                    m_g_force.d_view,
                    m_p_dxdt.d_view,
                    m_g_momentum.d_view,
                    m_g_mass.d_view,
                    m_pg_w.d_view,
                    m_pg_nn.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::map_g2p_strainrate() {
            Kokkos::parallel_for(
                "map grid force/momentum to particles' strainrate", 
                m_p_size, 
                functors::map_g2p_strainrate<F>(
                    pg_npp,
                    m_p_strainrate.d_view,
                    m_p_spinrate.d_view,
                    m_g_momentum.d_view,
                    m_pg_dwdx.d_view,
                    m_g_mass.d_view,
                    m_pg_nn.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::g_apply_momentum_boundary_conditions(const int itimestep, const F dt) {
            Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::Rank<3>>>(
                "apply grid momentum boundary conditions", 
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0, 0, 0},
                    {m_ngrid[0], m_ngrid[1], m_ngrid[2]}
                ),
                MB(
                    m_g_momentum.d_view,
                    m_ngrid[0],
                    m_ngrid[1],
                    m_ngrid[2],
                    dt, itimestep
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::g_apply_force_boundary_conditions(const int itimestep, const F dt) {
            Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::Rank<3>>>(
                "apply grid force boundary conditions", 
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0, 0, 0},
                    {m_ngrid[0], m_ngrid[1], m_ngrid[2]}
                ),
                FB(
                    m_g_force.d_view,
                    m_ngrid[0],
                    m_ngrid[1],
                    m_ngrid[2],
                    dt, itimestep
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::p_update_stress(const F &dt) {
            f_stress_update.dt = dt;
            Kokkos::parallel_for("update particles' stress", m_p_size, f_stress_update);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::g_update_momentum(const F &dt) {
            Kokkos::parallel_for(
                "update grid momentum", 
                m_g_size, 
                functors::update_data<F>(
                    dt,
                    m_g_momentum.d_view,
                    m_g_force.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::p_update_velocity(const F &dt) {
            Kokkos::parallel_for(
                "update particles' velocity", 
                m_p_size, 
                functors::update_data<F>(
                    dt,
                    m_p_v.d_view,
                    m_p_a.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::p_update_position(const F &dt) {
            Kokkos::parallel_for(
                "update particles' position", 
                m_p_size, 
                functors::update_data<F>(
                    dt,
                    m_p_x.d_view,
                    m_p_dxdt.d_view
                )
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::p_update_density(const F &dt) {
            Kokkos::parallel_for(
                "update particles' density", 
                m_p_size, 
                functors::update_density<F>(
                    dt,
                    m_p_rho.d_view,
                    m_p_strainrate.d_view
                )
            );
        }
    }
}
#endif