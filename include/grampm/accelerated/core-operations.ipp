#ifndef GRAMPM_ACCELERATED_CORE_OPERATIONS
#define GRAMPM_ACCELERATED_CORE_OPERATIONS

#include <Kokkos_Core.hpp>

namespace GraMPM {
    namespace accelerated {

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::update_particle_to_cell_map() {
            Kokkos::parallel_for("map particles to grid", m_p_size, f_map_gidx);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::find_neighbour_nodes() {
            Kokkos::parallel_for("find neighbour nodes", m_p_size, f_find_neighbour_nodes);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::map_p2g_mass() {
            Kokkos::deep_copy(m_g_mass.d_view, F(0.));
            Kokkos::parallel_for("map particle mass to grid", m_p_size, f_map_p2g_mass);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::map_p2g_momentum() {
            Kokkos::deep_copy(m_g_momentum.d_view, F(0.));
            Kokkos::parallel_for("map particle momentum to grid", m_p_size, f_map_p2g_momentum);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::map_p2g_force() {
            // update body force
            f_map_p2g_force.bfx = m_body_force[0];
            f_map_p2g_force.bfy = m_body_force[1];
            f_map_p2g_force.bfz = m_body_force[2];

            Kokkos::deep_copy(m_g_force.d_view, F(0.));
            Kokkos::parallel_for("map particle force to grid", m_p_size, f_map_p2g_force);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::map_g2p_acceleration() {
            Kokkos::parallel_for("map grid force/momentum to particles", m_p_size, f_map_g2p_acceleration);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::map_g2p_strainrate() {
            Kokkos::parallel_for("map grid force/momentum to particles' strainrate", m_p_size, 
                f_map_g2p_strainrate);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::g_apply_momentum_boundary_conditions(const int itimestep, const F dt) {
            f_momentum_boundary.itimestep = itimestep;
            f_momentum_boundary.dt = dt;
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({0, 0, 0}, {m_ngrid[0], m_ngrid[1], m_ngrid[2]});
            Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::Rank<3>>>(
                "apply grid momentum boundary conditions", 
                exec_policy,
                f_momentum_boundary
            );
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::g_apply_force_boundary_conditions(const int itimestep, const F dt) {
            f_force_boundary.itimestep = itimestep;
            f_force_boundary.dt = dt;
            Kokkos::MDRangePolicy<Kokkos::Rank<3>> exec_policy({0, 0, 0}, {m_ngrid[0], m_ngrid[1], m_ngrid[2]});
            Kokkos::parallel_for<Kokkos::MDRangePolicy<Kokkos::Rank<3>>>(
                "apply grid force boundary conditions", 
                exec_policy,
                f_force_boundary
            );
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::p_update_stress(const F &dt) {
            f_stress_update.dt = dt;
            Kokkos::parallel_for("update particles' stress", m_p_size, f_stress_update);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::g_update_momentum(const F &dt) {
            f_g_update_momentum.dt = dt;
            Kokkos::parallel_for("update grid momentum", m_g_size, f_g_update_momentum);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::p_update_velocity(const F &dt) {
            f_p_update_velocity.dt = dt;
            Kokkos::parallel_for("update particles' velocity", m_p_size, f_p_update_velocity);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::p_update_position(const F &dt) {
            f_p_update_position.dt = dt;
            Kokkos::parallel_for("update particles' position", m_p_size, f_p_update_position);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::p_update_density(const F &dt) {
            f_p_update_density.dt = dt;
            Kokkos::parallel_for("update particles' density", m_p_size, f_p_update_density);
        }
    }
}
#endif