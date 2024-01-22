#ifndef GRAMPM_KOKKOS_INTEGRATORS
#define GRAMPM_KOKKOS_INTEGRATORS

#include <cstdlib>

namespace GraMPM {
    namespace integrators {
        
        template<typename F, typename MPMtype>
        void MUSL(MPMtype &myMPM, const F &dt, const size_t &max_timestep, 
            const size_t &print_timestep_interval, const size_t &save_timestep_interval) {

            for (size_t itimestep = 1; itimestep < max_timestep+1; ++itimestep) {

                // print to terminal
                if (itimestep % print_timestep_interval == 0) {
                    std::cout << itimestep << ' ' << dt*itimestep << '\n';
                }

                // get which cell each particle is located within
                myMPM.update_particle_to_cell_map();

                /* get all neighbour nodes of each particle, calculate dx between particles and adjacent nodes, and
                update kernel and kernel gradient values. */ 
                myMPM.find_neighbour_nodes();

                // map particles' mass to nodes
                myMPM.map_p2g_mass();

                // map particles' momentum to nodes
                myMPM.map_p2g_momentum();

                // apply user-defined momentum boundary conditions to grid
                myMPM.g_apply_momentum_boundary_conditions(itimestep, dt);

                // map particles' force to nodes
                myMPM.map_p2g_force();

                // apply user-defined force boundary conditions to grid
                myMPM.g_apply_force_boundary_conditions(itimestep, dt);

                // update nodal momentums
                myMPM.g_update_momentum(dt);

                // map nodal forces to particle accelerations
                myMPM.map_g2p_acceleration();

                // update particles' velocities with calculated accelerations
                myMPM.p_update_velocity(dt);

                // update particles' position
                myMPM.p_update_position(dt);

                // map particles' momentum to nodes, in preparation for updating stress
                myMPM.map_p2g_momentum();

                // apply user-defined momentum boundary conditions to grid
                myMPM.g_apply_momentum_boundary_conditions(itimestep, dt);

                // map nodal velocities to particle strain/spin rates
                myMPM.map_g2p_strainrate();

                // update particles' density
                myMPM.p_update_density(dt);

                // update particles' stress
                myMPM.p_update_stress(dt);

                if (itimestep % save_timestep_interval == 0) {
                    myMPM.d2h();
                    myMPM.p_save_to_file("p_", itimestep);
                    myMPM.g_save_to_file("g_", itimestep);
                }

            }
        }
    }
}
#endif