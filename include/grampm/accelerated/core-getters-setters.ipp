#ifndef GRAMPM_ACCELERATED_CORE_GETTERS_SETTERS
#define GRAMPM_ACCELERATED_CORE_GETTERS_SETTERS

namespace GraMPM {
    namespace accelerated {
        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::p_size() const {
            return m_p_size;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_cell_size() const {
            return m_g_cell_size;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        std::array<F, dims> MPM_system<F, K, SU, MB, FB>::g_mingrid() const {
            return std::array<F, dims>{m_g_extents.min[0], m_g_extents.min[1], m_g_extents.min[2]};
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        std::array<F, dims> MPM_system<F, K, SU, MB, FB>::g_maxgrid() const {
            return std::array<F, dims>{m_g_extents.max[0], m_g_extents.max[1], m_g_extents.max[2]};
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        std::array<int, dims> MPM_system<F, K, SU, MB, FB>::g_ngrid() const {
            return m_ngrid;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::g_size() const {
            return m_g_size;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_mingridx() const {
            return m_g_extents.min[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_mingridy() const {
            return m_g_extents.min[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_mingridz() const {
            return m_g_extents.min[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_maxgridx() const {
            return m_g_extents.max[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_maxgridy() const {
            return m_g_extents.max[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::g_maxgridz() const {
            return m_g_extents.max[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::g_ngridx() const {
            return m_ngrid[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::g_ngridy() const {
            return m_ngrid[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::g_ngridz() const {
            return m_ngrid[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_x(const int i) {
            return h_p_x(i, 0);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_y(const int i) {
            return h_p_x(i, 1);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_z(const int i) {
            return h_p_x(i, 2);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_vx(const int i) {
            return h_p_v(i, 0);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_vy(const int i) {
            return h_p_v(i, 1);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_vz(const int i) {
            return h_p_v(i, 2);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_ax(const int i) {
            return h_p_a(i, 0);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_ay(const int i) {
            return h_p_a(i, 1);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_az(const int i) {
            return h_p_a(i, 2);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_dxdt(const int i) {
            return h_p_dxdt(i, 0);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_dydt(const int i) {
            return h_p_dxdt(i, 1);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_dzdt(const int i) {
            return h_p_dxdt(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_mass(const int i) {
            return h_p_mass(i);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_rho(const int i) {
            return h_p_rho(i);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_sigmaxx(const int i) {
            return h_p_sigma(i, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_sigmayy(const int i) {
            return h_p_sigma(i, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_sigmazz(const int i) {
            return h_p_sigma(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_sigmaxy(const int i) {
            return h_p_sigma(i, 3);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_sigmaxz(const int i) {
            return h_p_sigma(i, 4);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_sigmayz(const int i) {
            return h_p_sigma(i, 5);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_strainratexx(const int i) {
            return h_p_strainrate(i, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_strainrateyy(const int i) {
            return h_p_strainrate(i, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_strainratezz(const int i) {
            return h_p_strainrate(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_strainratexy(const int i) {
            return h_p_strainrate(i, 3);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_strainratexz(const int i) {
            return h_p_strainrate(i, 4);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_strainrateyz(const int i) {
            return h_p_strainrate(i, 5);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_spinratexy(const int i) {
            return h_p_spinrate(i, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_spinratexz(const int i) {
            return h_p_spinrate(i, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::p_spinrateyz(const int i) {
            return h_p_spinrate(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::p_grid_idx(const int i) const {
            return h_p_grid_idx(i);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        std::array<int, dims> MPM_system<F, K, SU, MB, FB>::p_grid_idx_unravelled(const int i) const {
            return unravel_idx<int>(
                h_p_grid_idx(i)
            );
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        particle<F> MPM_system<F, K, SU, MB, FB>::p_at(const int i) const {
            return particle<F>(h_p_x(i, 0), h_p_x(i, 1), h_p_x(i, 2), 
                h_p_mass(i), 
                h_p_rho(i), 
                h_p_v(i, 0), h_p_v(i, 1), h_p_v(i, 2), 
                h_p_sigma(i, 0), h_p_sigma(i, 1), h_p_sigma(i, 2), h_p_sigma(i, 3), h_p_sigma(i, 4), h_p_sigma(i, 5), 
                h_p_a(i, 0), h_p_a(i, 1), h_p_a(i, 2), 
                h_p_dxdt(i, 0), h_p_dxdt(i, 1), h_p_dxdt(i, 2), 
                h_p_strainrate(i, 0), h_p_strainrate(i, 1), h_p_strainrate(i, 2), h_p_strainrate(i, 3), h_p_strainrate(i, 4), h_p_strainrate(i, 5), 
                h_p_spinrate(i, 0), h_p_spinrate(i, 1), h_p_spinrate(i, 2));
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_momentumx(const int i) {
            return h_g_momentum(i, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_momentumx(const int i, const int j, const int k) {
            return h_g_momentum(calc_idx(i, j, k), 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_momentumy(const int i) {
            return h_g_momentum(i, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_momentumy(const int i, const int j, const int k) {
            return h_g_momentum(calc_idx(i, j, k), 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_momentumz(const int i) {
            return h_g_momentum(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_momentumz(const int i, const int j, const int k) {
            return h_g_momentum(calc_idx(i, j, k), 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_forcex(const int i) {
            return h_g_force(i, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_forcex(const int i, const int j, const int k) {
            return h_g_force(calc_idx(i, j, k), 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_forcey(const int i) {
            return h_g_force(i, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_forcey(const int i, const int j, const int k) {
            return h_g_force(calc_idx(i, j, k), 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_forcez(const int i) {
            return h_g_force(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_forcez(const int i, const int j, const int k) {
            return h_g_force(calc_idx(i, j, k), 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_mass(const int i) {
            return h_g_mass(i);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::g_mass(const int i, const int j, const int k) {
            return h_g_mass(calc_idx(i, j, k));
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::pg_nn(const int i) const {
            return h_pg_nn(i);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::pg_nn(const int i, const int j) const {
            return h_pg_nn(i*pg_npp+j);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_w(const int i) const {
            return h_pg_w(i);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_w(const int i, const int j) const {
            return h_pg_w(i*pg_npp+j);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_dwdx(const int i) const {
            return h_pg_dwdx(i, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_dwdx(const int i, const int j) const {
            return h_pg_dwdx(i*pg_npp+j, 0);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_dwdy(const int i) const {
            return h_pg_dwdx(i, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_dwdy(const int i, const int j) const {
            return h_pg_dwdx(i*pg_npp+j, 1);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_dwdz(const int i) const {
            return h_pg_dwdx(i, 2);
        }
        
        template<typename F, typename K, typename SU, typename MB, typename FB>
        F MPM_system<F, K, SU, MB, FB>::pg_dwdz(const int i, const int j) const {
            return h_pg_dwdx(i*pg_npp+j, 2);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        std::array<F, dims>& MPM_system<F, K, SU, MB, FB>::body_force() {
            return m_body_force;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::body_forcex() {
            return m_body_force[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::body_forcey() {
            return m_body_force[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        F& MPM_system<F, K, SU, MB, FB>::body_forcez() {
            return m_body_force[2];
        }

#ifdef GRAMPM_MPI
        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_min_idxx(const int i) const {
            return h_ORB_extents(i).min[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_min_idxy(const int i) const {
            return h_ORB_extents(i).min[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_min_idxz(const int i) const {
            return h_ORB_extents(i).min[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_max_idxx(const int i) const {
            return h_ORB_extents(i).max[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_max_idxy(const int i) const {
            return h_ORB_extents(i).max[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_max_idxz(const int i) const {
            return h_ORB_extents(i).max[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_n_neighbours() const {
            return n_ORB_neighbours;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_neighbour(const int i) const {
            return h_ORB_neighbours(i);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_send_halo_minx(const int neighbour) const {
            return h_ORB_send_halo(neighbour).min[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_send_halo_miny(const int neighbour) const {
            return h_ORB_send_halo(neighbour).min[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_send_halo_minz(const int neighbour) const {
            return h_ORB_send_halo(neighbour).min[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_send_halo_maxx(const int neighbour) const {
            return h_ORB_send_halo(neighbour).max[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_send_halo_maxy(const int neighbour) const {
            return h_ORB_send_halo(neighbour).max[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_send_halo_maxz(const int neighbour) const {
            return h_ORB_send_halo(neighbour).max[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_recv_halo_minx(const int neighbour) const {
            return h_ORB_recv_halo(neighbour).min[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_recv_halo_miny(const int neighbour) const {
            return h_ORB_recv_halo(neighbour).min[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_recv_halo_minz(const int neighbour) const {
            return h_ORB_recv_halo(neighbour).min[2];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_recv_halo_maxx(const int neighbour) const {
            return h_ORB_recv_halo(neighbour).max[0];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_recv_halo_maxy(const int neighbour) const {
            return h_ORB_recv_halo(neighbour).max[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::ORB_recv_halo_maxz(const int neighbour) const {
            return h_ORB_recv_halo(neighbour).max[2];
        }

#endif
    }
}
#endif