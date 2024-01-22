#ifndef GRAMPM_ACCLELERATED_CORE_HELPERS
#define GRAMPM_ACCLELERATED_CORE_HELPERS

#include <array>
#include <string>

namespace GraMPM {
    namespace accelerated {

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        template<typename I>
        std::array<I, dims> MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::unravel_idx(const I &idx) const {
            std::array<I, dims> unravelled_idx;
            // div_t tmp = std::div(idx, m_g_ngridy*m_g_ngridz);
            // unravelled_idx[0] = tmp.quot;
            // tmp = std::div(tmp.rem, m_g_ngridz);
            // unravelled_idx[1] = tmp.quot;
            // unravelled_idx[2] = tmp.rem;
            // return unravelled_idx;
            unravelled_idx[0] = idx / (m_ngrid[1]*m_ngrid[2]);
            I rem = idx % (m_ngrid[1]*m_ngrid[2]);
            unravelled_idx[1] = rem / m_ngrid[2];
            unravelled_idx[2] = rem % m_ngrid[2];
            return unravelled_idx;
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        size_t MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::calc_idx(const size_t i, const size_t j, const size_t k) const {
            return i*m_ngrid[1]*m_ngrid[2] + j*m_ngrid[2] + k;
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::h2d() {
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
            deep_copy(d_p_grid_idx, h_p_grid_idx);
            deep_copy(d_pg_nn, h_pg_nn);
            deep_copy(d_pg_w, h_pg_w);
            deep_copy(d_pg_dwdx, h_pg_dwdx);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::d2h() {
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
            deep_copy(h_p_grid_idx, d_p_grid_idx);
            deep_copy(h_pg_nn, d_pg_nn);
            deep_copy(h_pg_w, d_pg_w);
            deep_copy(h_pg_dwdx, d_pg_dwdx);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::d_zero_grid() {
            Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_momentum, 0.);
            Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_force, 0.);
            Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_mass, 0.);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::h_zero_grid() {
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

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::p_save_to_file(const std::string &prefix, const int &timestep) const {

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

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::g_save_to_file(const std::string &prefix, const int &timestep) const {

            // convert timestep number to string (width of 7 chars, for up to 9,999,999,999 timesteps)
            std::string str_timestep = std::to_string(timestep);
            str_timestep = std::string(7-str_timestep.length(), '0') + str_timestep;

            std::string fname {prefix + str_timestep};

            std::ofstream outfile(fname);

            const int i_width = 7, f_width = 12, f_precision=10;

            outfile << std::setw(i_width) << "id "
                    << std::setw(f_width) << "x "
                    << std::setw(f_width) << "y "
                    << std::setw(f_width) << "z "
                    << std::setw(f_width) << "px "
                    << std::setw(f_width) << "py "
                    << std::setw(f_width) << "pz "
                    << std::setw(f_width) << "mass "
                    << std::setw(f_width) << "fx "
                    << std::setw(f_width) << "fy "
                    << std::setw(f_width) << "fz "
                    << '\n';

            for (size_t i = 0; i < m_ngrid[0]; ++i) {
                for (size_t j = 0; j < m_ngrid[1]; ++j) {
                    for (size_t k = 0; k < m_ngrid[2]; ++k) {
                        const size_t idx = i*m_ngrid[1]*m_ngrid[2] + j*m_ngrid[2] + k;
                        outfile << std::setw(i_width) << idx << ' ' << std::setprecision(f_precision)
                                << std::setw(f_width) << std::fixed << m_mingrid[0]+i*m_g_cell_size << ' '
                                << std::setw(f_width) << std::fixed << m_mingrid[1]+j*m_g_cell_size << ' '
                                << std::setw(f_width) << std::fixed << m_mingrid[2]+k*m_g_cell_size << ' '
                                << std::setw(f_width) << std::fixed << h_g_momentum(idx, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_g_momentum(idx, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_g_momentum(idx, 2) << ' '
                                << std::setw(f_width) << std::fixed << h_g_mass(idx) << ' '
                                << std::setw(f_width) << std::fixed << h_g_force(idx, 0) << ' '
                                << std::setw(f_width) << std::fixed << h_g_force(idx, 1) << ' '
                                << std::setw(f_width) << std::fixed << h_g_force(idx, 2) << ' '
                                << '\n';
                    }
                }
            }
        }
        

    }
}
#endif