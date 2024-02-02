#ifndef GRAMPM_ACCLELERATED_CORE_HELPERS
#define GRAMPM_ACCLELERATED_CORE_HELPERS

#include <array>
#include <string>
#include <hdf5.h>

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
        int MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::calc_idx(const int i, const int j, const int k) const {
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
            Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_momentum, F(0.));
            Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_force, F(0.));
            Kokkos::Experimental::fill(Kokkos::DefaultExecutionSpace(), d_g_mass, F(0.));
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::h_zero_grid() {
            for (size_t i = 0; i < m_g_size; ++i) {
                h_g_momentum(i, 0) = F(0.);
                h_g_momentum(i, 1) = F(0.);
                h_g_momentum(i, 2) = F(0.);
                h_g_force(i, 0) = F(0.);
                h_g_force(i, 1) = F(0.);
                h_g_force(i, 2) = F(0.);
                h_g_mass(i) = F(0.);
            }
        }

        static herr_t write2h5(const int r, hsize_t* dims, const double* data, const hid_t gid, const char* dset_name) {
            herr_t status;
            // hsize_t dims[2] {v.extent(0), v.extent(1)};
            hid_t dspace_id = H5Screate_simple(r, dims, NULL);
            hid_t dset_id = H5Dcreate(gid, dset_name, H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            status = H5Dclose(dset_id);
            status = H5Sclose(dspace_id);
            return status;
        }

        static herr_t write2h5(const int r, hsize_t* dims, const float* data, const hid_t gid, const char* dset_name) {
            herr_t status;
            // hsize_t dims[2] {v.extent(0), v.extent(1)};
            hid_t dspace_id = H5Screate_simple(r, dims, NULL);
            hid_t dset_id = H5Dcreate(gid, dset_name, H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            status = H5Dclose(dset_id);
            status = H5Sclose(dspace_id);
            return status;
        }

        static herr_t write2h5(const int r, hsize_t* dims, const int* data, const hid_t gid, const char* dset_name) {
            herr_t status;
            // hsize_t dims[2] {v.extent(0), v.extent(1)};
            hid_t dspace_id = H5Screate_simple(r, dims, NULL);
            hid_t dset_id = H5Dcreate(gid, dset_name, H5T_NATIVE_INT, dspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
            status = H5Dclose(dset_id);
            status = H5Sclose(dspace_id);
            return status;
        }

        static herr_t write_grid_extents(const hid_t gid, const double mins[3], const double maxs[3], 
            const double dcell) {
            herr_t status;
            hsize_t dims[1] {3};
            hid_t dspace_id = H5Screate_simple(1, dims, NULL);
            hid_t attr_id = H5Acreate2(gid, "mingrid", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, mins);
            status = H5Aclose(attr_id);
            attr_id = H5Acreate2(gid, "maxgrid", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, maxs);
            status = H5Aclose(attr_id);
            status = H5Sclose(dspace_id);
            dspace_id = H5Screate(H5S_SCALAR);
            attr_id = H5Acreate2(gid, "cell_size", H5T_NATIVE_DOUBLE, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &dcell);
            status = H5Aclose(attr_id);
            status = H5Sclose(dspace_id);
            return status;
        }

        static herr_t write_grid_extents(const hid_t gid, const float mins[3], const float maxs[3], const float dcell) {
            herr_t status;
            hsize_t dims[1] {3};
            hid_t dspace_id = H5Screate_simple(1, dims, NULL);
            hid_t attr_id = H5Acreate2(gid, "mingrid", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite(attr_id, H5T_NATIVE_FLOAT, mins);
            status = H5Aclose(attr_id);
            attr_id = H5Acreate2(gid, "maxgrid", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite(attr_id, H5T_NATIVE_FLOAT, maxs);
            status = H5Aclose(attr_id);
            status = H5Sclose(dspace_id);
            dspace_id = H5Screate(H5S_SCALAR);
            attr_id = H5Acreate2(gid, "cell_size", H5T_NATIVE_FLOAT, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
            status = H5Awrite(attr_id, H5T_NATIVE_FLOAT, &dcell);
            status = H5Aclose(attr_id);
            status = H5Sclose(dspace_id);
            return status;
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::save_to_h5(const std::string &prefix, const int &timestep) const {

            // convert timestep number to string (width of 7 chars, for up to 9,999,999,999 timesteps)
            std::string str_timestep = std::to_string(timestep);
            str_timestep = std::string(7-str_timestep.length(), '0') + str_timestep;

            std::string fname {prefix + str_timestep};

            herr_t status;

            hsize_t dims_vec[2] {h_p_x.extent(1), h_p_x.extent(0)},
                dims_tens[2] {h_p_sigma.extent(1), h_p_sigma.extent(0)},
                dims_scalar[1] {h_p_rho.extent(0)},
                dims_spintens[2] {h_p_spinrate.extent(1), h_p_spinrate.extent(0)};
            
            hid_t file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hid_t group_id = H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = write2h5(2, dims_vec, h_p_x.data(), group_id, "x");
            status = write2h5(2, dims_vec, h_p_v.data(), group_id, "v");
            status = write2h5(2, dims_vec, h_p_a.data(), group_id, "a");
            status = write2h5(2, dims_vec, h_p_dxdt.data(), group_id, "dxdt");
            status = write2h5(2, dims_tens, h_p_sigma.data(), group_id, "sigma");
            status = write2h5(2, dims_tens, h_p_strainrate.data(), group_id, "strainrate");
            status = write2h5(2, dims_spintens, h_p_spinrate.data(), group_id, "spinrate");
            status = write2h5(1, dims_scalar, h_p_mass.data(), group_id, "mass");
            status = write2h5(1, dims_scalar, h_p_rho.data(), group_id, "rho");
            status = H5Gclose(group_id);

            hsize_t dims_grid_vec[4] {3, static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])},
                dims_grid_scalar[3] {static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])};
            group_id = H5Gcreate(file_id, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = write_grid_extents(group_id, m_g_extents.min, m_g_extents.max, m_g_cell_size);
            status = write2h5(4, dims_grid_vec, h_g_momentum.data(), group_id, "momentum");
            status = write2h5(4, dims_grid_vec, h_g_force.data(), group_id, "force");
            status = write2h5(3, dims_grid_scalar, h_g_mass.data(), group_id, "mass");
            status = H5Gclose(group_id);
            status = H5Fclose(file_id);
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        void MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::save_to_h5_async(const std::string &prefix, const int &timestep) const {

            // initiate transfer of first batch of data
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_x, d_p_x);

            // do stuff
            // convert timestep number to string (width of 7 chars, for up to 9,999,999,999 timesteps)
            std::string str_timestep = std::to_string(timestep);
            str_timestep = std::string(7-str_timestep.length(), '0') + str_timestep;

            std::string fname {prefix + str_timestep};

            herr_t status;

            hsize_t dims_vec[2] {h_p_x.extent(1), h_p_x.extent(0)},
                dims_tens[2] {h_p_sigma.extent(1), h_p_sigma.extent(0)},
                dims_scalar[1] {h_p_rho.extent(0)},
                dims_spintens[2] {h_p_spinrate.extent(1), h_p_spinrate.extent(0)};
            
            hid_t file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hid_t group_id = H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            // wait for first batch of data
            Kokkos::fence();
            // initiate next batch
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_v, d_p_v);
            status = write2h5(2, dims_vec, h_p_x.data(), group_id, "x");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_a, d_p_a);
            status = write2h5(2, dims_vec, h_p_v.data(), group_id, "v");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_dxdt, d_p_dxdt);
            status = write2h5(2, dims_vec, h_p_a.data(), group_id, "a");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_sigma, d_p_sigma);
            status = write2h5(2, dims_vec, h_p_dxdt.data(), group_id, "dxdt");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_strainrate, d_p_strainrate);
            status = write2h5(2, dims_tens, h_p_sigma.data(), group_id, "sigma");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_spinrate, d_p_spinrate);
            status = write2h5(2, dims_tens, h_p_strainrate.data(), group_id, "strainrate");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_mass, d_p_mass);
            status = write2h5(2, dims_spintens, h_p_spinrate.data(), group_id, "spinrate");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_p_rho, d_p_rho);
            status = write2h5(1, dims_scalar, h_p_mass.data(), group_id, "mass");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_g_momentum, d_g_momentum);
            status = write2h5(1, dims_scalar, h_p_rho.data(), group_id, "rho");
            status = H5Gclose(group_id);

            hsize_t dims_grid_vec[4] {3, static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])},
                dims_grid_scalar[3] {static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])},
                dims_grid_tens[4] {6, static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])};
            group_id = H5Gcreate(file_id, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = write_grid_extents(group_id, m_g_extents.min, m_g_extents.max, m_g_cell_size);
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_g_force, d_g_force);
            status = write2h5(4, dims_grid_vec, h_g_momentum.data(), group_id, "momentum");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_g_mass, d_g_mass);
            status = write2h5(4, dims_grid_vec, h_g_force.data(), group_id, "force");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), h_g_sigma, d_g_sigma);
            status = write2h5(3, dims_grid_scalar, h_g_mass.data(), group_id, "mass");
            Kokkos::fence();
            status = write2h5(4, dims_grid_tens, h_g_sigma.data(), group_id, "sigma");
            status = H5Gclose(group_id);
            status = H5Fclose(file_id);
        }

    }
}
#endif
