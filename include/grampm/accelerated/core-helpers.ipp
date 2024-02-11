#ifndef GRAMPM_ACCLELERATED_CORE_HELPERS
#define GRAMPM_ACCLELERATED_CORE_HELPERS

#include <array>
#include <string>
#include <hdf5.h>

namespace GraMPM {
    namespace accelerated {

        template<typename F, typename K, typename SU, typename MB, typename FB>
        template<typename I>
        std::array<I, dims> MPM_system<F, K, SU, MB, FB>::unravel_idx(const I &idx) const {
            std::array<I, dims> unravelled_idx;
            unravelled_idx[2] = idx % m_ngrid[2];
            I rem = idx / m_ngrid[2];
            unravelled_idx[1] = rem % m_ngrid[1];
            unravelled_idx[0] = rem / m_ngrid[1];
            return unravelled_idx;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        KOKKOS_INLINE_FUNCTION
        void MPM_system<F, K, SU, MB, FB>::unravel_idx(const int idx, int &i, int &j, int &k) const {
            k = idx % m_ngrid[2];
            int rem = idx / m_ngrid[2];
            j = rem % m_ngrid[1];
            i = rem / m_ngrid[1];
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        int MPM_system<F, K, SU, MB, FB>::calc_idx(const int i, const int j, const int k) const {
            return i*m_ngrid[1]*m_ngrid[2] + j*m_ngrid[2] + k;
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::h2d() {
            deep_copy(m_p_x.d_view, m_p_x.h_view);
            deep_copy(m_p_v.d_view, m_p_v.h_view);
            deep_copy(m_p_a.d_view, m_p_a.h_view);
            deep_copy(m_p_dxdt.d_view, m_p_dxdt.h_view);
            deep_copy(m_g_momentum.d_view, m_g_momentum.h_view);
            deep_copy(m_g_force.d_view, m_g_force.h_view);
            deep_copy(m_p_sigma.d_view, m_p_sigma.h_view);
            deep_copy(m_p_strainrate.d_view, m_p_strainrate.h_view);
            deep_copy(m_p_spinrate.d_view, m_p_spinrate.h_view);
            deep_copy(m_p_mass.d_view, m_p_mass.h_view);
            deep_copy(m_p_rho.d_view, m_p_rho.h_view);
            deep_copy(m_g_mass.d_view, m_g_mass.h_view);
            deep_copy(m_p_grid_idx.d_view, m_p_grid_idx.h_view);
            deep_copy(m_pg_nn.d_view, m_pg_nn.h_view);
            deep_copy(m_pg_w.d_view, m_pg_w.h_view);
            deep_copy(m_pg_dwdx.d_view, m_pg_dwdx.h_view);
#ifdef GRAMPM_MPI
            deep_copy(m_ORB_extents.d_view, m_ORB_extents.h_view);
            deep_copy(m_ORB_send_halo.d_view, m_ORB_send_halo.h_view);
            deep_copy(m_ORB_recv_halo.d_view, m_ORB_recv_halo.h_view);
            deep_copy(m_ORB_neighbours.d_view, m_ORB_neighbours.h_view);
#endif
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::d2h() {
            deep_copy(m_p_x.h_view, m_p_x.d_view);
            deep_copy(m_p_v.h_view, m_p_v.d_view);
            deep_copy(m_p_a.h_view, m_p_a.d_view);
            deep_copy(m_p_dxdt.h_view, m_p_dxdt.d_view);
            deep_copy(m_g_momentum.h_view, m_g_momentum.d_view);
            deep_copy(m_g_force.h_view, m_g_force.d_view);
            deep_copy(m_p_sigma.h_view, m_p_sigma.d_view);
            deep_copy(m_p_strainrate.h_view, m_p_strainrate.d_view);
            deep_copy(m_p_spinrate.h_view, m_p_spinrate.d_view);
            deep_copy(m_p_mass.h_view, m_p_mass.d_view);
            deep_copy(m_p_rho.h_view, m_p_rho.d_view);
            deep_copy(m_g_mass.h_view, m_g_mass.d_view);
            deep_copy(m_p_grid_idx.h_view, m_p_grid_idx.d_view);
            deep_copy(m_pg_nn.h_view, m_pg_nn.d_view);
            deep_copy(m_pg_w.h_view, m_pg_w.d_view);
            deep_copy(m_pg_dwdx.h_view, m_pg_dwdx.d_view);
#ifdef GRAMPM_MPI
            deep_copy(m_ORB_extents.h_view, m_ORB_extents.d_view);
            deep_copy(m_ORB_send_halo.h_view, m_ORB_send_halo.d_view);
            deep_copy(m_ORB_recv_halo.h_view, m_ORB_recv_halo.d_view);
            deep_copy(m_ORB_neighbours.h_view, m_ORB_neighbours.d_view);
#endif
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::d_zero_grid() {
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_g_momentum.d_view, F(0.));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_g_force.d_view, F(0.));
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_g_mass.d_view, F(0.));
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::h_zero_grid() {
            Kokkos::deep_copy(m_g_momentum.h_view, F(0.));
            Kokkos::deep_copy(m_g_force.h_view, F(0.));
            Kokkos::deep_copy(m_g_mass.h_view, F(0.));
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

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::save_to_h5(const std::string &prefix, const int &timestep) const {

            // convert timestep number to string (width of 7 chars, for up to 9,999,999,999 timesteps)
            std::string str_timestep = std::to_string(timestep);
            str_timestep = std::string(7-str_timestep.length(), '0') + str_timestep;

            std::string fname {prefix + str_timestep};

            herr_t status;

            hsize_t dims_vec[2] {m_p_x.extent(1), m_p_x.extent(0)},
                dims_tens[2] {m_p_sigma.extent(1), m_p_sigma.extent(0)},
                dims_scalar[1] {m_p_rho.extent(0)},
                dims_spintens[2] {m_p_spinrate.extent(1), m_p_spinrate.extent(0)};
            
            hid_t file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hid_t group_id = H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = write2h5(2, dims_vec, m_p_x.h_view.data(), group_id, "x");
            status = write2h5(2, dims_vec, m_p_v.h_view.data(), group_id, "v");
            status = write2h5(2, dims_vec, m_p_a.h_view.data(), group_id, "a");
            status = write2h5(2, dims_vec, m_p_dxdt.h_view.data(), group_id, "dxdt");
            status = write2h5(2, dims_tens, m_p_sigma.h_view.data(), group_id, "sigma");
            status = write2h5(2, dims_tens, m_p_strainrate.h_view.data(), group_id, "strainrate");
            status = write2h5(2, dims_spintens, m_p_spinrate.h_view.data(), group_id, "spinrate");
            status = write2h5(1, dims_scalar, m_p_mass.h_view.data(), group_id, "mass");
            status = write2h5(1, dims_scalar, m_p_rho.h_view.data(), group_id, "rho");
            status = H5Gclose(group_id);

            hsize_t dims_grid_vec[4] {3, static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])},
                dims_grid_scalar[3] {static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])};
            group_id = H5Gcreate(file_id, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = write_grid_extents(group_id, m_g_extents.min, m_g_extents.max, m_g_cell_size);
            status = write2h5(4, dims_grid_vec, m_g_momentum.h_view.data(), group_id, "momentum");
            status = write2h5(4, dims_grid_vec, m_g_force.h_view.data(), group_id, "force");
            status = write2h5(3, dims_grid_scalar, m_g_mass.h_view.data(), group_id, "mass");
            status = H5Gclose(group_id);
            status = H5Fclose(file_id);
        }

        template<typename F, typename K, typename SU, typename MB, typename FB>
        void MPM_system<F, K, SU, MB, FB>::save_to_h5_async(const std::string &prefix, const int &timestep) const {

            // initiate transfer of first batch of data
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_x.h_view, m_p_x.d_view);

            // do stuff
            // convert timestep number to string (width of 7 chars, for up to 9,999,999,999 timesteps)
            std::string str_timestep = std::to_string(timestep);
            str_timestep = std::string(7-str_timestep.length(), '0') + str_timestep;

            std::string fname {prefix + str_timestep};

            herr_t status;

            hsize_t dims_vec[2] {m_p_x.extent(1), m_p_x.extent(0)},
                dims_tens[2] {m_p_sigma.extent(1), m_p_sigma.extent(0)},
                dims_scalar[1] {m_p_rho.extent(0)},
                dims_spintens[2] {m_p_spinrate.extent(1), m_p_spinrate.extent(0)};
            
            hid_t file_id = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            hid_t group_id = H5Gcreate(file_id, "/particles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            
            // wait for first batch of data
            Kokkos::fence();
            // initiate next batch
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_v.h_view, m_p_v.d_view);
            status = write2h5(2, dims_vec, m_p_x.h_view.data(), group_id, "x");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_a.h_view, m_p_a.d_view);
            status = write2h5(2, dims_vec, m_p_v.h_view.data(), group_id, "v");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_dxdt.h_view, m_p_dxdt.d_view);
            status = write2h5(2, dims_vec, m_p_a.h_view.data(), group_id, "a");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_sigma.h_view, m_p_sigma.d_view);
            status = write2h5(2, dims_vec, m_p_dxdt.h_view.data(), group_id, "dxdt");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_strainrate.h_view, m_p_strainrate.d_view);
            status = write2h5(2, dims_tens, m_p_sigma.h_view.data(), group_id, "sigma");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_spinrate.h_view, m_p_spinrate.d_view);
            status = write2h5(2, dims_tens, m_p_strainrate.h_view.data(), group_id, "strainrate");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_mass.h_view, m_p_mass.d_view);
            status = write2h5(2, dims_spintens, m_p_spinrate.h_view.data(), group_id, "spinrate");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_p_rho.h_view, m_p_rho.d_view);
            status = write2h5(1, dims_scalar, m_p_mass.h_view.data(), group_id, "mass");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_g_momentum.h_view, m_g_momentum.d_view);
            status = write2h5(1, dims_scalar, m_p_rho.h_view.data(), group_id, "rho");
            status = H5Gclose(group_id);

            hsize_t dims_grid_vec[4] {3, static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])},
                dims_grid_scalar[3] {static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])},
                dims_grid_tens[4] {6, static_cast<hsize_t>(m_ngrid[2]), static_cast<hsize_t>(m_ngrid[1]), static_cast<hsize_t>(m_ngrid[0])};
            group_id = H5Gcreate(file_id, "/grid", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            status = write_grid_extents(group_id, m_g_extents.min, m_g_extents.max, m_g_cell_size);
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_g_force.h_view, m_g_force.d_view);
            status = write2h5(4, dims_grid_vec, m_g_momentum.h_view.data(), group_id, "momentum");
            Kokkos::fence();
            Kokkos::deep_copy(Kokkos::DefaultExecutionSpace(), m_g_mass.h_view, m_g_mass.d_view);
            status = write2h5(4, dims_grid_vec, m_g_force.h_view.data(), group_id, "force");
            Kokkos::fence();
            status = write2h5(3, dims_grid_scalar, m_g_mass.h_view.data(), group_id, "mass");
            status = H5Gclose(group_id);
            status = H5Fclose(file_id);
        }

    }
}
#endif
