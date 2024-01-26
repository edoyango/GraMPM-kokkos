#ifndef GRAMPM_ACCELERATED_CORE_CONSTRUCTORS
#define GRAMPM_ACCELERATED_CORE_CONSTRUCTORS

#include <string>
#include <hdf5.h>

static int h5_get_nparticles(std::string fname) {
    hid_t f_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t dset_id = H5Dopen(f_id, "/particles/x", H5P_DEFAULT);
    hid_t dspace_id = H5Dget_space(dset_id);
    int ndims = H5Sget_simple_extent_ndims(dspace_id);
    hsize_t dims[ndims];
    H5Sget_simple_extent_dims(dspace_id, dims, NULL);
    H5Sclose(dspace_id);
    H5Dclose(dset_id);
    H5Fclose(f_id);
    return dims[1];
}

static double h5_get_cellsize(std::string fname) {
    hid_t f_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t g_id = H5Gopen(f_id, "/grid", H5P_DEFAULT);
    hid_t attr_id = H5Aopen(g_id, "cell_size", H5P_DEFAULT);
    double cell_size;
    H5Aread(attr_id, H5T_NATIVE_DOUBLE, &cell_size);
    H5Aclose(attr_id);
    H5Gclose(g_id);
    H5Fclose(f_id);
    return cell_size;
}

static std::array<double, 3> h5_get_mingrid(std::string fname) {
    hid_t f_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t g_id = H5Gopen(f_id, "/grid", H5P_DEFAULT);
    hid_t attr_id = H5Aopen(g_id, "mingrid", H5P_DEFAULT);
    std::array<double, 3> mingrid;
    H5Aread(attr_id, H5T_NATIVE_DOUBLE, mingrid.data());
    H5Aclose(attr_id);
    H5Gclose(g_id);
    H5Fclose(f_id);
    return mingrid;
}

static std::array<double, 3> h5_get_maxgrid(std::string fname) {
    hid_t f_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t g_id = H5Gopen(f_id, "/grid", H5P_DEFAULT);
    hid_t attr_id = H5Aopen(g_id, "maxgrid", H5P_DEFAULT);
    std::array<double, 3> maxgrid;
    H5Aread(attr_id, H5T_NATIVE_DOUBLE, maxgrid.data());
    H5Aclose(attr_id);
    H5Gclose(g_id);
    H5Fclose(f_id);
    return maxgrid;
}

namespace GraMPM {
    namespace accelerated {

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(std::vector<particle<F>> &pv, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
            : knl(dcell)
            , m_p_size {static_cast<int>(pv.size())}
            , m_g_cell_size {dcell}
            , m_mingrid {mingrid}
            , m_maxgrid {maxgrid}
            , m_ngrid {
                static_cast<int>(std::ceil((maxgrid[0]-mingrid[0])/dcell))+1,
                static_cast<int>(std::ceil((maxgrid[1]-mingrid[1])/dcell))+1,
                static_cast<int>(std::ceil((maxgrid[2]-mingrid[2])/dcell))+1
            }
            , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
            , d_p_x("Particles' 3D positions", m_p_size)
            , d_p_v("Particles' 3D velocity", m_p_size)
            , d_p_a("Particles' 3D accelerations", m_p_size)
            , d_p_dxdt("Particles' 3D dxdt", m_p_size)
            , d_g_momentum("Grid cells' 3D momentum", m_g_size)
            , d_g_force("Grid cells' 3D force", m_g_size)
            , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
            , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
            , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
            , d_p_mass("Particles' mass", m_p_size)
            , d_p_rho("Particles' mass", m_p_size)
            , d_g_mass("Grid cells' mass", m_g_size)
            , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
            , pg_npp {static_cast<int>(8*knl.radius*knl.radius*knl.radius)}
            , d_pg_nn("Particles' grid neighbour nodes", m_p_size*pg_npp)
            , d_pg_w("Particles' grid neighbour nodes' kernel values", m_p_size*pg_npp)
            , d_pg_dwdx("Particles' grid neighbour nodes' kernel gradient values", m_p_size*pg_npp)
            , h_p_x {create_mirror_view(d_p_x)}
            , h_p_v {create_mirror_view(d_p_v)}
            , h_p_a {create_mirror_view(d_p_a)}
            , h_p_dxdt{create_mirror_view(d_p_dxdt)}
            , h_g_momentum{create_mirror_view(d_g_momentum)}
            , h_g_force{create_mirror_view(d_g_force)}
            , h_p_sigma{create_mirror_view(d_p_sigma)}
            , h_p_strainrate{create_mirror_view(d_p_strainrate)}
            , h_p_spinrate{create_mirror_view(d_p_spinrate)}
            , h_p_mass{create_mirror_view(d_p_mass)}
            , h_p_rho{create_mirror_view(d_p_rho)}
            , h_g_mass{create_mirror_view(d_g_mass)}
            , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
            , h_pg_nn {create_mirror_view(d_pg_nn)}
            , h_pg_w {create_mirror_view(d_pg_w)}
            , h_pg_dwdx {create_mirror_view(d_pg_dwdx)}
            , f_momentum_boundary(d_g_momentum, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_force_boundary(d_g_force, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_map_gidx(dcell, mingrid[0], mingrid[1], mingrid[2], m_ngrid[0], m_ngrid[1], 
                m_ngrid[2], d_p_x, d_p_grid_idx)
            , f_find_neighbour_nodes(dcell, mingrid[0], mingrid[1], mingrid[2], m_ngrid[0], m_ngrid[1],
                m_ngrid[2], static_cast<int>(knl.radius), d_p_x, d_p_grid_idx, d_pg_nn, d_pg_w, d_pg_dwdx, knl)
            , f_map_p2g_mass(pg_npp, d_p_mass, d_g_mass, d_pg_nn, d_pg_w)
            , f_map_p2g_momentum(pg_npp, d_p_mass, d_p_v, d_g_momentum, d_pg_nn, d_pg_w)
            , f_map_p2g_force(pg_npp, d_p_mass, d_p_rho, d_p_sigma, d_g_force, d_pg_nn, d_pg_w, d_pg_dwdx, 
                m_body_force[0], m_body_force[1], m_body_force[2])
            , f_map_g2p_acceleration(pg_npp, d_p_a, d_g_force, d_p_dxdt, d_g_momentum, d_g_mass, d_pg_w, 
                d_pg_nn)
            , f_map_g2p_strainrate(pg_npp, d_p_strainrate, d_p_spinrate, d_g_momentum, d_pg_dwdx, d_g_mass, 
                d_pg_nn)
            , f_g_update_momentum(d_g_momentum, d_g_force)
            , f_p_update_velocity(d_p_v, d_p_a)
            , f_p_update_position(d_p_x, d_p_dxdt)
            , f_p_update_density(d_p_rho, d_p_strainrate)
            , f_stress_update(d_p_sigma, d_p_strainrate, d_p_spinrate)
        {
            for (int i = 0; i < m_p_size; ++i) {
                for (int d = 0; d < dims; ++d) {
                    h_p_x(i, d) = pv[i].x[d];
                    h_p_v(i, d) = pv[i].v[d];
                    h_p_a(i, d) = pv[i].a[d];
                    h_p_dxdt(i, d) = pv[i].dxdt[d];
                }
                for (int d = 0; d < voigt_tens_elems; ++d) {
                    h_p_sigma(i, d) = pv[i].sigma[d];
                    h_p_strainrate(i, d) = pv[i].strainrate[d];
                }
                for (int d = 0; d < spin_tens_elems; ++d) {
                    h_p_spinrate(i, d) = pv[i].spinrate[d];
                }
                h_p_mass(i) = pv[i].mass;
                h_p_rho(i) = pv[i].rho;
            }
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(const int n, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
            : knl(dcell)
            , m_p_size {n}
            , m_g_cell_size {dcell}
            , m_mingrid {mingrid}
            , m_maxgrid {maxgrid}
            , m_ngrid{
                static_cast<int>(std::ceil((maxgrid[0]-mingrid[0])/dcell))+1,
                static_cast<int>(std::ceil((maxgrid[1]-mingrid[1])/dcell))+1,
                static_cast<int>(std::ceil((maxgrid[2]-mingrid[2])/dcell))+1
            }
            , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
            , d_p_x("Particles' 3D positions", m_p_size)
            , d_p_v("Particles' 3D velocity", m_p_size)
            , d_p_a("Particles' 3D accelerations", m_p_size)
            , d_p_dxdt("Particles' 3D dxdt", m_p_size)
            , d_g_momentum("Grid cells' 3D momentum", m_g_size)
            , d_g_force("Grid cells' 3D force", m_g_size)
            , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
            , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
            , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
            , d_p_mass("Particles' mass", m_p_size)
            , d_p_rho("Particles' mass", m_p_size)
            , d_g_mass("Grid cells' mass", m_g_size)
            , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
            , pg_npp {static_cast<int>(8*knl.radius*knl.radius*knl.radius)}
            , d_pg_nn("Particles' grid neighbour nodes", m_p_size*pg_npp)
            , d_pg_w("Particles' grid neighbour nodes' kernel values", m_p_size*pg_npp)
            , d_pg_dwdx("Particles' grid neighbour nodes' kernel gradient values", m_p_size*pg_npp)
            , h_p_x {create_mirror_view(d_p_x)}
            , h_p_v {create_mirror_view(d_p_v)}
            , h_p_a {create_mirror_view(d_p_a)}
            , h_p_dxdt{create_mirror_view(d_p_dxdt)}
            , h_g_momentum{create_mirror_view(d_g_momentum)}
            , h_g_force{create_mirror_view(d_g_force)}
            , h_p_sigma{create_mirror_view(d_p_sigma)}
            , h_p_strainrate{create_mirror_view(d_p_strainrate)}
            , h_p_spinrate{create_mirror_view(d_p_spinrate)}
            , h_p_mass{create_mirror_view(d_p_mass)}
            , h_p_rho{create_mirror_view(d_p_rho)}
            , h_g_mass{create_mirror_view(d_g_mass)}
            , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
            , h_pg_nn {create_mirror_view(d_pg_nn)}
            , h_pg_w {create_mirror_view(d_pg_w)}
            , h_pg_dwdx {create_mirror_view(d_pg_dwdx)}
            , f_momentum_boundary(d_g_momentum, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_force_boundary(d_g_force, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_map_gidx(dcell, mingrid[0], mingrid[1], mingrid[2], m_ngrid[0], m_ngrid[1], 
                m_ngrid[2], d_p_x, d_p_grid_idx)
            , f_find_neighbour_nodes(dcell, mingrid[0], mingrid[1], mingrid[2], m_ngrid[0], m_ngrid[1],
                m_ngrid[2], static_cast<int>(knl.radius), d_p_x, d_p_grid_idx, d_pg_nn, d_pg_w, d_pg_dwdx, knl)
            , f_map_p2g_mass(pg_npp, d_p_mass, d_g_mass, d_pg_nn, d_pg_w)
            , f_map_p2g_momentum(pg_npp, d_p_mass, d_p_v, d_g_momentum, d_pg_nn, d_pg_w)
            , f_map_p2g_force(pg_npp, d_p_mass, d_p_rho, d_p_sigma, d_g_force, d_pg_nn, d_pg_w, d_pg_dwdx, 
                m_body_force[0], m_body_force[1], m_body_force[2])
            , f_map_g2p_acceleration(pg_npp, d_p_a, d_g_force, d_p_dxdt, d_g_momentum, d_g_mass, d_pg_w, 
                d_pg_nn)
            , f_map_g2p_strainrate(pg_npp, d_p_strainrate, d_p_spinrate, d_g_momentum, d_pg_dwdx, d_g_mass, 
                d_pg_nn)
            , f_g_update_momentum(d_g_momentum, d_g_force)
            , f_p_update_velocity(d_p_v, d_p_a)
            , f_p_update_position(d_p_x, d_p_dxdt)
            , f_p_update_density(d_p_rho, d_p_strainrate)
            , f_stress_update(d_p_sigma, d_p_strainrate, d_p_spinrate)
        {
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(std::string fname)
            : knl(h5_get_cellsize(fname))
            , m_p_size {h5_get_nparticles(fname)}
            , m_g_cell_size {h5_get_cellsize(fname)}
            , m_mingrid {h5_get_mingrid(fname)}
            , m_maxgrid {h5_get_maxgrid(fname)}
            , m_ngrid{
                static_cast<int>(std::ceil((m_maxgrid[0]-m_mingrid[0])/m_g_cell_size))+1,
                static_cast<int>(std::ceil((m_maxgrid[1]-m_mingrid[1])/m_g_cell_size))+1,
                static_cast<int>(std::ceil((m_maxgrid[2]-m_mingrid[2])/m_g_cell_size))+1
            }
            , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
            , d_p_x("Particles' 3D positions", m_p_size)
            , d_p_v("Particles' 3D velocity", m_p_size)
            , d_p_a("Particles' 3D accelerations", m_p_size)
            , d_p_dxdt("Particles' 3D dxdt", m_p_size)
            , d_g_momentum("Grid cells' 3D momentum", m_g_size)
            , d_g_force("Grid cells' 3D force", m_g_size)
            , d_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
            , d_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
            , d_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
            , d_p_mass("Particles' mass", m_p_size)
            , d_p_rho("Particles' mass", m_p_size)
            , d_g_mass("Grid cells' mass", m_g_size)
            , d_p_grid_idx("Particles' hashed grid indices", m_p_size)
            , pg_npp {static_cast<int>(8*knl.radius*knl.radius*knl.radius)}
            , d_pg_nn("Particles' grid neighbour nodes", m_p_size*pg_npp)
            , d_pg_w("Particles' grid neighbour nodes' kernel values", m_p_size*pg_npp)
            , d_pg_dwdx("Particles' grid neighbour nodes' kernel gradient values", m_p_size*pg_npp)
            , h_p_x {create_mirror_view(d_p_x)}
            , h_p_v {create_mirror_view(d_p_v)}
            , h_p_a {create_mirror_view(d_p_a)}
            , h_p_dxdt{create_mirror_view(d_p_dxdt)}
            , h_g_momentum{create_mirror_view(d_g_momentum)}
            , h_g_force{create_mirror_view(d_g_force)}
            , h_p_sigma{create_mirror_view(d_p_sigma)}
            , h_p_strainrate{create_mirror_view(d_p_strainrate)}
            , h_p_spinrate{create_mirror_view(d_p_spinrate)}
            , h_p_mass{create_mirror_view(d_p_mass)}
            , h_p_rho{create_mirror_view(d_p_rho)}
            , h_g_mass{create_mirror_view(d_g_mass)}
            , h_p_grid_idx{create_mirror_view(d_p_grid_idx)}
            , h_pg_nn {create_mirror_view(d_pg_nn)}
            , h_pg_w {create_mirror_view(d_pg_w)}
            , h_pg_dwdx {create_mirror_view(d_pg_dwdx)}
            , f_momentum_boundary(d_g_momentum, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_force_boundary(d_g_force, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_map_gidx(m_g_cell_size, m_mingrid[0], m_mingrid[1], m_mingrid[2], m_ngrid[0], m_ngrid[1], 
                m_ngrid[2], d_p_x, d_p_grid_idx)
            , f_find_neighbour_nodes(m_g_cell_size, m_mingrid[0], m_mingrid[1], m_mingrid[2], m_ngrid[0], m_ngrid[1],
                m_ngrid[2], static_cast<int>(knl.radius), d_p_x, d_p_grid_idx, d_pg_nn, d_pg_w, d_pg_dwdx, knl)
            , f_map_p2g_mass(pg_npp, d_p_mass, d_g_mass, d_pg_nn, d_pg_w)
            , f_map_p2g_momentum(pg_npp, d_p_mass, d_p_v, d_g_momentum, d_pg_nn, d_pg_w)
            , f_map_p2g_force(pg_npp, d_p_mass, d_p_rho, d_p_sigma, d_g_force, d_pg_nn, d_pg_w, d_pg_dwdx, 
                m_body_force[0], m_body_force[1], m_body_force[2])
            , f_map_g2p_acceleration(pg_npp, d_p_a, d_g_force, d_p_dxdt, d_g_momentum, d_g_mass, d_pg_w, 
                d_pg_nn)
            , f_map_g2p_strainrate(pg_npp, d_p_strainrate, d_p_spinrate, d_g_momentum, d_pg_dwdx, d_g_mass, 
                d_pg_nn)
            , f_g_update_momentum(d_g_momentum, d_g_force)
            , f_p_update_velocity(d_p_v, d_p_a)
            , f_p_update_position(d_p_x, d_p_dxdt)
            , f_p_update_density(d_p_rho, d_p_strainrate)
            , f_stress_update(d_p_sigma, d_p_strainrate, d_p_spinrate)
        {
            std::ifstream file(fname);
            std::string line, header;
            // pull out header
            std::getline(file, header);
            int i = 0;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int idx;
                iss >> idx 
                    >> h_p_x(i, 0) >> h_p_x(i, 1) >> h_p_x(i, 2) 
                    >> h_p_v(i, 0) >> h_p_v(i, 1) >> h_p_v(i, 2) 
                    >> h_p_mass(i) 
                    >> h_p_rho(i) 
                    >> h_p_sigma(i, 0) >> h_p_sigma(i, 1) >> h_p_sigma(i, 2)>> h_p_sigma(i, 3)>> h_p_sigma(i, 4)>> h_p_sigma(i, 5)
                    >> h_p_a(i, 0) >> h_p_a(i, 1) >> h_p_a(i, 2)
                    >> h_p_dxdt(i, 0) >> h_p_dxdt(i, 1) >> h_p_dxdt(i, 2)
                    >> h_p_strainrate(i, 0) >> h_p_strainrate(i, 1) >> h_p_strainrate(i, 2) >> h_p_strainrate(i, 3) >> h_p_strainrate(i, 4) >> h_p_strainrate(i, 5)
                    >> h_p_spinrate(i, 0) >> h_p_spinrate(i, 1) >> h_p_spinrate(i, 2);
                i++;
            }
        }
    }
}

#endif