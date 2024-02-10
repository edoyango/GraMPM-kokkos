#ifndef GRAMPM_ACCELERATED_CORE_CONSTRUCTORS
#define GRAMPM_ACCELERATED_CORE_CONSTRUCTORS

#include <string>
#include <hdf5.h>
#ifdef GRAMPM_MPI
#include <mpi.h>
#endif

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

static herr_t read_h5(const int r, hsize_t* dims, double* data, const hid_t gid, const char* dset_name) {
    herr_t status;
    hid_t dset_id = H5Dopen(gid, dset_name, H5P_DEFAULT);
    status = H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
    status = H5Dclose(dset_id);
    return status;
}

#ifdef GRAMPM_MPI
static int get_MPI_Comm_rank() {
    int procid;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    return procid;
}

static int get_MPI_Comm_size() {
    int numprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    return numprocs;
}
#else

static int get_MPI_Comm_rank() {return 0;}
static int get_MPI_Comm_size() {return 1;}

#endif

namespace GraMPM {

    template<typename F>
    particle<F>::particle(const F inx, const F iny, const F inz, const F inmass, const F inrho, const F invx, 
        const F invy, const F invz, const F insigmaxx, const F insigmayy, const F insigmazz, const F insigmaxy,
        const F insigmaxz, const F insigmayz, const F inax, const F inay, const F inaz, const F indxdt, 
        const F indydt, const F indzdt, const F instrainratexx, const F instrainrateyy, const F instrainratezz, 
        const F instrainratexy, const F instrainratexz, const F instrainrateyz, const F inspinratexy,
        const F inspinratexz, const F inspinrateyz)
        : x {inx, iny, inz}
        , v {invx, invy, invz}
        , a {inax, inay, inaz}
        , dxdt {indxdt, indydt, indzdt}
        , sigma {insigmaxx, insigmayy, insigmazz, insigmaxy, insigmaxz, insigmayz}
        , strainrate {instrainratexx, instrainrateyy, instrainratezz, instrainratexy, instrainratexz, instrainrateyz}
        , spinrate {inspinratexy, inspinratexz, inspinrateyz}
        , mass {inmass}
        , rho {inrho}
    {
    }

    template<typename F>
    particle<F>::particle(const std::array<F, 3> inx, const std::array<F, 3> inv, const F inmass, const F inrho, 
        const std::array<F, 6> insigma, const std::array<F, 3> ina, const std::array<F, 3> indxdt, 
        const std::array<F, 6> instrainrate, const std::array<F, 3> inspinrate)
        : particle<F>(inx[0], inx[1], inx[2], inmass, inrho, inv[0], inv[1], inv[2], insigma[0], insigma[1], insigma[2],
            insigma[3], insigma[4], insigma[5], ina[0], ina[1], ina[2], indxdt[0], indxdt[1], indxdt[2],
            instrainrate[0], instrainrate[1], instrainrate[2], instrainrate[3], instrainrate[4], instrainrate[5],
            inspinrate[0], inspinrate[1], inspinrate[2])
    {
    }

    template<typename F> particle<F>::particle()
        : particle(0., 0., 0., 0., 0.)
    {
    }

    namespace accelerated {

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(const int n, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
            : knl(dcell)
            , m_p_size {n}
            , m_g_cell_size {dcell}
            , m_g_extents{mingrid[0], mingrid[1], mingrid[2], maxgrid[0], maxgrid[1], maxgrid[2]}
            , m_ngrid{
                static_cast<int>(std::ceil((maxgrid[0]-mingrid[0])/dcell))+1,
                static_cast<int>(std::ceil((maxgrid[1]-mingrid[1])/dcell))+1,
                static_cast<int>(std::ceil((maxgrid[2]-mingrid[2])/dcell))+1
            }
            , m_g_size {m_ngrid[0]*m_ngrid[1]*m_ngrid[2]}
            , procid {get_MPI_Comm_rank()}
            , numprocs {get_MPI_Comm_size()}
            , m_p_x("Particles' 3D positions", m_p_size)
            , m_p_v("Particles' 3D velocity", m_p_size)
            , m_p_a("Particles' 3D accelerations", m_p_size)
            , m_p_dxdt("Particles' 3D dxdt", m_p_size)
            , m_p_sigma("Particles' 3D cauchy stress tensor", m_p_size)
            , m_p_strainrate("Particles' 3D cauchy strain rate tensor", m_p_size)
            , m_p_spinrate("Particles' 3D cauchy spin rate tensor (off-axis elements only)", m_p_size)
            , m_p_mass("Particles' mass", m_p_size)
            , m_p_rho("Particles' mass", m_p_size)
            , m_p_grid_idx("Particles' hashed grid indices", m_p_size)
            , m_g_momentum("Grid cells' 3D momentum", m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , m_g_force("Grid cells' 3D force", m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , m_g_mass("Grid cells' mass", m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , pg_npp {static_cast<int>(8*knl.radius*knl.radius*knl.radius)}
            , m_pg_nn("Particles' grid neighbour nodes", m_p_size*pg_npp)
            , m_pg_w("Particles' grid neighbour nodes' kernel values", m_p_size*pg_npp)
            , m_pg_dwdx("Particles' grid neighbour nodes' kernel gradient values", m_p_size*pg_npp)
            , f_momentum_boundary(m_g_momentum.d_view, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_force_boundary(m_g_force.d_view, m_ngrid[0], m_ngrid[1], m_ngrid[2])
            , f_map_gidx(dcell, mingrid.data(), m_ngrid.data(), m_p_x.d_view, m_p_grid_idx.d_view)
            , f_find_neighbour_nodes(dcell, mingrid.data(), m_ngrid.data(), static_cast<int>(knl.radius), m_p_x.d_view, 
                m_p_grid_idx.d_view, m_pg_nn.d_view, m_pg_w.d_view, m_pg_dwdx.d_view, knl)
            , f_map_p2g_mass(pg_npp, m_p_mass.d_view, m_g_mass.d_view, m_pg_nn.d_view, m_pg_w.d_view)
            , f_map_p2g_momentum(pg_npp, m_p_mass.d_view, m_p_v.d_view, m_g_momentum.d_view, m_pg_nn.d_view, 
                m_pg_w.d_view)
            , f_map_p2g_force(pg_npp, m_p_mass.d_view, m_p_rho.d_view, m_p_sigma.d_view, m_g_force.d_view, 
                m_pg_nn.d_view, m_pg_w.d_view, m_pg_dwdx.d_view, m_body_force[0], m_body_force[1], m_body_force[2])
            , f_map_g2p_acceleration(pg_npp, m_p_a.d_view, m_g_force.d_view, m_p_dxdt.d_view, m_g_momentum.d_view, 
                m_g_mass.d_view, m_pg_w.d_view, m_pg_nn.d_view)
            , f_map_g2p_strainrate(pg_npp, m_p_strainrate.d_view, m_p_spinrate.d_view, m_g_momentum.d_view, 
                m_pg_dwdx.d_view, m_g_mass.d_view, m_pg_nn.d_view)
            , f_g_update_momentum(m_g_momentum.d_view, m_g_force.d_view)
            , f_p_update_velocity(m_p_v.d_view, m_p_a.d_view)
            , f_p_update_position(m_p_x.d_view, m_p_dxdt.d_view)
            , f_p_update_density(m_p_rho.d_view, m_p_strainrate.d_view)
            , f_stress_update(m_p_sigma.d_view, m_p_strainrate.d_view, m_p_spinrate.d_view)
#ifdef GRAMPM_MPI
            , m_ORB_extents("List of all process' boundary boxes", numprocs)
            , m_ORB_send_halo("List of box indices of halo regions to send to neighbours", numprocs)
            , m_ORB_recv_halo("List of box indices of halo regions to receiv from neighbours", numprocs)
            , m_ORB_neighbours("IDs of processes that are neighbours", numprocs)
#endif
        {}

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(std::vector<particle<F>> &pv, std::array<F, 3> mingrid, std::array<F, 3> maxgrid, F dcell)
            : MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(
                static_cast<int>(pv.size()),
                mingrid,
                maxgrid,
                dcell)
        {
            for (int i = 0; i < m_p_size; ++i) {
                for (int d = 0; d < dims; ++d) {
                    m_p_x.h_view(i, d) = pv[i].x[d];
                    m_p_v.h_view(i, d) = pv[i].v[d];
                    m_p_a.h_view(i, d) = pv[i].a[d];
                    m_p_dxdt.h_view(i, d) = pv[i].dxdt[d];
                }
                for (int d = 0; d < voigt_tens_elems; ++d) {
                    m_p_sigma.h_view(i, d) = pv[i].sigma[d];
                    m_p_strainrate.h_view(i, d) = pv[i].strainrate[d];
                }
                for (int d = 0; d < spin_tens_elems; ++d) {
                    m_p_spinrate.h_view(i, d) = pv[i].spinrate[d];
                }
                m_p_mass.h_view(i) = pv[i].mass;
                m_p_rho.h_view(i) = pv[i].rho;
            }
        }

        template<typename F, typename kernel, typename stress_update, typename momentum_boundary, typename force_boundary>
        MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(std::string fname)
            : MPM_system<F, kernel, stress_update, momentum_boundary, force_boundary>::MPM_system(
                h5_get_nparticles(fname),
                h5_get_mingrid(fname),
                h5_get_maxgrid(fname),
                h5_get_cellsize(fname)
            )
        {
            herr_t status;
            hid_t f_id = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
            hid_t g_id = H5Gopen(f_id, "/particles", H5P_DEFAULT);
            hsize_t dims_vec[2] {m_p_x.extent(1), m_p_x.extent(0)}, 
                dims_tens[2] {m_p_sigma.extent(1), m_p_sigma.extent(0)},
                dims_spinrate_tens[2] {m_p_spinrate.extent(1), m_p_spinrate.extent(0)},
                dims_scalar[1] {m_p_mass.extent(0)};
            status = read_h5(2, dims_vec, m_p_x.h_view.data(), g_id, "x");
            status = read_h5(2, dims_vec, m_p_v.h_view.data(), g_id, "v");
            status = read_h5(2, dims_vec, m_p_a.h_view.data(), g_id, "a");
            status = read_h5(2, dims_vec, m_p_dxdt.h_view.data(), g_id, "dxdt");
            status = read_h5(2, dims_tens, m_p_sigma.h_view.data(), g_id, "sigma");
            status = read_h5(2, dims_tens, m_p_strainrate.h_view.data(), g_id, "strainrate");
            status = read_h5(2, dims_spinrate_tens, m_p_spinrate.h_view.data(), g_id, "spinrate");
            status = read_h5(2, dims_scalar, m_p_mass.h_view.data(), g_id, "mass");
            status = read_h5(2, dims_scalar, m_p_rho.h_view.data(), g_id, "rho");
            status = H5Gclose(g_id);
            status = H5Fclose(f_id);
        }
    }
}

#endif