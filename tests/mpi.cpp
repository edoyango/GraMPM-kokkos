// tests to check that GraMPM and members have been initialized correctly

#include <algorithm>
#include <grampm_kernels.hpp>
#include <array>
#include <grampm/accelerated/core.hpp>
#include <grampm/accelerated/stressupdate.hpp>
#include <grampm/accelerated/kernels.hpp>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <mpi.h>

using namespace GraMPM::accelerated;


// const std::array<double, 3> bf {1., 2., 3.};

// GraMPM::kernel_linear_bspline<double> knl(dcell_in);

// GraMPM::MPM_system<double> myMPM(5, bf, knl, mingridx_in, maxgridx_in, dcell_in);

// assumes running with 4 procs
TEST(ORB, uniform_square) {

    int numprocs, procid;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    const std::array<double, 3> mingridx_in {-0.1, -0.1, -0.1}, maxgridx_in {1.1, 1.1, 1.1};
    const double dcell_in = 0.1;


    const int xp = 20, yp = 20, zp = 20, ntotal = xp*yp*zp;

    const int nperproc = std::ceil(static_cast<double>(ntotal)/numprocs);

    ASSERT_EQ(nperproc, 2000);
    const int nstart = procid*nperproc;
    const int nend = std::min(ntotal, (procid+1)*nperproc);
    
    int nlocal = 0, n = 0;
    std::vector<GraMPM::particle<double>> vp;
    for (int i = 0; i < xp; ++i) {
        for (int j = 0; j < yp; ++j) {
            for (int k = 0; k < zp; ++k) {
                if (n >= nstart && n < nend) {
                    GraMPM::particle<double> p;
                    p.x[0] = mingridx_in[0] + (i+0.5)*dcell_in/2.;
                    p.x[1] = mingridx_in[1] + (j+0.5)*dcell_in/2.;
                    p.x[2] = mingridx_in[2] + (k+0.5)*dcell_in/2.;
                    vp.push_back(p);
                }
                n++;
            }
        }
    }
    
    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingridx_in, maxgridx_in, dcell_in);

    ASSERT_EQ(myMPM.p_size(), 2000);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    MPI_Finalize();
    return success;
}