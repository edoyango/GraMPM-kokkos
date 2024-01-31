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
#include <algorithm>

using namespace GraMPM::accelerated;


// const std::array<double, 3> bf {1., 2., 3.};

// GraMPM::kernel_linear_bspline<double> knl(dcell_in);

// GraMPM::MPM_system<double> myMPM(5, bf, knl, mingridx_in, maxgridx_in, dcell_in);

// assumes running with 4 procs
TEST(ORB, test_cax) {

    int numprocs, procid;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    ASSERT_EQ(numprocs, 4);

    // 4 procs stitched together creates 
    Kokkos::View<int***> p("number of particles in cells, local", 10, 5, 5);
    typename Kokkos::View<int***>::HostMirror h_p(p);
    Kokkos::deep_copy(h_p, 0);
    int minidx[3], maxidx[3];
    if (procid==0) {
        minidx[0] = 0; maxidx[0] = 20;
        minidx[1] = 0; maxidx[1] = 5;
        minidx[2] = 0; maxidx[2] = 5;
    } else if (procid == 1) {
        minidx[0] = 0; maxidx[0] = 20;
        minidx[1] = 5; maxidx[1] = 10;
        minidx[2] = 0; maxidx[2] = 5;
    } else if (procid == 2) {
        minidx[0] = 0; maxidx[0] = 20;
        minidx[1] = 0; maxidx[1] = 5;
        minidx[2] = 5; maxidx[2] = 10;
    } else if (procid == 3) {
        minidx[0] = 0; maxidx[0] = 20;
        minidx[1] = 5; maxidx[1] = 10;
        minidx[2] = 5; maxidx[2] = 10;
    }

    // rectangle, x longest
    if (procid==0) {
        for (int i = 1; i < 9; ++i) {
            for (int j = 1; j < 5; ++j) {
                for (int k = 1; k < 5; ++k) {
                    h_p(i, j, k) = 1;
                }
            }
        }
    } else if (procid==1) {
        for (int i = 1; i < 9; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 1; k < 5; ++k) {
                    h_p(i, j, k) = 1;
                }
            }
        }
    } else if (procid==2) {
        for (int i = 1; i < 9; ++i) {
            for (int j = 1; j < 5; ++j) {
                for (int k = 0; k < 2; ++k) {
                    h_p(i, j, k) = 1;
                }
            }
        }
    } else if (procid==3) {
        for (int i = 1; i < 9; ++i) {
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    h_p(i, j, k) = 1;
                }
            }
        }
    }

    Kokkos::deep_copy(p, h_p);

    int start_idx[3] {0, 0, 0}, end_idx[3] {10, 10, 10};

    int cax = choose_cut_axis(p, minidx, maxidx, start_idx, end_idx);

    EXPECT_EQ(cax, 0);

    end_idx[0] = 5;
    
    cax = choose_cut_axis(p, minidx, maxidx, start_idx, end_idx);

    EXPECT_EQ(cax, 1);

}

TEST(ORB, boundaries) {

    int numprocs, procid;
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    const double dcell_in = 0.1;
    const std::array<double, 3> mingridx_in {-0.1, -0.1, -0.1}, maxgridx_in {1.1, 2.1, 3.1};
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
                    p.x[0] = (i+0.5)*dcell_in/2.;
                    p.x[1] = (j+0.5)*dcell_in/2.;
                    p.x[2] = (k+0.5)*dcell_in/2.;
                    vp.push_back(p);
                }
                n++;
            }
        }
    }
    
    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingridx_in, maxgridx_in, dcell_in);

    ASSERT_EQ(myMPM.p_size(), 2000);

    myMPM.update_particle_to_cell_map();

    myMPM.ORB_determine_boundaries();

    if (procid == 0) {
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(), -0.1) << "incorrect mingridx from procid 0";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(), -0.1) << "incorrect mingridy from procid 0";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(), -0.1) << "incorrect mingridz from procid 0";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(), 0.5) << "incorrect maxgridx from procid 0";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(), 0.5) << "incorrect maxgridy from procid 0";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(), 3.2) << "incorrect maxgridz from procid 0";
    } else if (procid == 1) {
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(), -0.1) << "incorrect mingridx from procid 1";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(), 0.5) << "incorrect mingridy from procid 1";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(), -0.1) << "incorrect mingridz from procid 1";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(), 0.5) << "incorrect maxgridx from procid 1";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(), 2.2) << "incorrect maxgridy from procid 1";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(), 3.2) << "incorrect maxgridz from procid 1";
    } else if (procid == 2) {
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(), 0.5) << "incorrect mingridx from procid 2";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(), -0.1) << "incorrect mingridy from procid 2";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(), -0.1) << "incorrect mingridz from procid 2";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(), 1.3) << "incorrect maxgridx from procid 2";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(), 0.5) << "incorrect maxgridy from procid 2";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(), 3.2) << "incorrect maxgridz from procid 2";
    } else if (procid == 3) {
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(), 0.5) << "incorrect mingridx from procid 3";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(), 0.5) << "incorrect mingridy from procid 3";
        EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(), -0.1) << "incorrect mingridz from procid 3";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(), 1.3) << "incorrect maxgridx from procid 3";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(), 2.2) << "incorrect maxgridy from procid 3";
        EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(), 3.2) << "incorrect maxgridz from procid 3";
    }
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