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
#include <grampm/extra.hpp>

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
    typename Kokkos::View<int***>::HostMirror h_p(create_mirror_view(p));
    Kokkos::deep_copy(h_p, 0);
    box<int> proc_box;
    if (procid==0) {
        proc_box.start[0] = 0; proc_box.end[0] = 20;
        proc_box.start[1] = 0; proc_box.end[1] = 5;
        proc_box.start[2] = 0; proc_box.end[2] = 5;
    } else if (procid == 1) {
        proc_box.start[0] = 0; proc_box.end[0] = 20;
        proc_box.start[1] = 5; proc_box.end[1] = 10;
        proc_box.start[2] = 0; proc_box.end[2] = 5;
    } else if (procid == 2) {
        proc_box.start[0] = 0; proc_box.end[0] = 20;
        proc_box.start[1] = 0; proc_box.end[1] = 5;
        proc_box.start[2] = 5; proc_box.end[2] = 10;
    } else if (procid == 3) {
        proc_box.start[0] = 0; proc_box.end[0] = 20;
        proc_box.start[1] = 5; proc_box.end[1] = 10;
        proc_box.start[2] = 5; proc_box.end[2] = 10;
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

    box<int> node_box;
    for (int d = 0; d < 3; ++d) {
        node_box.start[d] = 0;
        node_box.end[d] = 10;
    }

    int cax = choose_cut_axis(p, proc_box, node_box);

    EXPECT_EQ(cax, 0);

    node_box.end[0] = 5;
    
    cax = choose_cut_axis(p, proc_box, node_box);

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

    EXPECT_EQ(nperproc, 2000);
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

    EXPECT_EQ(myMPM.p_size(), 2000);

    myMPM.h2d();

    myMPM.update_particle_to_cell_map();

    myMPM.ORB_determine_boundaries();

    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(0), -0.1) << "incorrect mingridx for proc 0 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(0), -0.1) << "incorrect mingridy for proc 0 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(0), -0.1) << "incorrect mingridz for proc 0 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(0), 0.5) << "incorrect maxgridx for proc 0 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(0), 0.5) << "incorrect maxgridy for proc 0 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(0), 3.2) << "incorrect maxgridz for proc 0 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(1), -0.1) << "incorrect mingridx for proc 1 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(1), 0.5) << "incorrect mingridy for proc 1 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(1), -0.1) << "incorrect mingridz for proc 1 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(1), 0.5) << "incorrect maxgridx for proc 1 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(1), 2.2) << "incorrect maxgridy for proc 1 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(1), 3.2) << "incorrect maxgridz for proc 1 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(2), 0.5) << "incorrect mingridx for proc 2 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(2), -0.1) << "incorrect mingridy for proc 2 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(2), -0.1) << "incorrect mingridz for proc 2 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(2), 1.3) << "incorrect maxgridx for proc 2 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(2), 0.5) << "incorrect maxgridy for proc 2 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(2), 3.2) << "incorrect maxgridz for proc 2 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridx(3), 0.5) << "incorrect mingridx for proc 3from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridy(3), 0.5) << "incorrect mingridy for proc 3from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_mingridz(3), -0.1) << "incorrect mingridz for proc 3 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridx(3), 1.3) << "incorrect maxgridx for proc 3 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridy(3), 2.2) << "incorrect maxgridy for proc 3 from procid " << procid;
    EXPECT_DOUBLE_EQ(myMPM.ORB_maxgridz(3), 3.2) << "incorrect maxgridz for proc 3 from procid " << procid;
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