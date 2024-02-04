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
        proc_box.min[0] = 0; proc_box.max[0] = 20;
        proc_box.min[1] = 0; proc_box.max[1] = 5;
        proc_box.min[2] = 0; proc_box.max[2] = 5;
    } else if (procid == 1) {
        proc_box.min[0] = 0; proc_box.max[0] = 20;
        proc_box.min[1] = 5; proc_box.max[1] = 10;
        proc_box.min[2] = 0; proc_box.max[2] = 5;
    } else if (procid == 2) {
        proc_box.min[0] = 0; proc_box.max[0] = 20;
        proc_box.min[1] = 0; proc_box.max[1] = 5;
        proc_box.min[2] = 5; proc_box.max[2] = 10;
    } else if (procid == 3) {
        proc_box.min[0] = 0; proc_box.max[0] = 20;
        proc_box.min[1] = 5; proc_box.max[1] = 10;
        proc_box.min[2] = 5; proc_box.max[2] = 10;
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
        node_box.min[d] = 0;
        node_box.max[d] = 10;
    }

    int cax = choose_cut_axis(p, proc_box, node_box);

    EXPECT_EQ(cax, 0);

    node_box.max[0] = 5;
    
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

    myMPM.d2h();

    EXPECT_EQ(myMPM.ORB_min_idxx(0),                0) << "incorrect min_idxx for proc 0 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxy(0),                0) << "incorrect min_idxy for proc 0 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxz(0),                0) << "incorrect min_idxz for proc 0 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxx(0),                6) << "incorrect max_idxx for proc 0 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxy(0),                6) << "incorrect max_idxy for proc 0 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxz(0), myMPM.g_ngridz()) << "incorrect max_idxz for proc 0 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxx(1),                0) << "incorrect min_idxx for proc 1 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxy(1),                6) << "incorrect min_idxy for proc 1 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxz(1),                0) << "incorrect min_idxz for proc 1 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxx(1),                6) << "incorrect max_idxx for proc 1 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxy(1), myMPM.g_ngridy()) << "incorrect max_idxy for proc 1 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxz(1), myMPM.g_ngridz()) << "incorrect max_idxz for proc 1 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxx(2),                6) << "incorrect min_idxx for proc 2 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxy(2),                0) << "incorrect min_idxy for proc 2 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxz(2),                0) << "incorrect min_idxz for proc 2 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxx(2), myMPM.g_ngridx()) << "incorrect max_idxx for proc 2 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxy(2),                6) << "incorrect max_idxy for proc 2 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxz(2), myMPM.g_ngridz()) << "incorrect max_idxz for proc 2 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxx(3),                6) << "incorrect min_idxx for proc 3from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxy(3),                6) << "incorrect min_idxy for proc 3from procid " << procid;
    EXPECT_EQ(myMPM.ORB_min_idxz(3),                0) << "incorrect min_idxz for proc 3 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxx(3), myMPM.g_ngridx()) << "incorrect max_idxx for proc 3 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxy(3), myMPM.g_ngridy()) << "incorrect max_idxy for proc 3 from procid " << procid;
    EXPECT_EQ(myMPM.ORB_max_idxz(3), myMPM.g_ngridz()) << "incorrect max_idxz for proc 3 from procid " << procid;

    EXPECT_EQ(myMPM.ORB_n_neighbours(), 3);
    n = 0;
    for (int i = 0; i < numprocs; ++i) {
        if (i != procid) {
            EXPECT_EQ(myMPM.ORB_neighbour(n), i) << "Incorrect neighbour at " << n << " from procid " << procid;
            n++;
        }
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