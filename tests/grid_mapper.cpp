// Ensuring that functions to determine grid neighbours do their job

#include <gtest/gtest.h>
#include <grampm.hpp>
#include <grampm/accelerated/core.hpp>
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/stressupdate.hpp>
#include <algorithm>
#include <array>

using namespace GraMPM::accelerated;

const double dcell = 0.1;
const std::array<double, 3> mingrid {-0.1, 0.05, 0.15}, maxgrid {0.1, 0.3, 0.5};
// std::array<double, 3> bf {0.};

const double dx = (maxgrid[0]-mingrid[0])/4;
const double dy = (maxgrid[1]-mingrid[1])/4;
const double dz = (maxgrid[2]-mingrid[2])/4;

const int correct_idxx[5] {0, 0, 1, 1, 2};
const int correct_idxy[5] {0, 0, 1, 1, 2};
const int correct_idxz[5] {0, 0, 1, 2, 3};
const int correct_ravelled_idx[5] {0, 0, 1*4*5+1*5+1, 1*4*5+1*5+2, 2*4*5+2*5+3};

TEST(map_particles_to_grid, particle_to_grid_assignment) {

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(5, mingrid, maxgrid, dcell);

    // initialize particles
    for (int i = 0; i < myMPM.p_size(); ++i) {
        myMPM.p_x(i) = i*dx + mingrid[0];
        myMPM.p_y(i) = i*dy + mingrid[1];
        myMPM.p_z(i) = i*dz + mingrid[2];
    }

    myMPM.h2d();

    myMPM.update_particle_to_cell_map();

    myMPM.d2h();

    for (int i = 0; i < myMPM.p_size(); ++i) {
        EXPECT_EQ(myMPM.p_grid_idx(i), correct_ravelled_idx[i]) << "Incorrect ravelled grid idx at " << i;
        std::array<int, dims> idx = myMPM.p_grid_idx_unravelled(i);
        EXPECT_EQ(idx[0], correct_idxx[i]) << "Incorrect unravelled grid idx (x) at " << i;
        EXPECT_EQ(idx[1], correct_idxy[i]) << "Incorrect unravelled grid idx (y) at " << i;
        EXPECT_EQ(idx[2], correct_idxz[i]) << "Incorrect unravelled grid idx (z) at " << i;
    }

}

TEST(map_particles_to_grid, neighbours_radius1) {

    MPM_system<double, kernels::linear_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(5, mingrid, maxgrid, dcell);

    for (int i = 0; i < myMPM.p_size(); ++i) {
        myMPM.p_x(i) = i*dx + mingrid[0];
        myMPM.p_y(i) = i*dy + mingrid[1];
        myMPM.p_z(i) = i*dz + mingrid[2];
    }

    myMPM.h2d();

    myMPM.update_particle_to_cell_map();

    myMPM.find_neighbour_nodes();

    myMPM.d2h();

    GraMPM::kernel_linear_bspline<double> knl(dcell);

    for (int i = 0; i < 5; ++i) {
        int n = 0;
        for (int di = 0; di <= 1; ++di) {
            for (int dj = 0; dj <= 1; ++dj) {
                for (int dk = 0; dk <= 1; ++dk) {
                    ASSERT_EQ(myMPM.pg_nn(i, n), correct_ravelled_idx[i] + di*5*4 + dj*5 + dk) << "incorrect neighbour node at " << i << " " << di << " " << dj  << " " << dk;
                    std::array<int, 3> idx = myMPM.p_grid_idx_unravelled(i);
                    // REQUIRE(p.pg_nn_dx(i, n)==p.p_x(i)-((idx[0]+di)*dcell+mingrid[0]));
                    // REQUIRE(p.pg_nn_dy(i, n)==p.p_y(i)-((idx[1]+dj)*dcell+mingrid[1]));
                    // REQUIRE(p.pg_nn_dz(i, n)==p.p_z(i)-((idx[2]+dk)*dcell+mingrid[2]));
                    double w, dwdx, dwdy, dwdz;
                    knl.w_dwdx(
                        myMPM.p_x(i)-((idx[0]+di)*dcell+mingrid[0]), 
                        myMPM.p_y(i)-((idx[1]+dj)*dcell+mingrid[1]), 
                        myMPM.p_z(i)-((idx[2]+dk)*dcell+mingrid[2]),
                        w,
                        dwdx, dwdy, dwdz);
                    // for some reason the values are different to double precision
                    EXPECT_NEAR(myMPM.pg_w(i, n), w, 1.e-14) << "incorrect neighbour node w value at " << i << " " << di << " " << dj  << " " << dk;
                    EXPECT_NEAR(myMPM.pg_dwdx(i, n), dwdx, 1.e-14) << "incorrect neighbour node dwdx value at " << i << " " << di << " " << dj  << " " << dk;
                    EXPECT_NEAR(myMPM.pg_dwdy(i, n), dwdy, 1.e-14) << "incorrect neighbour node dwdy value at " << i << " " << di << " " << dj  << " " << dk;
                    EXPECT_NEAR(myMPM.pg_dwdz(i, n), dwdz, 1.e-14) << "incorrect neighbour node dwdz value at " << i << " " << di << " " << dj  << " " << dk;
                    n++;
                }
            }
        }
    }
}

TEST(map_particles_to_grid, neighbours_radius2) {

    std::vector<GraMPM::particle<double>> pv;
    GraMPM::particle<double> p;
    p.x[0] = 0.01;
    p.x[1] = 0.16;
    p.x[2] = 0.26;
    pv.push_back(p);
    p.x[0] = 0.01;
    p.x[1] = 0.3;
    p.x[2] = 0.5;
    pv.push_back(p);

    // initialize particle system with cublic spline kernel
    const std::array<double, 3> mingrid {-0.1, 0.05, 0.15}, maxgrid {0.19, 0.4, 0.6};
    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(pv, mingrid, maxgrid, dcell);

    ASSERT_EQ(myMPM.p_size(), 2);
    ASSERT_EQ(myMPM.g_ngridx(), 4);
    ASSERT_EQ(myMPM.g_ngridy(), 5);
    ASSERT_EQ(myMPM.g_ngridz(), 6);

    int correct_ravelled_idx[2] {37, 45};

    myMPM.h2d();

    myMPM.update_particle_to_cell_map();

    myMPM.find_neighbour_nodes();

    myMPM.d2h();

    ASSERT_EQ(myMPM.p_grid_idx(0), correct_ravelled_idx[0]);
    ASSERT_EQ(myMPM.p_grid_idx(1), correct_ravelled_idx[1]);

    GraMPM::kernel_cubic_bspline<double> knlc(dcell);

    for (int i = 0; i < 2; ++i) {
        int n = 0;
        for (int di = -1; di <= 2; ++di) {
            for (int dj = -1; dj <=2; ++dj) {
                for (int dk = -1; dk <=2; ++dk) {
                    ASSERT_EQ(myMPM.pg_nn(i, n), correct_ravelled_idx[i] + di*6*5 + dj*6 + dk) << "incorrect neighbour node at " << i << " " << di << " " << dj  << " " << dk;
                    std::array<int, 3> idx = myMPM.p_grid_idx_unravelled(i);
                    double w, dwdx, dwdy, dwdz;
                    knlc.w_dwdx(
                        myMPM.p_x(i)-((idx[0]+di)*dcell+mingrid[0]), 
                        myMPM.p_y(i)-((idx[1]+dj)*dcell+mingrid[1]), 
                        myMPM.p_z(i)-((idx[2]+dk)*dcell+mingrid[2]), 
                        w,
                        dwdx, dwdy, dwdz);
                    // for some reason the values are different to double precision
                    EXPECT_NEAR(myMPM.pg_w(i, n), w, 1.e-14) << "incorrect neighbour node w value at " << i << " " << di << " " << dj  << " " << dk;
                    EXPECT_NEAR(myMPM.pg_dwdx(i, n), dwdx, 1.e-14) << "incorrect neighbour node dwdx value at " << i << " " << di << " " << dj  << " " << dk;
                    EXPECT_NEAR(myMPM.pg_dwdy(i, n), dwdy, 1.e-14) << "incorrect neighbour node dwdy value at " << i << " " << di << " " << dj  << " " << dk;
                    EXPECT_NEAR(myMPM.pg_dwdz(i, n), dwdz, 1.e-14) << "incorrect neighbour node dwdz value at " << i << " " << di << " " << dj  << " " << dk;
                    n++;
                }
            }
        }
    }

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}