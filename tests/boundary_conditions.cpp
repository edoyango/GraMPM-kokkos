#include <grampm/accelerated/core.hpp>
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/stressupdate.hpp>
#include <gtest/gtest.h>

using namespace GraMPM::accelerated;

struct apply_lower_momentum {
    const int itimestep;
    const double dt;
    const double ngridx, ngridy, ngridz;
    const Kokkos::View<double*[3]> data;
    apply_lower_momentum(Kokkos::View<double*[3]> data_, double ngridx_, double ngridy_, double ngridz_, double dt_,
        int itimestep_)
        : data {data_} 
        , ngridx {ngridx_}
        , ngridy {ngridy_}
        , ngridz {ngridz_}
        , dt {dt_}
        , itimestep {itimestep_}
    {};
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, const int k) const {
        if (k == 0) {
            const int idx = i*ngridy*ngridz + j*ngridz + k;
            data(idx, 0) = -1;
            data(idx, 1) = -1;
            data(idx, 2) = -1;
        }
    }
};

struct apply_lower_force {
    const int itimestep;
    const double dt;
    const double ngridx, ngridy, ngridz;
    const Kokkos::View<double*[3]> data;
    apply_lower_force(Kokkos::View<double*[3]> data_, double ngridx_, double ngridy_, double ngridz_, double dt_, 
        int itimestep_)
        : data {data_} 
        , ngridx {ngridx_}
        , ngridy {ngridy_}
        , ngridz {ngridz_}
        , dt {dt_}
        , itimestep {itimestep_}
    {};
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, const int k) const {
        if (k == 0) {
            const int idx = i*ngridy*ngridz + j*ngridz + k;
            data(idx, 0) = -2;
            data(idx, 1) = -2;
            data(idx, 2) = -2;
        }
    }
};

TEST(check_boundary, all) {

    // check that the function works when initialized with the grid
    const double dcell = 0.2;
    // GraMPM::grid<double> g(0., 0., 0., 0.99, 1.99, 2.99, dcell, apply_lower_momentum, apply_lower_force);
    std::array<double, 3> bf, mingrid {0., 0., 0.}, maxgrid {0.99, 1.99, 2.99};
    MPM_system<double, kernels::cubic_bspline<double>, empty_stress_update<double>, apply_lower_momentum, apply_lower_force>
        myMPM(0, mingrid, maxgrid, dcell);

    for (int i = 0; i < myMPM.g_size(); ++i) {
        myMPM.g_momentumx(i) = 1.;
        myMPM.g_momentumy(i) = 1.;
        myMPM.g_momentumz(i) = 1.;
        myMPM.g_forcex(i) = 2.;
        myMPM.g_forcey(i) = 2.;
        myMPM.g_forcez(i) = 2.;
    }

    myMPM.h2d();

    myMPM.g_apply_momentum_boundary_conditions(1, 0.);
    myMPM.g_apply_force_boundary_conditions(1, 0.);

    myMPM.d2h();

    for (int i = 0; i < myMPM.g_ngridx(); ++i)
        for (int j = 0; j < myMPM.g_ngridy(); ++j)
            for (int k = 0; k < myMPM.g_ngridz(); ++k) {
                if (k == 0) {
                    EXPECT_DOUBLE_EQ(myMPM.g_momentumx(i, j, k), -1.) << "grid momentum x at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_momentumy(i, j, k), -1.) << "grid momentum y at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_momentumz(i, j, k), -1.) << "grid momentum z at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_forcex(i, j, k), -2.) << "grid force x at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_forcey(i, j, k), -2.) << "grid force y at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_forcez(i, j, k), -2.) << "grid force z at " << i << " " << j << " " << k << " set incorrectly";
                } else {
                    EXPECT_DOUBLE_EQ(myMPM.g_momentumx(i, j, k), 1.) << "grid momentum x at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_momentumy(i, j, k), 1.) << "grid momentum y at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_momentumz(i, j, k), 1.) << "grid momentum z at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_forcex(i, j, k), 2.) << "grid force x at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_forcey(i, j, k), 2.) << "grid force y at " << i << " " << j << " " << k << " set incorrectly";
                    EXPECT_DOUBLE_EQ(myMPM.g_forcez(i, j, k), 2.) << "grid force z at " << i << " " << j << " " << k << " set incorrectly";
                }
            }

    // // setting new momentum boundary condition function
    // myMPM.g_set_momentum_boundary_function(apply_west_momentum);
    // myMPM.g_apply_momentum_boundary_conditions(1, 0.);

    // for (int i = 0; i < myMPM.g_ngridx(); ++i)
    //     for (int j = 0; j < myMPM.g_ngridy(); ++j)
    //         for (int k = 0; k < myMPM.g_ngridz(); ++k) {
    //             if (i == 0) {
    //                 REQUIRE(myMPM.g_momentumx(i, j, k)==-3.);
    //                 REQUIRE(myMPM.g_momentumy(i, j, k)==-3.);
    //                 REQUIRE(myMPM.g_momentumz(i, j, k)==-3.);
    //             } else if (k == 0) {
    //                 REQUIRE(myMPM.g_momentumx(i, j, k)==-1.);
    //                 REQUIRE(myMPM.g_momentumy(i, j, k)==-1.);
    //                 REQUIRE(myMPM.g_momentumz(i, j, k)==-1.);
    //             } else {
    //                 REQUIRE(myMPM.g_momentumx(i, j, k)==1.);
    //                 REQUIRE(myMPM.g_momentumy(i, j, k)==1.);
    //                 REQUIRE(myMPM.g_momentumz(i, j, k)==1.);
    //             }
    //         }

    // // setting new force boundary condition function
    // myMPM.g_set_force_boundary_function(apply_west_force);
    // myMPM.g_apply_force_boundary_conditions(1, 0.);

    // for (int i = 0; i < myMPM.g_ngridx(); ++i)
    //     for (int j = 0; j < myMPM.g_ngridy(); ++j)
    //         for (int k = 0; k < myMPM.g_ngridz(); ++k) {
    //             if (i == 0) {
    //                 REQUIRE(myMPM.g_forcex(i, j, k)==-4.);
    //                 REQUIRE(myMPM.g_forcey(i, j, k)==-4.);
    //                 REQUIRE(myMPM.g_forcez(i, j, k)==-4.);
    //             } else if (k == 0) {
    //                 REQUIRE(myMPM.g_forcex(i, j, k)==-2.);
    //                 REQUIRE(myMPM.g_forcey(i, j, k)==-2.);
    //                 REQUIRE(myMPM.g_forcez(i, j, k)==-2.);
    //             } else {
    //                 REQUIRE(myMPM.g_forcex(i, j, k)==2.);
    //                 REQUIRE(myMPM.g_forcey(i, j, k)==2.);
    //                 REQUIRE(myMPM.g_forcez(i, j, k)==2.);
    //             }
    //         }

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}