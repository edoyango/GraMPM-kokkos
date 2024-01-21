#include <grampm.hpp>
#include <grampm-kokkos.hpp>
#include <grampm-kokkos-kernels.hpp>
#include <grampm-kokkos-functors-stressupdate.hpp>
#include <algorithm>
#include <grampm_kernels.hpp>
#include <array>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

using namespace GraMPM::accelerated;

TEST(stress_update, hookes_law) {
    
    const double dcell = 0.2;

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(1, std::array<double, 3>{0., 0., 0.}, std::array<double, 3>{0.99, 1.99, 2.99}, dcell);

    myMPM.f_stress_update.set_E_v(100., 0.25);

    myMPM.p_strainratexx(0) = 1.;
    myMPM.p_strainrateyy(0) = 2.;
    myMPM.p_strainratezz(0) = 3.;
    myMPM.p_strainratexy(0) = 4.;
    myMPM.p_strainratexz(0) = 5.;
    myMPM.p_strainrateyz(0) = 6.;

    myMPM.h2d();

    myMPM.p_update_stress(0.1);

    myMPM.d2h();

    ASSERT_DOUBLE_EQ(myMPM.p_sigmaxx(0), 32.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmayy(0), 40.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmazz(0), 48.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmaxy(0), 32.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmaxz(0), 40.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmayz(0), 48.);
}

TEST(stress_update, jaumann_rate) {
    
    const double dcell = 0.2;

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(1, std::array<double, 3>{0., 0., 0.}, std::array<double, 3>{0.99, 1.99, 2.99}, dcell);

    myMPM.f_stress_update.set_E_v(0., 0.);

    myMPM.p_spinratexy(0)= -0.8;
    myMPM.p_spinratexz(0)= -0.9;
    myMPM.p_spinrateyz(0)= -1.0;
    myMPM.p_sigmaxx(0) = 100.;
    myMPM.p_sigmayy(0) = 200.;
    myMPM.p_sigmazz(0) = 300.;
    myMPM.p_sigmaxy(0) = 400.;
    myMPM.p_sigmaxz(0) = 500.;
    myMPM.p_sigmayz(0) = 600.;

    ASSERT_DOUBLE_EQ(myMPM.p_spinratexy(0), -0.8);
    ASSERT_DOUBLE_EQ(myMPM.p_spinratexz(0), -0.9);
    ASSERT_DOUBLE_EQ(myMPM.p_spinrateyz(0), -1.0);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmaxx(0), 100.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmayy(0), 200.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmazz(0), 300.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmaxy(0), 400.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmaxz(0), 500.);
    ASSERT_DOUBLE_EQ(myMPM.p_sigmayz(0), 600.);

    myMPM.h2d();
    myMPM.p_update_stress(0.1);
    myMPM.d2h();

    EXPECT_DOUBLE_EQ(myMPM.p_sigmaxx(0),  254.);
    EXPECT_DOUBLE_EQ(myMPM.p_sigmayy(0),  256.);
    EXPECT_DOUBLE_EQ(myMPM.p_sigmazz(0),  90.);
    EXPECT_DOUBLE_EQ(myMPM.p_sigmaxy(0),  512.);
    EXPECT_DOUBLE_EQ(myMPM.p_sigmaxz(0),  526.);
    EXPECT_DOUBLE_EQ(myMPM.p_sigmayz(0),  534.);

}

TEST(stress_update, DP) {
    
    const double dcell = 0.2;

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::drucker_prager_elastoplastic<double>> 
        myMPM(1, std::array<double, 3>{0., 0., 0.}, std::array<double, 3>{0.99, 1.99, 2.99}, dcell);
    const double pi = std::acos(-1.);
    myMPM.f_stress_update.set_DP_params(pi/4., pi/36., 0., 100., 0.25);

    ASSERT_DOUBLE_EQ(myMPM.f_stress_update.phi, pi/4.);
    ASSERT_DOUBLE_EQ(myMPM.f_stress_update.psi, pi/36.);
    ASSERT_DOUBLE_EQ(myMPM.f_stress_update.cohesion, 0.);
    ASSERT_DOUBLE_EQ(myMPM.f_stress_update.E, 100.);
    ASSERT_DOUBLE_EQ(myMPM.f_stress_update.v, 0.25);

    // check elasto-plastic compression
    myMPM.p_strainratexx(0) = -1.;
    myMPM.p_strainrateyy(0) = -2.;
    myMPM.p_strainratezz(0) = -3.;
    myMPM.p_strainratexy(0) = -4.;
    myMPM.p_strainratexz(0) = -5.;
    myMPM.p_strainrateyz(0) = -6.;

    myMPM.h2d();
    myMPM.p_update_stress(0.1);
    myMPM.d2h();

    EXPECT_NEAR(myMPM.p_sigmaxx(0), -39.52509695, 1.e-8);
    EXPECT_NEAR(myMPM.p_sigmayy(0), -44.96397315, 1.e-8);
    EXPECT_NEAR(myMPM.p_sigmazz(0), -50.40284935, 1.e-8);
    EXPECT_NEAR(myMPM.p_sigmaxy(0), -21.75550481, 1.e-8);
    EXPECT_NEAR(myMPM.p_sigmaxz(0), -27.19438102, 1.e-8);
    EXPECT_NEAR(myMPM.p_sigmayz(0), -32.63325722, 1.e-8);

//     // check tensile correction
//     p.p_sigmaxx(0) = 0.;
//     p.p_sigmayy(0) = 0.;
//     p.p_sigmazz(0) = 0.;
//     p.p_sigmaxy(0) = 0.;
//     p.p_sigmaxz(0) = 0.;
//     p.p_sigmayz(0) = 0.;
//     p.p_strainratexx(0) = 1.;
//     p.p_strainrateyy(0) = 2.;
//     p.p_strainratezz(0) = 3.;
//     p.p_strainratexy(0) = 0.;
//     p.p_strainratexz(0) = 0.;
//     p.p_strainrateyz(0) = 0.;

//     p.p_update_stress(0.1);
//     REQUIRE(p.p_sigmaxx(0)==0.);
//     REQUIRE(p.p_sigmayy(0)==0.);
//     REQUIRE(p.p_sigmazz(0)==0.);
//     REQUIRE(p.p_sigmaxy(0)==0.);
//     REQUIRE(p.p_sigmaxz(0)==0.);
//     REQUIRE(p.p_sigmayz(0)==0.);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}