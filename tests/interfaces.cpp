// tests to check that GraMPM and members have been initialized correctly

#include <grampm.hpp>
#include <algorithm>
#include <grampm_kernels.hpp>
#include <array>
#include <grampm-kokkos.hpp>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

const std::array<double, 3> mingridx_in {-0.1, 0.05, 0.15}, maxgridx_in {0.1, 0.3, 0.5};
const double dcell_in = 0.1;
// const std::array<double, 3> bf {1., 2., 3.};

// GraMPM::kernel_linear_bspline<double> knl(dcell_in);

// GraMPM::MPM_system<double> myMPM(5, bf, knl, mingridx_in, maxgridx_in, dcell_in);

TEST(initialization_getters_setters, grid_init) {

    GraMPM::accelerated::MPM_system<double> myMPM(0, mingridx_in, maxgridx_in, dcell_in);

    ASSERT_EQ(myMPM.g_cell_size(), dcell_in);

    // test array get interface
    std::array<double, dims> mingridx_out {myMPM.g_mingrid()}, maxgridx_out {myMPM.g_maxgrid()};

    for (int d = 0; d < dims; ++d) {
        EXPECT_EQ(mingridx_out[d], mingridx_in[d]) << "min extents of grid not saved properly in " << d << "th dim";
        EXPECT_EQ(maxgridx_out[d], maxgridx_in[d]) << "max extents of grid not saved properly in " << d << "th dim";
    }

    std::array<size_t, 3> ngridx_out = myMPM.g_ngrid();
    ASSERT_EQ(ngridx_out[0], 3) << "number of cells in x direction does not match expected";
    ASSERT_EQ(ngridx_out[1], 4) << "number of cells in y direction does not match expected";
    ASSERT_EQ(ngridx_out[2], 5) << "number of cells in z direction does not match expected";

    // test element-by-element get interface
    ASSERT_EQ(myMPM.g_mingridx(), mingridx_in[0]) << "x grid min extent not retrieved correctly";
    ASSERT_EQ(myMPM.g_mingridy(), mingridx_in[1]) << "y grid min extent not retrieved correctly";
    ASSERT_EQ(myMPM.g_mingridz(), mingridx_in[2]) << "z grid min extent not retrieved correctly";
    
    ASSERT_EQ(myMPM.g_maxgridx(), maxgridx_in[0]) << "x grid max extent not retrieved correctly";
    ASSERT_EQ(myMPM.g_maxgridy(), maxgridx_in[1]) << "y grid max extent not retrieved correctly";
    ASSERT_EQ(myMPM.g_maxgridz(), maxgridx_in[2]) << "z grid max extent not retrieved correctly";

    ASSERT_EQ(myMPM.g_ngridx(), 3) << "number of cells not retrieved correctly in x direction";
    ASSERT_EQ(myMPM.g_ngridy(), 4) << "number of cells not retrieved correctly in y direction";
    ASSERT_EQ(myMPM.g_ngridz(), 5) << "number of cells not retrieved correctly in z direction";
    ASSERT_EQ(myMPM.g_size(), 60) << "total number of cells not retrieved correctly";

    myMPM.h_zero_grid();

    for (size_t i = 0; i < myMPM.g_size(); ++i) {
        EXPECT_EQ(myMPM.g_mass(i), 0.) << "grid mass at index " << i << " not zeroed properly";
        EXPECT_EQ(myMPM.g_momentumx(i), 0.) << "grid x momentum at index " << i << " not zeroed properly";
        EXPECT_EQ(myMPM.g_momentumy(i), 0.) << "grid y momentum at index " << i << " not zeroed properly";
        EXPECT_EQ(myMPM.g_momentumz(i), 0.) << "grid z momentum at index " << i << " not zeroed properly";
        EXPECT_EQ(myMPM.g_forcex(i), 0.) << "grid x force at index " << i << " not zeroed properly";
        EXPECT_EQ(myMPM.g_forcey(i), 0.) << "grid y force at index " << i << " not zeroed properly";
        EXPECT_EQ(myMPM.g_forcez(i), 0.) << "grid z force at index " << i << " not zeroed properly";
    }

    for (size_t i = 0; i < myMPM.g_ngridx(); ++i) 
        for (size_t j = 0; j < myMPM.g_ngridy(); ++j)
            for (size_t k = 0; k < myMPM.g_ngridz(); ++k) {
                myMPM.g_mass(i, j, k) = (i+j+k)*1.;
                EXPECT_EQ(myMPM.g_mass(i, j, k), (i+j+k)*1.) << "grid mass at index " << i << " not set properly";
                myMPM.g_momentumx(i, j, k) = (i+j+k)*2.;
                EXPECT_EQ(myMPM.g_momentumx(i, j, k), (i+j+k)*2.) << "grid x momentum at index " << i << " not set properly";
                myMPM.g_momentumy(i, j, k) = (i+j+k)*3.;
                EXPECT_EQ(myMPM.g_momentumy(i, j, k), (i+j+k)*3.) << "grid y momentum at index " << i << " not set properly";;
                myMPM.g_momentumz(i, j, k) = (i+j+k)*4.;
                EXPECT_EQ(myMPM.g_momentumz(i, j, k), (i+j+k)*4.) << "grid z momentum at index " << i << " not set properly";;
                myMPM.g_forcex(i, j, k) = (i+j+k)*5.;
                EXPECT_EQ(myMPM.g_forcex(i, j, k), (i+j+k)*5.) << "grid x force at index " << i << " not set properly";
                myMPM.g_forcey(i, j, k) = (i+j+k)*6.;
                EXPECT_EQ(myMPM.g_forcey(i, j, k), (i+j+k)*6.) << "grid y force at index " << i << " not set properly";
                myMPM.g_forcez(i, j, k) = (i+j+k)*7.;
                EXPECT_EQ(myMPM.g_forcez(i, j, k), (i+j+k)*7.) << "grid z force at index " << i << " not set properly";
            }
}

TEST(initialization_getters_setters, particles_aggregate_init) {

    // create vector of particles
    std::vector<GraMPM::particle<double>> pv;

    for (int i = 0; i < 5; ++i) {
        pv.push_back(
            GraMPM::particle<double>(1.*i, 2.*i, 3.*i, 4.*i, 5.*i, 6.*i, 10.*i, 100.*i, -0.1*i, -0.2*i, -0.3*i, -0.4*i, 
                -0.5*i, -0.6*i, 7.*i, 8.*i, 9.*i, 10.*i, 11.*i, 12.*i, -0.7*i, -0.8*i, -0.9*i, -1.0*i, -1.1*i, -1.2*i,
                -1.3*i, -1.4*i, -1.5*i)
        );
    }

    GraMPM::accelerated::MPM_system<double> myMPM(pv, mingridx_in, maxgridx_in, dcell_in);

    ASSERT_EQ(myMPM.p_size(), 5) << "Particle number not assigned expected value";

    for (int i = 0; i < myMPM.p_size(); ++i) {
        EXPECT_EQ(myMPM.p_x(i), 1.*i) << "x coordinate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_y(i), 2.*i) << "y coordinate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_z(i), 3.*i) << "z coordinate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_vx(i), 4.*i) << "x velocity not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_vy(i), 5.*i) << "y velocity not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_vz(i), 6.*i) << "z velocity not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_ax(i), 7.*i) << "x acceleration not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_ay(i), 8.*i) << "y acceleration not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_az(i), 9.*i) << "z acceleration not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_dxdt(i), 10.*i) << "x position change rate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_dydt(i), 11.*i) << "y position change rate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_dzdt(i), 12.*i) << "z position change rate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_mass(i), 10.*i) << "mass not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_rho(i), 100.*i) << "rho not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmaxx(i), -0.1*i) << "sigmaxx not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmayy(i), -0.2*i) << "sigmayy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmazz(i), -0.3*i) << "sigmazz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmaxy(i), -0.4*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmaxz(i), -0.5*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmayz(i), -0.6*i) << "sigmayz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratexx(i), -0.7*i) << "sigmaxx not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainrateyy(i), -0.8*i) << "sigmayy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratezz(i), -0.9*i) << "sigmazz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratexy(i), -1.0*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratexz(i), -1.1*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainrateyz(i), -1.2*i) << "sigmayz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_spinratexy(i), -1.3*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_spinratexz(i), -1.4*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_spinrateyz(i), -1.5*i) << "sigmayz not assigned correct value for index " << i;
    }

}

TEST(initialization_getters_setters, particles_post_init) {

    GraMPM::accelerated::MPM_system<double> myMPM(5, mingridx_in, maxgridx_in, dcell_in);

    ASSERT_EQ(myMPM.p_size(), 5) << "Particle number not assigned expected value";

    for (int i = 0; i < myMPM.p_size(); ++i) {
        myMPM.p_x(i) = 1.*i;
        myMPM.p_y(i) = 2.*i;
        myMPM.p_z(i) = 3.*i;
        myMPM.p_vx(i) = 4.*i;
        myMPM.p_vy(i) = 5.*i;
        myMPM.p_vz(i) = 6.*i;
        myMPM.p_ax(i) = 7.*i;
        myMPM.p_ay(i) = 8.*i;
        myMPM.p_az(i) = 9.*i;
        myMPM.p_dxdt(i) = 10.*i;
        myMPM.p_dydt(i) = 11.*i;
        myMPM.p_dzdt(i) = 12.*i;
        myMPM.p_mass(i) = 10.*i;
        myMPM.p_rho(i) = 100.*i;
        myMPM.p_sigmaxx(i) = -0.1*i;
        myMPM.p_sigmayy(i) = -0.2*i;
        myMPM.p_sigmazz(i) = -0.3*i;
        myMPM.p_sigmaxy(i) = -0.4*i;
        myMPM.p_sigmaxz(i) = -0.5*i;
        myMPM.p_sigmayz(i) = -0.6*i;
        myMPM.p_strainratexx(i) = -0.7*i;
        myMPM.p_strainrateyy(i) = -0.8*i;
        myMPM.p_strainratezz(i) = -0.9*i;
        myMPM.p_strainratexy(i) = -1.0*i;
        myMPM.p_strainratexz(i) = -1.1*i;
        myMPM.p_strainrateyz(i) = -1.2*i;
        myMPM.p_spinratexy(i) = -1.3*i;
        myMPM.p_spinratexz(i) = -1.4*i;
        myMPM.p_spinrateyz(i) = -1.5*i;
    }

    // check aggregate getters
    for (int i = 0; i < myMPM.p_size(); ++i) {
        GraMPM::particle<double> p = myMPM.p_at(i);
        EXPECT_EQ(p.x[0], 1.*i) << "x coordinate not assigned correct value for index " << i;
        EXPECT_EQ(p.x[1], 2.*i) << "y coordinate not assigned correct value for index " << i;
        EXPECT_EQ(p.x[2], 3.*i) << "z coordinate not assigned correct value for index " << i;
        EXPECT_EQ(p.v[0], 4.*i) << "x velocity not assigned correct value for index " << i;
        EXPECT_EQ(p.v[1], 5.*i) << "y velocity not assigned correct value for index " << i;
        EXPECT_EQ(p.v[2], 6.*i) << "z velocity not assigned correct value for index " << i;
        EXPECT_EQ(p.a[0], 7.*i) << "x acceleration not assigned correct value for index " << i;
        EXPECT_EQ(p.a[1], 8.*i) << "y acceleration not assigned correct value for index " << i;
        EXPECT_EQ(p.a[2], 9.*i) << "z acceleration not assigned correct value for index " << i;
        EXPECT_EQ(p.dxdt[0], 10.*i) << "x position change rate not assigned correct value for index " << i;
        EXPECT_EQ(p.dxdt[1], 11.*i) << "y position change rate not assigned correct value for index " << i;
        EXPECT_EQ(p.dxdt[2], 12.*i) << "z position change rate not assigned correct value for index " << i;
        EXPECT_EQ(p.mass, 10.*i) << "mass not assigned correct value for index " << i;
        EXPECT_EQ(p.rho, 100.*i) << "rho not assigned correct value for index " << i;
        EXPECT_EQ(p.sigma[0], -0.1*i) << "sigmaxx not assigned correct value for index " << i;
        EXPECT_EQ(p.sigma[1], -0.2*i) << "sigmayy not assigned correct value for index " << i;
        EXPECT_EQ(p.sigma[2], -0.3*i) << "sigmazz not assigned correct value for index " << i;
        EXPECT_EQ(p.sigma[3], -0.4*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(p.sigma[4], -0.5*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(p.sigma[5], -0.6*i) << "sigmayz not assigned correct value for index " << i;
        EXPECT_EQ(p.strainrate[0], -0.7*i) << "strainratexx not assigned correct value for index " << i;
        EXPECT_EQ(p.strainrate[1], -0.8*i) << "strainrateyy not assigned correct value for index " << i;
        EXPECT_EQ(p.strainrate[2], -0.9*i) << "strainratezz not assigned correct value for index " << i;
        EXPECT_EQ(p.strainrate[3], -1.0*i) << "strainratexy not assigned correct value for index " << i;
        EXPECT_EQ(p.strainrate[4], -1.1*i) << "strainratexz not assigned correct value for index " << i;
        EXPECT_EQ(p.strainrate[5], -1.2*i) << "strainrateyz not assigned correct value for index " << i;
        EXPECT_EQ(p.spinrate[0], -1.3*i) << "spinratexy not assigned correct value for index " << i;
        EXPECT_EQ(p.spinrate[1], -1.4*i) << "spinratexz not assigned correct value for index " << i;
        EXPECT_EQ(p.spinrate[2], -1.5*i) << "spinrateyz not assigned correct value for index " << i;
    }
    
}

TEST(initialization_getters_setters, particles_transfer) {

    // create vector of particles
    std::vector<GraMPM::particle<double>> pv;

    for (int i = 0; i < 5; ++i) {
        pv.push_back(
            GraMPM::particle<double>(1.*i, 2.*i, 3.*i, 4.*i, 5.*i, 6.*i, 10.*i, 100.*i, -0.1*i, -0.2*i, -0.3*i, -0.4*i, 
                -0.5*i, -0.6*i, 7.*i, 8.*i, 9.*i, 10.*i, 11.*i, 12.*i, -0.7*i, -0.8*i, -0.9*i, -1.0*i, -1.1*i, -1.2*i,
                -1.3*i, -1.4*i, -1.5*i)
        );
    }

    GraMPM::accelerated::MPM_system<double> myMPM(pv, mingridx_in, maxgridx_in, dcell_in);

    // send data to device
    myMPM.h2d();

    // zero the host data
    for (int i = 0; i < myMPM.p_size(); ++i) {
        myMPM.p_x(i) = 0.;
        myMPM.p_y(i) = 0.;
        myMPM.p_z(i) = 0.;
        myMPM.p_vx(i) = 0.;
        myMPM.p_vy(i) = 0.;
        myMPM.p_vz(i) = 0.;
        myMPM.p_ax(i) = 0.;
        myMPM.p_ay(i) = 0.;
        myMPM.p_az(i) = 0.;
        myMPM.p_dxdt(i) = 0.;
        myMPM.p_dydt(i) = 0.;
        myMPM.p_dzdt(i) = 0.;
        myMPM.p_mass(i) = 0.;
        myMPM.p_rho(i) = 0.;
        myMPM.p_sigmaxx(i) = 0.;
        myMPM.p_sigmayy(i) = 0.;
        myMPM.p_sigmazz(i) = 0.;
        myMPM.p_sigmaxy(i) = 0.;
        myMPM.p_sigmaxz(i) = 0.;
        myMPM.p_sigmayz(i) = 0.;
        myMPM.p_strainratexx(i) = 0.;
        myMPM.p_strainrateyy(i) = 0.;
        myMPM.p_strainratezz(i) = 0.;
        myMPM.p_strainratexy(i) = 0.;
        myMPM.p_strainratexz(i) = 0.;
        myMPM.p_strainrateyz(i) = 0.;
        myMPM.p_spinratexy(i) = 0.;
        myMPM.p_spinratexz(i) = 0.;
        myMPM.p_spinrateyz(i) = 0.;
    }

    // retrieve data from device
    myMPM.d2h();

    for (int i = 0; i < myMPM.p_size(); ++i) {
        EXPECT_EQ(myMPM.p_x(i), 1.*i) << "x coordinate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_y(i), 2.*i) << "y coordinate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_z(i), 3.*i) << "z coordinate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_vx(i), 4.*i) << "x velocity not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_vy(i), 5.*i) << "y velocity not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_vz(i), 6.*i) << "z velocity not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_ax(i), 7.*i) << "x acceleration not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_ay(i), 8.*i) << "y acceleration not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_az(i), 9.*i) << "z acceleration not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_dxdt(i), 10.*i) << "x position change rate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_dydt(i), 11.*i) << "y position change rate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_dzdt(i), 12.*i) << "z position change rate not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_mass(i), 10.*i) << "mass not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_rho(i), 100.*i) << "rho not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmaxx(i), -0.1*i) << "sigmaxx not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmayy(i), -0.2*i) << "sigmayy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmazz(i), -0.3*i) << "sigmazz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmaxy(i), -0.4*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmaxz(i), -0.5*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_sigmayz(i), -0.6*i) << "sigmayz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratexx(i), -0.7*i) << "sigmaxx not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainrateyy(i), -0.8*i) << "sigmayy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratezz(i), -0.9*i) << "sigmazz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratexy(i), -1.0*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainratexz(i), -1.1*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_strainrateyz(i), -1.2*i) << "sigmayz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_spinratexy(i), -1.3*i) << "sigmaxy not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_spinratexz(i), -1.4*i) << "sigmaxz not assigned correct value for index " << i;
        EXPECT_EQ(myMPM.p_spinrateyz(i), -1.5*i) << "sigmayz not assigned correct value for index " << i;
    }

}

// TEST_CASE("IO", "[myMPM]") {
//     // currently incomplete

//     myMPM.save_to_file("testfile", 1);

//     GraMPM::MPM_system<double> myMPM3("testfile0000001", bf, knl, mingridx_in, maxgridx_in, dcell_in);

//     REQUIRE(myMPM3.p_size()==5);

//     // check that manual setter functions work
//     for (int i = 0; i < 1; ++i) {
//         REQUIRE(myMPM3.p_x(i)==-1.*i);
//         REQUIRE(myMPM3.p_y(i)==-2.*i);
//         REQUIRE(myMPM3.p_z(i)==-3.*i);
//         REQUIRE(myMPM3.p_vx(i)==-4.*i);
//         REQUIRE(myMPM3.p_vy(i)==-5.*i);
//         REQUIRE(myMPM3.p_vz(i)==-6.*i);
//         REQUIRE(myMPM3.p_ax(i)==-7.*i);
//         REQUIRE(myMPM3.p_ay(i)==-8.*i);
//         REQUIRE(myMPM3.p_az(i)==-9.*i);
//         REQUIRE(myMPM3.p_dxdt(i)==-10.*i);
//         REQUIRE(myMPM3.p_dydt(i)==-11.*i);
//         REQUIRE(myMPM3.p_dzdt(i)==-12.*i);
//         REQUIRE(myMPM3.p_mass(i)==30.*i);
//         REQUIRE(myMPM3.p_rho(i)==40.*i);
//         REQUIRE(myMPM3.p_sigmaxx(i)==-0.1*i);
//         REQUIRE(myMPM3.p_sigmayy(i)==-0.2*i);
//         REQUIRE(myMPM3.p_sigmazz(i)==-0.3*i);
//         REQUIRE(myMPM3.p_sigmaxy(i)==-0.4*i);
//         REQUIRE(myMPM3.p_sigmaxz(i)==-0.5*i);
//         REQUIRE(myMPM3.p_sigmayz(i)==-0.6*i);
//         REQUIRE(myMPM3.p_strainratexx(i)==-0.7*i);
//         REQUIRE(myMPM3.p_strainrateyy(i)==-0.8*i);
//         REQUIRE(myMPM3.p_strainratezz(i)==-0.9*i);
//         REQUIRE(myMPM3.p_strainratexy(i)==-1.0*i);
//         REQUIRE(myMPM3.p_strainratexz(i)==-1.1*i);
//         REQUIRE(myMPM3.p_strainrateyz(i)==-1.2*i);
//         REQUIRE(myMPM3.p_spinratexy(i)==-1.3*i);
//         REQUIRE(myMPM3.p_spinratexz(i)==-1.4*i);
//         REQUIRE(myMPM3.p_spinrateyz(i)==-1.5*i);
//     }
// }

// TEST_CASE("Check clearing and resizing", "[myMPM]") {
//     myMPM.p_clear();
//     REQUIRE(myMPM.p_empty());
//     for (int i = 0; i < 3; ++i) {
//         GraMPM::particle<double> p(
//             std::array<double, 3> {-1.*i, -2.*i, -3.*i},
//             std::array<double, 3> {-4.*i, -5.*i, -6.*i},
//             10.*i, 100.*i,
//             std::array<double, 6> {-7.*i, -8.*i, -9.*i, -10.*i, -11.*i, -12.*i},
//             std::array<double, 3> {-13.*i, -14.*i, -15.*i},
//             std::array<double, 3> {-16.*i, -17.*i, -18.*i},
//             std::array<double, 6> {-19.*i, -20.*i, -21.*i, -22.*i, -23.*i, -24.*i},
//             std::array<double, 3> {-25.*i, -26.*i, -27.*i}
//         );
//         myMPM.p_push_back(p);
//     }

//     myMPM.p_resize(6);
//     REQUIRE(myMPM.p_size()==6);
    
//     for (int i = 0; i < 3; ++i) {
//         REQUIRE(myMPM.p_x(i)==-1.*i);
//         REQUIRE(myMPM.p_y(i)==-2.*i);
//         REQUIRE(myMPM.p_z(i)==-3.*i);
//         REQUIRE(myMPM.p_vx(i)==-4.*i);
//         REQUIRE(myMPM.p_vy(i)==-5.*i);
//         REQUIRE(myMPM.p_vz(i)==-6.*i);
//         REQUIRE(myMPM.p_mass(i)==10.*i);
//         REQUIRE(myMPM.p_rho(i)==100.*i);
//         REQUIRE(myMPM.p_sigmaxx(i)==-7*i);
//         REQUIRE(myMPM.p_sigmayy(i)==-8*i);
//         REQUIRE(myMPM.p_sigmazz(i)==-9*i);
//         REQUIRE(myMPM.p_sigmaxy(i)==-10*i);
//         REQUIRE(myMPM.p_sigmaxz(i)==-11*i);
//         REQUIRE(myMPM.p_sigmayz(i)==-12*i);
//         REQUIRE(myMPM.p_ax(i)==-13.*i);
//         REQUIRE(myMPM.p_ay(i)==-14.*i);
//         REQUIRE(myMPM.p_az(i)==-15.*i);
//         REQUIRE(myMPM.p_dxdt(i)==-16.*i);
//         REQUIRE(myMPM.p_dydt(i)==-17.*i);
//         REQUIRE(myMPM.p_dzdt(i)==-18.*i);
//         REQUIRE(myMPM.p_strainratexx(i)==-19.*i);
//         REQUIRE(myMPM.p_strainrateyy(i)==-20.*i);
//         REQUIRE(myMPM.p_strainratezz(i)==-21.*i);
//         REQUIRE(myMPM.p_strainratexy(i)==-22.*i);
//         REQUIRE(myMPM.p_strainratexz(i)==-23.*i);
//         REQUIRE(myMPM.p_strainrateyz(i)==-24.*i);
//         REQUIRE(myMPM.p_spinratexy(i)==-25.*i);
//         REQUIRE(myMPM.p_spinratexz(i)==-26.*i);
//         REQUIRE(myMPM.p_spinrateyz(i)==-27.*i);
//     }
//     for (int i = 3; i < 6; ++i) {
//         REQUIRE(myMPM.p_x(i)==0.);
//         REQUIRE(myMPM.p_y(i)==0.);
//         REQUIRE(myMPM.p_z(i)==0.);
//         REQUIRE(myMPM.p_vx(i)==0.);
//         REQUIRE(myMPM.p_vy(i)==0.);
//         REQUIRE(myMPM.p_vz(i)==0.);
//         REQUIRE(myMPM.p_ax(i)==0.);
//         REQUIRE(myMPM.p_ay(i)==0.);
//         REQUIRE(myMPM.p_az(i)==0.);
//         REQUIRE(myMPM.p_dxdt(i)==0.);
//         REQUIRE(myMPM.p_dydt(i)==0.);
//         REQUIRE(myMPM.p_dzdt(i)==0.);
//         REQUIRE(myMPM.p_mass(i)==0.);
//         REQUIRE(myMPM.p_rho(i)==0.);
//         REQUIRE(myMPM.p_sigmaxx(i)==0.);
//         REQUIRE(myMPM.p_sigmayy(i)==0.);
//         REQUIRE(myMPM.p_sigmazz(i)==0.);
//         REQUIRE(myMPM.p_sigmaxy(i)==0.);
//         REQUIRE(myMPM.p_sigmaxz(i)==0.);
//         REQUIRE(myMPM.p_sigmayz(i)==0.);
//         REQUIRE(myMPM.p_strainratexx(i)==0.);
//         REQUIRE(myMPM.p_strainrateyy(i)==0.);
//         REQUIRE(myMPM.p_strainratezz(i)==0.);
//         REQUIRE(myMPM.p_strainratexy(i)==0.);
//         REQUIRE(myMPM.p_strainratexz(i)==0.);
//         REQUIRE(myMPM.p_strainrateyz(i)==0.);
//         REQUIRE(myMPM.p_spinratexy(i)==0.);
//         REQUIRE(myMPM.p_spinratexz(i)==0.);
//         REQUIRE(myMPM.p_spinrateyz(i)==0.);
//         REQUIRE(myMPM.p_grid_idx(i)==0);
//     }
// }

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}