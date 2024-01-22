#include <grampm.hpp>
#include <grampm/accelerated/core.hpp>
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/stressupdate.hpp>
#include <algorithm>
#include <grampm_kernels.hpp>
#include <array>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

using namespace GraMPM::accelerated;

typedef MPM_system<double, kernels::linear_bspline<double>, functors::stress_update::hookes_law<double>> MPM_type;

MPM_type generate_mpm() {
    double dcell = 0.2;
    std::vector<GraMPM::particle<double>> vp;
    for (int i = 0; i < 5; ++i) {
        GraMPM::particle<double> p;
        p.x[0] = i*dcell/2.;
        p.x[1] = i*dcell/2.;
        p.x[2] = i*dcell/2.;
        p.v[0] = p.x[0]*i;
        p.v[1] = p.x[1]*i;
        p.v[2] = p.x[2]*i;
        p.a[0] = 1.*p.x[0]*i*i;
        p.a[1] = 2.*p.x[1]*i*i;
        p.a[2] = 3.*p.x[2]*i*i;
        p.dxdt[0] = 1.*p.x[0]*i*i;
        p.dxdt[1] = 2.*p.x[1]*i*i;
        p.dxdt[2] = 3.*p.x[2]*i*i;
        p.rho = 1000.;
        p.strainrate[0] = i*0.1;
        p.strainrate[1] = i*0.2;
        p.strainrate[2] = i*0.3;
        vp.push_back(p);
    }

    MPM_type myMPM(vp, std::array<double, 3> {0., 0., 0.}, std::array<double, 3> {0.99, 1.99, 2.99}, dcell);
    for (int i = 0; i < myMPM.g_ngridx(); ++i) {
        for (int j = 0; j < myMPM.g_ngridy(); ++j) {
            for (int k = 0; k < myMPM.g_ngridz(); ++k) {
                myMPM.g_momentumx(i, j, k) = 1.*(i+j+k);
                myMPM.g_momentumy(i, j, k) = 2.*(i+j+k);
                myMPM.g_momentumz(i, j, k) = 3.*(i+j+k);
                myMPM.g_forcex(i, j, k) = 1.;
                myMPM.g_forcey(i, j, k) = 2.;
                myMPM.g_forcez(i, j, k) = 3.;
            }
        }
    }
    return myMPM;
}

TEST(updates, all) {

    MPM_type myMPM = generate_mpm();

    myMPM.h2d();
    myMPM.g_update_momentum(0.1);
    myMPM.p_update_velocity(0.2);
    myMPM.p_update_position(0.3);
    myMPM.p_update_density(0.4);
    myMPM.d2h();

    // checking grid momentum values
    for (int i = 0; i < myMPM.g_ngridx(); ++i) {
        for (int j = 0; j < myMPM.g_ngridy(); ++j) {
            for (int k = 0; k < myMPM.g_ngridz(); ++k) {
                EXPECT_DOUBLE_EQ(myMPM.g_momentumx(i, j, k), 1.*(i+j+k)+0.1*1.) << "incorrect updated grid momentum x at " << i << ' ' << j << ' ' << k;
                EXPECT_DOUBLE_EQ(myMPM.g_momentumy(i, j, k), 2.*(i+j+k)+0.1*2.) << "incorrect updated grid momentum y at " << i << ' ' << j << ' ' << k;
                EXPECT_DOUBLE_EQ(myMPM.g_momentumz(i, j, k), 3.*(i+j+k)+0.1*3.) << "incorrect updated grid momentum z at " << i << ' ' << j << ' ' << k;
            }
        }
    }

    // checking particle values
    for (int i = 0; i < myMPM.p_size(); ++i) {
        EXPECT_DOUBLE_EQ(myMPM.p_vx(i), myMPM.g_cell_size()/2.*(i*i+0.2*i*i*i)) << "incorrect updated particle velocity x at " << i;
        EXPECT_DOUBLE_EQ(myMPM.p_vy(i), myMPM.g_cell_size()/2.*(i*i+0.4*i*i*i)) << "incorrect updated particle velocity y at " << i;
        EXPECT_DOUBLE_EQ(myMPM.p_vz(i), myMPM.g_cell_size()/2.*(i*i+0.6*i*i*i)) << "incorrect updated particle velocity z at " << i;
        EXPECT_DOUBLE_EQ(myMPM.p_x(i), myMPM.g_cell_size()/2.*(i+0.3*i*i*i)) << "incorrect updated particle position x at " << i;
        EXPECT_DOUBLE_EQ(myMPM.p_y(i), myMPM.g_cell_size()/2.*(i+0.6*i*i*i)) << "incorrect updated particle position y at " << i;
        EXPECT_DOUBLE_EQ(myMPM.p_z(i), myMPM.g_cell_size()/2.*(i+0.9*i*i*i)) << "incorrect updated particle position z at " << i;
        EXPECT_DOUBLE_EQ(myMPM.p_rho(i), 1000./(1.+i*0.24));
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}