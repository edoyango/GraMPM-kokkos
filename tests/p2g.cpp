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

// helper function to generate particles
std::vector<GraMPM::particle<double>> generate_particles() {

    std::vector<GraMPM::particle<double>> vp;

    for (int i = 0; i < 10; ++i) 
        for (int j = 0; j < 20; ++j) 
            for (int k = 0; k < 30; ++k) {
                vp.push_back(
                    GraMPM::particle<double>(0.1*(i+0.5), 0.1*(j+0.5), 0.1*(k+0.5), 
                                             -i, -j, -k,
                                             10.*(i+j+k), 100.*(i+j+k+3.),
                                             0., 0., 0., 0., 0., 0.)
                );
            }

    return vp;
}

TEST(p2g_mass, linear_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> mingrid {0., 0., 0.}, maxgrid {0.99, 1.99, 2.99};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::linear_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingrid, maxgrid, dcell);

    ASSERT_EQ(myMPM.g_ngridx(), 6)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 11) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 16) << "ngridz not calculated correctly";

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_mass();
    myMPM.d2h();

    // check total mass conservation
    // sum particles' mass
    double psum = 0., gsum = 0.;
    for (int i = 0; i < myMPM.p_size(); ++i)
        psum += myMPM.p_mass(i);
    // sum grid's mass
    for (int i = 0; i < myMPM.g_size(); ++i)
        gsum += myMPM.g_mass(i);

    ASSERT_DOUBLE_EQ(psum, gsum) << "sum of particles' mass not equal to sum of grid mass";

    // check a few nodal values
    EXPECT_DOUBLE_EQ(myMPM.g_mass(1, 1, 1), 360.) << "grid mass at i=1, j=1, k=1 not calculated correctly";
    EXPECT_DOUBLE_EQ(myMPM.g_mass(3, 4, 5), 1800.) << "grid mass at i=3, j=4, k=5 not calculated correctly";
    // using float equal due to roundoff error
    EXPECT_NEAR(myMPM.g_mass(5, 10, 15), 562.5, 2.e-12) << "grid mass at i=5, j=10, k=15 not calculated correctly";
}

TEST(p2g_mass, cubic_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> mingrid {-dcell, -dcell, -dcell}, maxgrid {0.99+dcell, 1.99+dcell, 2.99+dcell};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingrid, maxgrid, dcell);
    
    ASSERT_EQ(myMPM.g_ngridx(), 8)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 13) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 18) << "ngridz not calculated correctly";

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_mass();
    myMPM.d2h();

    // check total mass conservation
    // sum particles' mass
    double psum {0.}, gsum {0.};
    for (int i = 0; i < myMPM.p_size(); ++i)
        psum += myMPM.p_mass(i);
    // sum grid's mass
    for (int i = 0; i < myMPM.g_size(); ++i)
        gsum += myMPM.g_mass(i);

    // should be correct to 14 sigfigs
    EXPECT_NEAR(psum, gsum, 2.e-8) << "sum of particles' mass not equal to sum of grid mass";

    // check a few nodal values
    EXPECT_NEAR(myMPM.g_mass(2, 2, 2), 342.6422542996, 1.e-10); // should be correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_mass(4, 5, 6), 1800., 2.e-12); // should be correct to around 12 sigfigs
    EXPECT_NEAR(myMPM.g_mass(6, 11, 16), 556.09375, 1.e-6); // should be correct to around 14 sigfigs
}

TEST(p2g_momentum, linear_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> mingrid {0., 0., 0.}, maxgrid {0.99, 1.99, 2.99};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::linear_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingrid, maxgrid, dcell);

    ASSERT_EQ(myMPM.g_ngridx(), 6)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 11) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 16) << "ngridz not calculated correctly";

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_momentum();
    myMPM.d2h();

    // check momentum conservation
    // sum particles' momentum
    double psum[3] {0., 0., 0.}, gsum[3] {0., 0., 0.};
    for (int i = 0; i < myMPM.p_size(); ++i) {
        psum[0] += myMPM.p_mass(i)*myMPM.p_vx(i);
        psum[1] += myMPM.p_mass(i)*myMPM.p_vy(i);
        psum[2] += myMPM.p_mass(i)*myMPM.p_vz(i);
    }

    // sum grid's momentum
    for (int i = 0; i < myMPM.g_size(); ++i) {
        gsum[0] += myMPM.g_momentumx(i);
        gsum[1] += myMPM.g_momentumy(i);
        gsum[2] += myMPM.g_momentumz(i);
    }

    // test
    EXPECT_DOUBLE_EQ(psum[0], gsum[0]) << "sum of particles' momentum x not equal to sum of grid momentum x";
    EXPECT_DOUBLE_EQ(psum[1], gsum[1]) << "sum of particles' momentum y not equal to sum of grid momentum y";
    EXPECT_DOUBLE_EQ(psum[2], gsum[2]) << "sum of particles' momentum z not equal to sum of grid momentum z";

    // check a few nodal values
    EXPECT_DOUBLE_EQ(myMPM.g_momentumx(1, 1, 1), -600.);
    EXPECT_DOUBLE_EQ(myMPM.g_momentumx(3, 4, 5), -9960.);
    EXPECT_NEAR(myMPM.g_momentumx(5, 10, 15), -4923.75, 2.e-11); // correct to 14 sigfigs
    EXPECT_DOUBLE_EQ(myMPM.g_momentumy(1, 1, 1), -600.);
    EXPECT_DOUBLE_EQ(myMPM.g_momentumy(3, 4, 5), -13560.);
    EXPECT_NEAR(myMPM.g_momentumy(5, 10, 15), -10548.75, 5.e-11); // correct to 14 sigfigs
    EXPECT_DOUBLE_EQ(myMPM.g_momentumz(1, 1, 1), -600.);
    EXPECT_DOUBLE_EQ(myMPM.g_momentumz(3, 4, 5), -17160);
    EXPECT_NEAR(myMPM.g_momentumz(5, 10, 15), -16173.75, 6.e-11); // correct to 14 sigfigs
}

TEST(p2g_momentum, cubic_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> mingrid {-dcell, -dcell, -dcell}, maxgrid {0.99+dcell, 1.99+dcell, 2.99+dcell};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingrid, maxgrid, dcell);

    ASSERT_EQ(myMPM.g_ngridx(), 8)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 13) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 18) << "ngridz not calculated correctly";

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_momentum();
    myMPM.d2h();

    // check momentum conservation
    // sum particles' momentum
    double psum[3] {0., 0., 0.}, gsum[3] {0., 0., 0.};
    for (int i = 0; i < myMPM.p_size(); ++i) {
        psum[0] += myMPM.p_mass(i)*myMPM.p_vx(i);
        psum[1] += myMPM.p_mass(i)*myMPM.p_vy(i);
        psum[2] += myMPM.p_mass(i)*myMPM.p_vz(i);
    }

    // sum grid's momentum
    for (int i = 0; i < myMPM.g_size(); ++i) {
        gsum[0] += myMPM.g_momentumx(i);
        gsum[1] += myMPM.g_momentumy(i);
        gsum[2] += myMPM.g_momentumz(i);
    }

    // test
    EXPECT_NEAR(psum[0], gsum[0], 5.e-8) << "sum of particles' momentum x not equal to sum of grid momentum x";  // equal to 7 sigfigs
    EXPECT_NEAR(psum[1], gsum[1], 5.e-8) << "sum of particles' momentum y not equal to sum of grid momentum y";  // equal to 8 sigfigs
    EXPECT_NEAR(psum[2], gsum[2], 2.e-7) << "sum of particles' momentum z not equal to sum of grid momentum z";  // equal to 9 sigfigs

    // check a few nodal values
    EXPECT_NEAR(myMPM.g_momentumx(2, 2, 2), -627.705941, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_momentumx(4, 5, 6), -10006.666667, 5.e-7); // correct to 11 sigfigs
    EXPECT_NEAR(myMPM.g_momentumx(6, 11, 16), -4751.120334, 5.e-7); // correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_momentumy(2, 2, 2), -627.705941, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_momentumy(4, 5, 6), -13606.666667, 5.e-7); // correct to 11 sigfigs
    EXPECT_NEAR(myMPM.g_momentumy(6, 11, 16), -10312.057834, 5.e-7); // correct to 11 sigfigs
    EXPECT_NEAR(myMPM.g_momentumz(2, 2, 2), -627.705941, 5.e-7);  // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_momentumz(4, 5, 6), -17206.666667, 5.e-7);  // correct to 11 sigfigs
    EXPECT_NEAR(myMPM.g_momentumz(6, 11, 16), -15872.995334, 5.e-7); // correct to 11 sigfigs
}

TEST(p2g_force, linear_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> bf {1., 2., 3.}, mingrid {0., 0., 0.}, maxgrid {0.99, 1.99, 2.99};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::linear_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingrid, maxgrid, dcell);

    myMPM.body_force() = bf;

    ASSERT_EQ(myMPM.g_ngridx(), 6);
    ASSERT_EQ(myMPM.g_ngridy(), 11);
    ASSERT_EQ(myMPM.g_ngridz(), 16);
    ASSERT_DOUBLE_EQ(myMPM.body_forcex(), 1.);
    ASSERT_DOUBLE_EQ(myMPM.body_forcey(), 2.);
    ASSERT_DOUBLE_EQ(myMPM.body_forcez(), 3.);    

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_force();
    myMPM.d2h();

    // check conservation
    double psum[3] {0., 0., 0.}, gsum[3] {0., 0., 0.};
    for (int i = 0; i < myMPM.p_size(); ++i) {
        psum[0] += myMPM.p_mass(i)*myMPM.body_forcex();
        psum[1] += myMPM.p_mass(i)*myMPM.body_forcey();
        psum[2] += myMPM.p_mass(i)*myMPM.body_forcez();
    }
    for (int i = 0; i < myMPM.g_size(); ++i) {
        gsum[0] += myMPM.g_forcex(i);
        gsum[1] += myMPM.g_forcey(i);
        gsum[2] += myMPM.g_forcez(i);
    }

    EXPECT_DOUBLE_EQ(psum[0], gsum[0]) << "sum of particles' force x not equal to sum of grid force x";
    EXPECT_DOUBLE_EQ(psum[1], gsum[1]) << "sum of particles' force y not equal to sum of grid force y";
    EXPECT_DOUBLE_EQ(psum[2], gsum[2]) << "sum of particles' force z not equal to sum of grid force z";

    // check a few nodal values
    EXPECT_DOUBLE_EQ(myMPM.g_forcex(1, 1, 1), 360.);
    EXPECT_DOUBLE_EQ(myMPM.g_forcex(3, 4, 5), 1800.);
    EXPECT_NEAR(myMPM.g_forcex(5, 10, 15), 562.5, 1.e-11); // correct to 13 sigfigs
    EXPECT_DOUBLE_EQ(myMPM.g_forcey(1, 1, 1), 720.);
    EXPECT_DOUBLE_EQ(myMPM.g_forcey(3, 4, 5), 3600.);
    EXPECT_NEAR(myMPM.g_forcey(5, 10, 15), 1125., 1.e-11); // correct to 14 sigfigs
    EXPECT_DOUBLE_EQ(myMPM.g_forcez(1, 1, 1), 1080.);
    EXPECT_DOUBLE_EQ(myMPM.g_forcez(3, 4, 5), 5400.);
    EXPECT_NEAR(myMPM.g_forcez(5, 10, 15), 1687.5, 1.e-11); // correct to 14 sigfigs

    // try with non-zero stresses
    for (int i = 0; i < myMPM.p_size(); ++i) {
        myMPM.p_sigmaxx(i) = myMPM.p_x(i);
        myMPM.p_sigmayy(i) = myMPM.p_y(i);
        myMPM.p_sigmazz(i) = myMPM.p_z(i);
        myMPM.p_sigmaxy(i) = myMPM.p_x(i)-myMPM.p_y(i);
        myMPM.p_sigmaxz(i) = myMPM.p_x(i)-myMPM.p_z(i);
        myMPM.p_sigmayz(i) = myMPM.p_y(i)-myMPM.p_z(i);
    }

    myMPM.h2d();
    myMPM.map_p2g_force();
    myMPM.d2h();

    // check a few nodal values
    EXPECT_NEAR(myMPM.g_forcex(1, 1, 1), 359.6170004735, 5.e-11); // correct to 11 sigfigs
    EXPECT_NEAR(myMPM.g_forcex(3, 4, 5), 1799.2941517835, 5.e-11); // correct to 13 sigfigs
    EXPECT_NEAR(myMPM.g_forcex(5, 10, 15), 564.4457328379, 5.e-11); // correct to 13 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(1, 1, 1), 720.5551538826, 5.e-11); // correct to 13 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(3, 4, 5), 3600.7203137483, 5.e-11); // correct to 14 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(5, 10, 15), 1125.0948927821, 5.e-11); // correct to 14 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(1, 1, 1), 1081.4933072917, 5.e-11); // correct to 14 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(3, 4, 5), 5402.1315681200, 5.e-11); // correct to 14 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(5, 10, 15), 1687.6423394507, 5.e-11); // correct to 14 sigfigs

}

TEST(p2g_force, cubic_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> bf {1., 2., 3.}, mingrid {-dcell, -dcell, -dcell}, maxgrid {0.99+dcell, 1.99+dcell, 2.99+dcell};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::cubic_bspline<double>, functors::stress_update::hookes_law<double>> 
        myMPM(vp, mingrid, maxgrid, dcell);

    myMPM.body_force() = bf;

    ASSERT_EQ(myMPM.g_ngridx(), 8)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 13) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 18) << "ngridz not calculated correctly";
    ASSERT_DOUBLE_EQ(myMPM.body_forcex(), 1.);
    ASSERT_DOUBLE_EQ(myMPM.body_forcey(), 2.);
    ASSERT_DOUBLE_EQ(myMPM.body_forcez(), 3.);    

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_force();
    myMPM.d2h();

    // check conservation
    double psum[3] {0., 0., 0.}, gsum[3] {0., 0., 0.};
    for (int i = 0; i < myMPM.p_size(); ++i) {
        psum[0] += myMPM.p_mass(i)*myMPM.body_forcex();
        psum[1] += myMPM.p_mass(i)*myMPM.body_forcey();
        psum[2] += myMPM.p_mass(i)*myMPM.body_forcez();
    }
    for (int i = 0; i < myMPM.g_size(); ++i) {
        gsum[0] += myMPM.g_forcex(i);
        gsum[1] += myMPM.g_forcey(i);
        gsum[2] += myMPM.g_forcez(i);
    }

    EXPECT_FLOAT_EQ(psum[0], gsum[0]) << "sum of particles' force x not equal to sum of grid force x"; // correct to 12 sigfigs
    EXPECT_FLOAT_EQ(psum[1], gsum[1]) << "sum of particles' force y not equal to sum of grid force y"; // correct to 12 sigfigs
    EXPECT_DOUBLE_EQ(psum[2], gsum[2]) << "sum of particles' force z not equal to sum of grid force z";

    // check a few nodal values
    EXPECT_NEAR(myMPM.g_forcex(2, 2, 2), 342.642254, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcex(4, 5, 6), 1800., 1.e-11); // correct to 15 sigfigs
    EXPECT_NEAR(myMPM.g_forcex(6, 11, 16), 556.093750, 1.e-10); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(2, 2, 2), 685.284509, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(4, 5, 6), 3600., 1.e-11); // correct to 15 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(6, 11, 16), 1112.187500, 1.e-9); // correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(2, 2, 2), 1027.926763, 5.e-7); // correc to t10 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(4, 5, 6), 5400., 1.e-10); // correct to 15 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(6, 11, 16), 1668.281250, 1.e-9); // correct to 10 sigfigs

    // try with non-zero stresses
    for (int i = 0; i < myMPM.p_size(); ++i) {
        myMPM.p_sigmaxx(i) = myMPM.p_x(i);
        myMPM.p_sigmayy(i) = myMPM.p_y(i);
        myMPM.p_sigmazz(i) = myMPM.p_z(i);
        myMPM.p_sigmaxy(i) = myMPM.p_x(i)-myMPM.p_y(i);
        myMPM.p_sigmaxz(i) = myMPM.p_x(i)-myMPM.p_z(i);
        myMPM.p_sigmayz(i) = myMPM.p_y(i)-myMPM.p_z(i);
    }

    myMPM.h2d();
    myMPM.map_p2g_force();
    myMPM.d2h();

    // check a few nodal values
    EXPECT_NEAR(myMPM.g_forcex(2, 2, 2), 342.415998, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcex(4, 5, 6), 1799.294307, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcex(6, 11, 16), 557.428540, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(2, 2, 2), 685.779305, 5.e-7); // correct to 9 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(4, 5, 6), 3600.719937, 5.e-7); // correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_forcey(6, 11, 16), 1112.282300, 5.e-7); // correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(2, 2, 2), 1029.142612, 5.e-7); // correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(4, 5, 6), 5402.130522, 5.e-7); // correct to 10 sigfigs
    EXPECT_NEAR(myMPM.g_forcez(6, 11, 16), 1668.440054, 5.e-7); // correct to 10 sigfigs

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}