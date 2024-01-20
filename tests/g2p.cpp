#include <grampm.hpp>
#include <grampm-kokkos.hpp>
#include <grampm-kokkos-kernels.hpp>
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

TEST(g2p_a, linear_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> mingrid {0., 0., 0.}, maxgrid {0.99, 1.99, 2.99};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::linear_bspline<double>> myMPM(vp, mingrid, maxgrid, dcell);

    ASSERT_EQ(myMPM.g_ngridx(), 6)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 11) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 16) << "ngridz not calculated correctly";

    // setup grid
    for (size_t i = 0 ; i < myMPM.g_ngridx(); ++i) 
        for (size_t j = 0; j < myMPM.g_ngridy(); ++j)
            for (size_t k = 0; k < myMPM.g_ngridz(); ++k) {
                myMPM.g_momentumx(i, j, k) = 0.1*(i+j+k);
                myMPM.g_momentumy(i, j, k) = 0.2*(i+j+k);
                myMPM.g_momentumz(i, j, k) = 0.3*(i+j+k);
                myMPM.g_forcex(i, j, k) = 0.4*(i+j+k);
                myMPM.g_forcey(i, j, k) = 0.5*(i+j+k);
                myMPM.g_forcez(i, j, k) = 0.6*(i+j+k);
            }

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_mass();
    myMPM.map_g2p_acceleration();
    myMPM.d2h();

    // check conservation
    // sum particles' force (m*a) and momentum (dxdt*m)
    double psum[6] {0., 0., 0., 0., 0., 0.}, gsum[6] {0., 0., 0., 0., 0., 0.};
    for (size_t i = 0; i < myMPM.p_size(); ++i) {
        psum[0] += myMPM.p_ax(i)*myMPM.p_mass(i);
        psum[1] += myMPM.p_ay(i)*myMPM.p_mass(i);
        psum[2] += myMPM.p_az(i)*myMPM.p_mass(i);
        psum[3] += myMPM.p_dxdt(i)*myMPM.p_mass(i);
        psum[4] += myMPM.p_dydt(i)*myMPM.p_mass(i);
        psum[5] += myMPM.p_dzdt(i)*myMPM.p_mass(i);
    }
    for (size_t i = 0; i < myMPM.g_size(); ++i) {
        gsum[0] += myMPM.g_forcex(i);
        gsum[1] += myMPM.g_forcey(i);
        gsum[2] += myMPM.g_forcez(i);
        gsum[3] += myMPM.g_momentumx(i);
        gsum[4] += myMPM.g_momentumy(i);
        gsum[5] += myMPM.g_momentumz(i);
    }

    // test conservation
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(psum[i], gsum[i]); // correct to 14 sigfigs
}

TEST(g2p_a, cubic_bspline) {
    const double dcell = 0.2;
    std::array<double, 3> mingrid {-dcell, -dcell, -dcell}, maxgrid {0.99+dcell, 1.99+dcell, 2.99+dcell};

    std::vector<GraMPM::particle<double>> vp = generate_particles();

    MPM_system<double, kernels::cubic_bspline<double>> myMPM(vp, mingrid, maxgrid, dcell);

    ASSERT_EQ(myMPM.g_ngridx(), 8)  << "ngridx not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridy(), 13) << "ngridy not calculated correctly";
    ASSERT_EQ(myMPM.g_ngridz(), 18) << "ngridz not calculated correctly";

    // setup grid
    for (size_t i = 0 ; i < myMPM.g_ngridx(); ++i) 
        for (size_t j = 0; j < myMPM.g_ngridy(); ++j)
            for (size_t k = 0; k < myMPM.g_ngridz(); ++k) {
                myMPM.g_momentumx(i, j, k) = 0.1*(i+j+k);
                myMPM.g_momentumy(i, j, k) = 0.2*(i+j+k);
                myMPM.g_momentumz(i, j, k) = 0.3*(i+j+k);
                myMPM.g_forcex(i, j, k) = 0.4*(i+j+k);
                myMPM.g_forcey(i, j, k) = 0.5*(i+j+k);
                myMPM.g_forcez(i, j, k) = 0.6*(i+j+k);
            }

    myMPM.h2d();
    myMPM.update_particle_to_cell_map();
    myMPM.find_neighbour_nodes();
    myMPM.map_p2g_mass();
    myMPM.map_g2p_acceleration();
    myMPM.d2h();

    // check conservation
    // sum particles' force (m*a) and momentum (dxdt*m)
    double psum[6] {0., 0., 0., 0., 0., 0.}, gsum[6] {0., 0., 0., 0., 0., 0.};
    for (size_t i = 0; i < myMPM.p_size(); ++i) {
        psum[0] += myMPM.p_ax(i)*myMPM.p_mass(i);
        psum[1] += myMPM.p_ay(i)*myMPM.p_mass(i);
        psum[2] += myMPM.p_az(i)*myMPM.p_mass(i);
        psum[3] += myMPM.p_dxdt(i)*myMPM.p_mass(i);
        psum[4] += myMPM.p_dydt(i)*myMPM.p_mass(i);
        psum[5] += myMPM.p_dzdt(i)*myMPM.p_mass(i);
    }
    for (size_t i = 0; i < myMPM.g_size(); ++i) {
        gsum[0] += myMPM.g_forcex(i);
        gsum[1] += myMPM.g_forcey(i);
        gsum[2] += myMPM.g_forcez(i);
        gsum[3] += myMPM.g_momentumx(i);
        gsum[4] += myMPM.g_momentumy(i);
        gsum[5] += myMPM.g_momentumz(i);
    }

    // test conservation
    for (int i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(psum[i], gsum[i]); // correct to 15 sigfigs
}

// TEST_CASE("Calculate particles' strain/spin rates (linear bspline)") {
    
//     const double dcell = 0.2;
//     // GraMPM::grid<double> g(0., 0., 0., 0.99, 1.99, 2.99, dcell);
//     std::array<double, 3> bf {0.}, mingrid {0., 0., 0.}, maxgrid {0.99, 1.99, 2.99};

//     GraMPM::kernel_linear_bspline<double> knl(dcell);

//     GraMPM::MPM_system<double> p(bf, knl, mingrid, maxgrid, dcell);

//     CHECK(p.g_ngridx()==6);
//     CHECK(p.g_ngridy()==11);
//     CHECK(p.g_ngridz()==16);

//     generate_particles(p);

//     // setup grid
//     for (size_t i = 0; i < p.g_ngridx(); ++i)
//         for (size_t j = 0; j < p.g_ngridy(); ++j) 
//             for (size_t k = 0; k < p.g_ngridz(); ++k) {
//                 const double x = i*0.2 + p.g_mingridx();
//                 const double y = j*0.2 + p.g_mingridy();
//                 const double z = k*0.2 + p.g_mingridz();
//                 p.g_momentumx(i, j, k) = 0.1*(x-y-z);
//                 p.g_momentumy(i, j, k) = 0.2*(y-x-z);
//                 p.g_momentumz(i, j, k) = 0.3*(z-x-y);
//                 p.g_mass(i, j, k) = 1.;
//             }
    
//     p.map_particles_to_grid();
//     p.map_g2p_strainrate();

//     REQUIRE(std::round(p.p_strainratexx(0)*100.)==10.);
//     REQUIRE(std::round(p.p_strainrateyy(0)*100.)==20.);
//     REQUIRE(std::round(p.p_strainratezz(0)*100.)==30.);
//     REQUIRE(std::round(p.p_strainratexy(0)*100.)==-15.);
//     REQUIRE(std::round(p.p_strainratexz(0)*100.)==-20.);
//     REQUIRE(std::round(p.p_strainrateyz(0)*100.)==-25.);
//     REQUIRE(std::round(p.p_spinratexy(0)*100.)==5.);
//     REQUIRE(std::round(p.p_spinratexz(0)*100.)==10.);
//     REQUIRE(std::round(p.p_spinrateyz(0)*100.)==5.);

//     REQUIRE(std::round(p.p_strainratexx(p.p_size()-1)*100.)==10.);
//     REQUIRE(std::round(p.p_strainrateyy(p.p_size()-1)*100.)==20.);
//     REQUIRE(std::round(p.p_strainratezz(p.p_size()-1)*100.)==30.);
//     REQUIRE(std::round(p.p_strainratexy(p.p_size()-1)*100.)==-15.);
//     REQUIRE(std::round(p.p_strainratexz(p.p_size()-1)*100.)==-20.);
//     REQUIRE(std::round(p.p_strainrateyz(p.p_size()-1)*100.)==-25.);
//     REQUIRE(std::round(p.p_spinratexy(p.p_size()-1)*100.)==5.);
//     REQUIRE(std::round(p.p_spinratexz(p.p_size()-1)*100.)==10.);
//     REQUIRE(std::round(p.p_spinrateyz(p.p_size()-1)*100.)==5.);
// }

// TEST_CASE("Calculate particles' strain/spin rates (cubic bspline)") {
    
//     const double dcell = 0.2;
//     std::array<double, 3> bf {0.}, mingrid {-0.2, -0.2, -0.2}, maxgrid {1.19, 2.19, 3.19};

//     GraMPM::kernel_cubic_bspline<double> knl(dcell);

//     GraMPM::MPM_system<double> p(bf, knl, mingrid, maxgrid, dcell);

//     CHECK(p.g_ngridx()==8);
//     CHECK(p.g_ngridy()==13);
//     CHECK(p.g_ngridz()==18);

//     generate_particles(p);

//     // setup grid
//     for (size_t i = 0; i < p.g_ngridx(); ++i)
//         for (size_t j = 0; j < p.g_ngridy(); ++j) 
//             for (size_t k = 0; k < p.g_ngridz(); ++k) {
//                 const double x = i*0.2 + p.g_mingridx();
//                 const double y = j*0.2 + p.g_mingridy();
//                 const double z = k*0.2 + p.g_mingridz();
//                 p.g_momentumx(i, j, k) = 0.1*(x-y-z);
//                 p.g_momentumy(i, j, k) = 0.2*(y-x-z);
//                 p.g_momentumz(i, j, k) = 0.3*(z-x-y);
//                 p.g_mass(i, j, k) = 1.;
//             }
    
//     p.map_particles_to_grid();
//     p.map_g2p_strainrate();

//     REQUIRE(std::round(p.p_strainratexx(0)*100.)==10.);
//     REQUIRE(std::round(p.p_strainrateyy(0)*100.)==20.);
//     REQUIRE(std::round(p.p_strainratezz(0)*100.)==30.);
//     REQUIRE(std::round(p.p_strainratexy(0)*100.)==-15.);
//     REQUIRE(std::round(p.p_strainratexz(0)*100.)==-20.);
//     REQUIRE(std::round(p.p_strainrateyz(0)*100.)==-25.);
//     REQUIRE(std::round(p.p_spinratexy(0)*100.)==5.);
//     REQUIRE(std::round(p.p_spinratexz(0)*100.)==10.);
//     REQUIRE(std::round(p.p_spinrateyz(0)*100.)==5.);

//     REQUIRE(std::round(p.p_strainratexx(p.p_size()-1)*100.)==10.);
//     REQUIRE(std::round(p.p_strainrateyy(p.p_size()-1)*100.)==20.);
//     REQUIRE(std::round(p.p_strainratezz(p.p_size()-1)*100.)==30.);
//     REQUIRE(std::round(p.p_strainratexy(p.p_size()-1)*100.)==-15.);
//     REQUIRE(std::round(p.p_strainratexz(p.p_size()-1)*100.)==-20.);
//     REQUIRE(std::round(p.p_strainrateyz(p.p_size()-1)*100.)==-25.);
//     REQUIRE(std::round(p.p_spinratexy(p.p_size()-1)*100.)==5.);
//     REQUIRE(std::round(p.p_spinratexz(p.p_size()-1)*100.)==10.);
//     REQUIRE(std::round(p.p_spinrateyz(p.p_size()-1)*100.)==5.);
// }

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    Kokkos::initialize(argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}