#include <grampm_kernels.hpp>
#include <cmath>
#include <grampm-kokkos.hpp>
#include <gtest/gtest.h>
#include <array>

using namespace GraMPM::accelerated;

std::array<double, 3> mingrid {0., 0., 0.}, maxgrid {1., 1., 1.};
double cell_size = 0.1;

TEST(kernels, linear) {

    MPM_system<double, kernels::linear_bspline<double>> myMPM(0, mingrid, maxgrid, cell_size);

    // check attributes are set correctly
    ASSERT_EQ(myMPM.knl.radius, 1.) << "kernel radius not set correctly";
    ASSERT_EQ(myMPM.knl.dcell, 0.1) << "kernel cell size not set correctly";

    // check kernel function returns correct values
    // below round thing is because C++ is stupid

    double w, dwdx, dwdy, dwdz;
    // check somewhere within 1 cell (R+)
    myMPM.knl(0.05, 0.06, 0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(std::round(1e2*w), 6.) << "(R+) kernel value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdx*10.), -12.) << "(R+) kernel gradient x value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdy*10.), -15.) << "(R+) kernel gradient y value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdz*1.), -2.) << "(R+) kernel gradient z value not calculated correctly";

    // check somewhere within 1 cell (R-)
    myMPM.knl(-0.05, -0.06, -0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(std::round(100*w), 6.) << "(R-) kernel value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdx*10.), 12.) << "(R-) kernel gradient x value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdy*10.), 15.) << "(R-) kernel gradient y value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdz*1.), 2.) << "(R-) kernel gradient z value not calculated correctly";

    // check somewhere on edge (x)
    myMPM.knl(0.1, 0.06, 0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel edge value not calculated correctly";
    
    // check somewhere on edge (y)
    myMPM.knl(0.05, 0.1, 0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel edge value not calculated correctly";

    // check somewhere on edge (z)
    myMPM.knl(0.05, 0.06, 0.1, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel edge value not calculated correctly";

    // check somewhere on edge (x, y, z)
    myMPM.knl(0.1, 0.1, 0.1, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(dwdx, 0.) << "kernel gradient x edge value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdy, 0.) << "kernel gradient y edge value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdz, 0.) << "kernel gradient z edge value not calculated correctly";
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel out-of-bounds value not calculated correctly";;

    // check centre
    myMPM.knl(0., 0., 0., w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 1.) << "kernel peak value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdx, 0.) << "kernel gradient x centre value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdy, 0.) << "kernel gradient y centre value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdz, 0.) << "kernel gradient z centre value not calculated correctly";
}

TEST(kernels, cubic) {

    MPM_system<double, kernels::cubic_bspline<double>> myMPM(0, mingrid, maxgrid, cell_size);

    // check attributes are set correctly
    ASSERT_EQ(myMPM.knl.radius, 2.) << "kernel radius not set correctly";
    ASSERT_EQ(myMPM.knl.dcell, 0.1) << "kernel cell size not set correctly";


    // check kernel function returns correct values
    // below round thing is because C++ is stupid

    double w, dwdx, dwdy, dwdz;
    // check somewhere within 1 cell (R+)
    myMPM.knl(0.05, 0.06, 0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(std::round(1e7*w), std::round(1e7*0.0691787824)) << "(R+) kernel value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdx*1e7), -9023319) << "(R+) kernel gradient x value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdy*1e7), -11010771) << "(R+) kernel gradient y value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdz*1e7), -13213181) << "(R+) kernel gradient z value not calculated correctly";

    // check somewhere within 1 cell (R-)
    myMPM.knl(-0.05, -0.06, -0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(std::round(1e7*w), std::round(1e7*0.0691787824)) << "(R-) kernel value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdx*1e7), 9023319) << "(R-) kernel gradient x value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdy*1e7), 11010771) << "(R-) kernel gradient y value not calculated correctly";
    ASSERT_DOUBLE_EQ(std::round(dwdz*1e7), 13213181) << "(R-) kernel gradient z value not calculated correctly";

    // check somewhere within 2 cells
    myMPM.knl(0.15, 0.16, 0.17, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(std::round(1e6*w), 1.) << "kernel value not calculated correctly";

    // check somewhere on edge (x)
    myMPM.knl(0.2, 0.1, 0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel edge value not calculated correctly";
    
    // check somewhere on edge (y)
    myMPM.knl(0.05, 0.2, 0.07, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel edge value not calculated correctly";

    // check somewhere on edge (z)
    myMPM.knl(0.05, 0.06, 0.2, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel edge value not calculated correctly";

    // check somewhere on edge (x, y, z)
    myMPM.knl(0.2, 0.2, 0.2, w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(dwdx, 0.) << "kernel gradient x edge value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdy, 0.) << "kernel gradient y edge value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdz, 0.) << "kernel gradient z edge value not calculated correctly";
    ASSERT_DOUBLE_EQ(w, 0.) << "kernel out-of-bounds value not calculated correctly";;

    // check centre
    myMPM.knl(0., 0., 0., w, dwdx, dwdy, dwdz);
    ASSERT_DOUBLE_EQ(w, 8./27.) << "kernel peak value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdx, 0.) << "kernel gradient x centre value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdy, 0.) << "kernel gradient y centre value not calculated correctly";
    ASSERT_DOUBLE_EQ(dwdz, 0.) << "kernel gradient z centre value not calculated correctly";
}

int main(int argc, char **argv) {
    Kokkos::initialize(argc, argv);
    testing::InitGoogleTest(&argc, argv);
    int success = RUN_ALL_TESTS();
    Kokkos::finalize();
    return success;
}