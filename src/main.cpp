#include <grampm-kokkos.hpp>
#include <grampm.hpp>
#include <vector>

int main() {

    const double dcell = 0.002;

    std::vector<GraMPM::particle<double>> pv;

    const int xp {100}, yp {50}, zp {50};

    for (int i = 0; i < xp; ++i) {
        for (int j = 0; j < yp; ++j) {
            for (int k = 0; k < zp; ++k) {
                GraMPM::particle<double> p;
                p.x[0] = (i+0.5)*dcell/2.;
                p.x[1] = (j+0.5)*dcell/2.;
                p.x[2] = (k+0.5)*dcell/2.;
                pv.push_back(p);
            }
        }
    }

    GraMPM::Kokkos::MPM_system<double> myMPM(pv);
}