#include <grampm/accelerated/core.hpp>
#include <grampm/accelerated/kernels.hpp>
#include <grampm/accelerated/stressupdate.hpp>
#include <Kokkos_Core.hpp>
#include <grampm.hpp>
#include <array>
#include <vector>
#include <grampm/accelerated/integrators.hpp>
#include <iostream>

const int nbuffer = 2;

struct apply_boundary {
    int itimestep;
    double dt;
    const double ngridx, ngridy, ngridz;
    const Kokkos::View<double*[3]> data;

    apply_boundary(Kokkos::View<double*[3]> data_, double ngridx_, double ngridy_, double ngridz_)
        : data {data_} 
        , ngridx {ngridx_}
        , ngridy {ngridy_}
        , ngridz {ngridz_}
    {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, const int j, const int k) const {
        const int idx = i*ngridy*ngridz + j*ngridz + k;
        // fully-fixed
        if (k < nbuffer || i < nbuffer || i >= ngridx-nbuffer) {
            data(idx, 0) = 0.;
            data(idx, 1) = 0.;
            data(idx, 2) = 0.;
        
        // free-slip on north-south walls
        } else if (j < nbuffer || j >= ngridy-nbuffer) {
            data(idx, 1) = 0.;
        }
    }
};

using namespace GraMPM::accelerated;

typedef MPM_system<double, 
                   kernels::cubic_bspline<double>, 
                   functors::stress_update::drucker_prager_elastoplastic<double>, 
                   apply_boundary,
                   apply_boundary> MPM_type;

int main() {

    Kokkos::initialize();
    {
    // geometric properties
    const double dcell = 0.002;
    const int xp = 100, yp = 12, zp = 50;
    const std::array<double, 3> mingrid {-dcell, -dcell, -dcell}, maxgrid {0.299+dcell, 0.011+dcell, 0.049+dcell}, 
        gf {0., 0., -9.81};

    // material properties
    const double pi = std::acos(-1.);
    const double rho_ini = 1650., E = 0.86e6, v = 0.3, phi = pi/9., psi = 0., cohesion = 0.;

    std::vector<GraMPM::particle<double>> vp;

    for (int i = 0; i < xp; ++i) {
        for (int j = 0; j < yp; ++j) {
            for (int k = 0; k < zp; ++k) {
                GraMPM::particle<double> p;
                p.x[0] = (i+0.5)*dcell/2.;
                p.x[1] = (j+0.5)*dcell/2.;
                p.x[2] = (k+0.5)*dcell/2.;
                p.rho = rho_ini;
                p.mass = rho_ini*dcell*dcell*dcell/8.;
                vp.push_back(p);
            }
        }
    }

    MPM_type myMPM(vp, mingrid, maxgrid, dcell);

    std::cout << myMPM.g_size() << '\n';
    myMPM.body_force() = gf;
    myMPM.f_stress_update.set_DP_params(phi, psi, cohesion, E, v);
    
    const double K = E/(3.*(1.-2.*v)), G = E/(2.*(1.+v));
    const double c = std::sqrt((K+4./3.*G)/rho_ini);
    const double dt = dcell/c;

    myMPM.h2d();

    GraMPM::integrators::MUSL<double, MPM_type>(myMPM, dt, 500, 500, 500);
    }
    Kokkos::finalize();
}