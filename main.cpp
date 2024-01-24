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

using ftype = double;

template<typename F>
struct apply_boundary {
    int itimestep;
    F dt;
    const int ngridx, ngridy, ngridz;
    const Kokkos::View<F*[3]> data;

    apply_boundary(Kokkos::View<F*[3]> data_, int ngridx_, int ngridy_, int ngridz_)
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
            data(idx, 0) = F(0.);
            data(idx, 1) = F(0.);
            data(idx, 2) = F(0.);
        
        // free-slip on north-south walls
        } else if (j < nbuffer || j >= ngridy-nbuffer) {
            data(idx, 1) = F(0.);
        }
    }
};

using namespace GraMPM::accelerated;

typedef MPM_system<ftype, 
                   kernels::cubic_bspline<ftype>, 
                   functors::stress_update::drucker_prager_elastoplastic<ftype>, 
                   apply_boundary<ftype>,
                   apply_boundary<ftype>> MPM_type;

int main() {

    Kokkos::initialize();
    {
    // geometric properties
    const ftype dcell = ftype(0.002);
    const int xp = 100, yp = 12, zp = 50;
    const std::array<ftype, 3> mingrid {-dcell, -dcell, -dcell}, maxgrid {ftype(0.299)+dcell, ftype(0.011)+dcell, ftype(0.049)+dcell}, 
        gf {ftype(0.), ftype(0.), ftype(-9.81)};

    // material properties
    const ftype pi = std::acos(-1.);
    const ftype rho_ini = ftype(1650.), E = ftype(0.86e6), v = ftype(0.3), phi = ftype(pi/9.), psi = ftype(0.), cohesion = ftype(0.);

    std::vector<GraMPM::particle<ftype>> vp;

    for (int i = 0; i < xp; ++i) {
        for (int j = 0; j < yp; ++j) {
            for (int k = 0; k < zp; ++k) {
                GraMPM::particle<ftype> p;
                p.x[0] = (i+ftype(0.5))*dcell/ftype(2.);
                p.x[1] = (j+ftype(0.5))*dcell/ftype(2.);
                p.x[2] = (k+ftype(0.5))*dcell/ftype(2.);
                p.rho = rho_ini;
                p.mass = rho_ini*dcell*dcell*dcell/ftype(8.);
                vp.push_back(p);
            }
        }
    }

    MPM_type myMPM(vp, mingrid, maxgrid, dcell);

    myMPM.body_force() = gf;
    myMPM.f_stress_update.set_DP_params(phi, psi, cohesion, E, v);
    
    const ftype K = E/(ftype(3.)*(ftype(1.)-ftype(2.)*v)), G = E/(ftype(2.)*(ftype(1.)+v));
    const ftype c = std::sqrt((K+ftype(4.)/ftype(3.)*G)/rho_ini);
    const ftype dt = dcell/c;

    myMPM.h2d();

    GraMPM::integrators::MUSL<ftype, MPM_type>(myMPM, dt, 500, 500, 501);
    }
    Kokkos::finalize();
}