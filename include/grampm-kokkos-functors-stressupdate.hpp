#ifndef GRAMPM_KOKKOS_FUNCTORS_STRESSUPDATE
#define GRAMPM_KOKKOS_FUNCTORS_STRESSUPDATE

#include <Kokkos_Core.hpp>

namespace GraMPM {
    namespace accelerated {
        namespace functors {
            namespace stress_update {

                template<typename F>
                struct hookes_law {
                    double E, v, dt;
                    const Kokkos::View<F*[6]> p_sigma, p_strainrate;
                    const Kokkos::View<F*[3]> p_spinrate;
                    hookes_law(Kokkos::View<F*[6]> p_sigma_, Kokkos::View<F*[6]> p_strainrate_, 
                        Kokkos::View<F*[3]> p_spinrate_)
                        : p_sigma {p_sigma_}
                        , p_strainrate {p_strainrate_}
                        , p_spinrate {p_spinrate_}
                    {}
                    KOKKOS_INLINE_FUNCTION
                    void operator()(const int i) const {
                        const F D0 {E/((1.+v)*(1.-2.*v))};
                        // elastic stiffness matrix
                        F dsigmaxx = D0*((1.-v)*p_strainrate(i, 0) + v*p_strainrate(i, 1) + v*p_strainrate(i, 2));
                        F dsigmayy = D0*(v*p_strainrate(i, 0) + (1.-v)*p_strainrate(i, 1) + v*p_strainrate(i, 2));
                        F dsigmazz = D0*(v*p_strainrate(i, 0) + v*p_strainrate(i, 1) + (1.-v)*p_strainrate(i, 2));
                        F dsigmaxy = D0*p_strainrate(i, 3)*(1.-2.*v);
                        F dsigmaxz = D0*p_strainrate(i, 4)*(1.-2.*v);
                        F dsigmayz = D0*p_strainrate(i, 5)*(1.-2.*v);

                        // jaumann stress rate
                        dsigmaxx -= 2.*(p_spinrate(i, 0)*p_sigma(i, 3) + p_spinrate(i, 1)*p_sigma(i, 4));
                        dsigmayy -= 2.*(-p_spinrate(i, 0)*p_sigma(i, 3) + p_spinrate(i, 2)*p_sigma(i, 5));
                        dsigmazz += 2.*(p_spinrate(i, 1)*p_sigma(i, 4) + p_spinrate(i, 2)*p_sigma(i, 5));
                        dsigmaxy += p_sigma(i, 0)*p_spinrate(i, 0) - p_sigma(i, 4)*p_spinrate(i, 2) -
                            p_spinrate(i, 0)*p_sigma(i, 1) - p_spinrate(i, 1)*p_sigma(i, 5);
                        dsigmaxz += p_sigma(i, 0)*p_spinrate(i, 1) + p_sigma(i, 3)*p_spinrate(i, 2) -
                            p_spinrate(i, 0)*p_sigma(i, 5) - p_spinrate(i, 1)*p_sigma(i, 2);
                        dsigmayz += p_sigma(i, 3)*p_spinrate(i, 1) + p_sigma(i, 1)*p_spinrate(i, 2) +
                            p_spinrate(i, 0)*p_sigma(i, 4) - p_spinrate(i, 2)*p_sigma(i, 2);
                        
                        // update stress state
                        p_sigma(i, 0) += dt*dsigmaxx;
                        p_sigma(i, 1) += dt*dsigmayy;
                        p_sigma(i, 2) += dt*dsigmazz;
                        p_sigma(i, 3) += dt*dsigmaxy;
                        p_sigma(i, 4) += dt*dsigmaxz;
                        p_sigma(i, 5) += dt*dsigmayz;
                    }
                };
            }
        }
    }
}

#endif