#ifndef GRAMPM_KOKKOS_FUNCTORS_STRESSUPDATE
#define GRAMPM_KOKKOS_FUNCTORS_STRESSUPDATE

#include <Kokkos_Core.hpp>

namespace GraMPM {
    namespace accelerated {
        namespace functors {
            namespace stress_update {

                template<typename F>
                struct hookes_law {

                    F E, v, D0, dt;
                    const Kokkos::View<F*[6]> p_sigma, p_strainrate;
                    const Kokkos::View<F*[3]> p_spinrate;

                    hookes_law(Kokkos::View<F*[6]> p_sigma_, Kokkos::View<F*[6]> p_strainrate_, 
                        Kokkos::View<F*[3]> p_spinrate_)
                        : p_sigma {p_sigma_}
                        , p_strainrate {p_strainrate_}
                        , p_spinrate {p_spinrate_}
                    {}

                    void set_E_v(F E_, F v_) {
                        E = E_;
                        v = v_;
                        D0 = E/((1.+v)*(1.-2.*v));
                    }

                    KOKKOS_INLINE_FUNCTION
                    void operator()(const int i) const {

                        // elastic stiffness matrix
                        F dsigmaxx = D0*((F(1.)-v)*p_strainrate(i, 0) + v*p_strainrate(i, 1) + v*p_strainrate(i, 2));
                        F dsigmayy = D0*(v*p_strainrate(i, 0) + (F(1.)-v)*p_strainrate(i, 1) + v*p_strainrate(i, 2));
                        F dsigmazz = D0*(v*p_strainrate(i, 0) + v*p_strainrate(i, 1) + (F(1.)-v)*p_strainrate(i, 2));
                        F dsigmaxy = D0*p_strainrate(i, 3)*(F(1.)-F(2.)*v);
                        F dsigmaxz = D0*p_strainrate(i, 4)*(F(1.)-F(2.)*v);
                        F dsigmayz = D0*p_strainrate(i, 5)*(F(1.)-F(2.)*v);

                        // jaumann stress rate
                        dsigmaxx -= F(2.)*(p_spinrate(i, 0)*p_sigma(i, 3) + p_spinrate(i, 1)*p_sigma(i, 4));
                        dsigmayy -= F(2.)*(-p_spinrate(i, 0)*p_sigma(i, 3) + p_spinrate(i, 2)*p_sigma(i, 5));
                        dsigmazz += F(2.)*(p_spinrate(i, 1)*p_sigma(i, 4) + p_spinrate(i, 2)*p_sigma(i, 5));
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

                template<typename F>
                struct drucker_prager_elastoplastic {
                    F dt, phi, psi, cohesion, alpha_phi, alpha_psi, k_c, E, v, D0;
                    const Kokkos::View<F*[6]> p_sigma, p_strainrate;
                    const Kokkos::View<F*[3]> p_spinrate;

                    drucker_prager_elastoplastic(Kokkos::View<F*[6]> p_sigma_, Kokkos::View<F*[6]> p_strainrate_, 
                        Kokkos::View<F*[3]> p_spinrate_)
                        : p_sigma {p_sigma_}
                        , p_strainrate {p_strainrate_}
                        , p_spinrate {p_spinrate_}
                    {}

                    void set_DP_params(F phi_, F psi_, F c_, F E_, F v_) {
                        phi = phi_;
                        psi = psi_;
                        cohesion = c_;
                        alpha_phi = F(2.)*std::sin(phi)/(std::sqrt(F(3.))*(F(3.)-std::sin(phi)));
                        alpha_psi = F(2.)*std::sin(psi)/(std::sqrt(F(3.))*(F(3.)-std::sin(phi)));
                        k_c = F(6.)*cohesion*std::cos(phi)/(std::sqrt(F(3.))*(F(3.)-std::sin(phi)));
                        E = E_;
                        v = v_;
                        D0 = E/((F(1.)+v)*(F(1.)-F(2.)*v));
                    }

                    KOKKOS_INLINE_FUNCTION
                    void operator()(const int i) const {

                        // elastic stiffness matrix
                        F dsigmaxx = D0*((F(1.)-v)*p_strainrate(i, 0) + v*p_strainrate(i, 1) + v*p_strainrate(i, 2));
                        F dsigmayy = D0*(v*p_strainrate(i, 0) + (F(1.)-v)*p_strainrate(i, 1) + v*p_strainrate(i, 2));
                        F dsigmazz = D0*(v*p_strainrate(i, 0) + v*p_strainrate(i, 1) + (F(1.)-v)*p_strainrate(i, 2));
                        F dsigmaxy = D0*p_strainrate(i, 3)*(F(1.)-F(2.)*v);
                        F dsigmaxz = D0*p_strainrate(i, 4)*(F(1.)-F(2.)*v);
                        F dsigmayz = D0*p_strainrate(i, 5)*(F(1.)-F(2.)*v);

                        // jaumann stress rate
                        dsigmaxx -= F(2.)*(p_spinrate(i, 0)*p_sigma(i, 3) + p_spinrate(i, 1)*p_sigma(i, 4));
                        dsigmayy -= F(2.)*(-p_spinrate(i, 0)*p_sigma(i, 3) + p_spinrate(i, 2)*p_sigma(i, 5));
                        dsigmazz += F(2.)*(p_spinrate(i, 1)*p_sigma(i, 4) + p_spinrate(i, 2)*p_sigma(i, 5));
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

                        // calculating invariants and deviatoric stress tensor
                        F I1 = p_sigma(i, 0) + p_sigma(i, 1) + p_sigma(i, 2);
                        const F s[6] {
                            p_sigma(i, 0) - I1/F(3.),
                            p_sigma(i, 1) - I1/F(3.),
                            p_sigma(i, 2) - I1/F(3.),
                            p_sigma(i, 3),
                            p_sigma(i, 4),
                            p_sigma(i, 5),
                        };
                        const F J2 = F(0.5)*(s[0]*s[0] + s[1]*s[1] + s[2]*s[2] + 2.*(s[3]*s[3] + s[4]*s[4] + s[5]*s[5]));
                        // tensile correction 1
                        if (J2 == 0. && I1 > k_c/alpha_phi) {
                            p_sigma(i, 0) = k_c/alpha_phi/F(3.);
                            p_sigma(i, 1) = k_c/alpha_phi/F(3.);
                            p_sigma(i, 2) = k_c/alpha_phi/F(3.);
                            I1 = k_c/alpha_phi;
                        }

                        // calculate yield function value
                        // max used to reduce branching
                        const F f {alpha_phi*I1 + Kokkos::sqrt(J2) - k_c};
                        
                        if (f > F(1.e-13)) {
                            const F snorm = F(2.)*Kokkos::sqrt(J2);
                            const F shat[6] {
                                s[0]/snorm,
                                s[1]/snorm,
                                s[2]/snorm,
                                s[3]/snorm,
                                s[4]/snorm,
                                s[5]/snorm,
                            };
                            const F dfdsig[6] {
                                alpha_phi + shat[0],
                                alpha_phi + shat[1],
                                alpha_phi + shat[2],
                                shat[3],
                                shat[4],
                                shat[5],
                            };
                            const F dgdsig[6] {
                                alpha_psi + shat[0],
                                alpha_psi + shat[1],
                                alpha_psi + shat[2],
                                shat[3],
                                shat[4],
                                shat[5],
                            };
                            const F dlambda {f/(
                                dfdsig[0]*D0*((F(1.)-v)*dgdsig[0] + v*dgdsig[1] + v*dgdsig[2]) +
                                dfdsig[1]*D0*(v*dgdsig[0] + (F(1.)-v)*dgdsig[1] + v*dgdsig[2]) +
                                dfdsig[2]*D0*(v*dgdsig[0] + v*dgdsig[1] + (F(1.)-v)*dgdsig[2]) +
                                F(2.)*dfdsig[3]*D0*dgdsig[3]*(F(1.)-F(2.)*v) +
                                F(2.)*dfdsig[4]*D0*dgdsig[4]*(F(1.)-F(2.)*v) +
                                F(2.)*dfdsig[5]*D0*dgdsig[5]*(F(1.)-F(2.)*v)
                            )};
                            
                            p_sigma(i, 0) -= dlambda*D0*((F(1.)-v)*dgdsig[0] + v*dgdsig[1] + v*dgdsig[2]);
                            p_sigma(i, 1) -= dlambda*D0*(v*dgdsig[0] + (F(1.)-v)*dgdsig[1] + v*dgdsig[2]);
                            p_sigma(i, 2) -= dlambda*D0*(v*dgdsig[0] + v*dgdsig[1] + (F(1.)-v)*dgdsig[2]);
                            p_sigma(i, 3) -= dlambda*D0*dgdsig[3]*(F(1.)-F(2.)*v);
                            p_sigma(i, 4) -= dlambda*D0*dgdsig[4]*(F(1.)-F(2.)*v);
                            p_sigma(i, 5) -= dlambda*D0*dgdsig[5]*(F(1.)-F(2.)*v);

                            I1 = p_sigma(i, 0) + p_sigma(i, 1) + p_sigma(i, 2);
                            if (I1 > k_c/alpha_phi) {
                                p_sigma(i, 0) = k_c/alpha_phi/F(3.);
                                p_sigma(i, 1) = k_c/alpha_phi/F(3.);
                                p_sigma(i, 2) = k_c/alpha_phi/F(3.);
                                p_sigma(i, 3) = F(0.);
                                p_sigma(i, 4) = F(0.);
                                p_sigma(i, 5) = F(0.);
                            }
                        }
                    }
                };
            }
        }
    }
}

#endif