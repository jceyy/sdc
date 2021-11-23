// @sect3{File: coeff.cu}
// This file contains the source for the class Coeff,
// which manages the Chandrasekhar coefficients.
#include "coeff.h"


// @sect4{Constuctor: Coeff}
//
// Set value for number of z modes (@p MOZ) and @p initialized status flag.
// @param moz : The maximum number of z modes.
template<typename T>
Coeff<T>::Coeff(int moz) : MOZ(moz),
    rza(MOZ, T(0.)),
    rzzza(MOZ, T(0.)),
    rac(MOZ*MOZ, T(0.)),
    rab(MOZ*MOZ, T(0.)),
    ra1(MOZ*MOZ, T(0.)),
    ra2(MOZ*MOZ, T(0.)),
    r112(MOZ*MOZ*MOZ, T(0.)),
    r122(MOZ*MOZ*MOZ, T(0.)),
    rab1(MOZ*MOZ*MOZ, T(0.)),
    rac1(MOZ*MOZ*MOZ, T(0.)),
    rbb1(MOZ*MOZ*MOZ, T(0.)),
    rbc1(MOZ*MOZ*MOZ, T(0.)),
    rcc1(MOZ*MOZ*MOZ, T(0.)),
    raab(MOZ*MOZ*MOZ, T(0.)),
    rabc(MOZ*MOZ*MOZ, T(0.)),
    ra11(MOZ*MOZ*MOZ, T(0.)),
    ra12(MOZ*MOZ*MOZ, T(0.)),
    ra22(MOZ*MOZ*MOZ, T(0.)),
    r111(MOZ*MOZ*MOZ, T(0.)),
    rb11(MOZ*MOZ*MOZ, T(0.)),
    rc11(MOZ*MOZ*MOZ, T(0.)),
    raa1(MOZ*MOZ*MOZ, T(0.)),
    raaa(MOZ*MOZ*MOZ, T(0.)),
    rabb(MOZ*MOZ*MOZ, T(0.)),
    rbc2(MOZ*MOZ*MOZ, T(0.)),
    rc12(MOZ*MOZ*MOZ, T(0.)),
    rb22(MOZ*MOZ*MOZ, T(0.)),
    rab2(MOZ*MOZ*MOZ, T(0.))
{
    initialize();
}


// @sect4{Destructor: Coeff}
//
// Free memory.
template<typename T>
Coeff<T>::~Coeff()
{
}


// @sect4{Function: initialize}
//
// This function calculates all the constant
// coefficients.
template<typename T>
void Coeff<T>::initialize()
{
    // Calculate rza and rzzza
    // Even terms
    for(int i = 2; i <= MOZ; i += 2) { // for i even
        T lambdai = lambda(i);
        T coti = 1.0 / tan(0.5 * lambdai);
        T cothi = 1.0 / tanh(0.5 * lambdai);
        T lambdai2 = lambdai * lambdai;
        rza[i-1]   = (coti + cothi) / lambdai - T(4.0) / lambdai2;
        rzzza[i-1] = ((lambdai2-T(24.0)) * coti + (lambdai2+T(24.0)) * cothi - T(12.0) * lambdai)
                     / (4*lambdai*lambdai2);
    }


    // Calculate rac
    // Odd terms
    for(int i = 1; i <= MOZ; i += 2) { // for i odd
        T lambdai = lambda(i);
        T lambdatani = lambdai * tan(0.5 * lambdai);
        T lambdai2 = lambdai * lambdai;

        // Diagonal
        rac[(i-1)+(i-1)*MOZ] = -2.0 * lambdatani - lambdatani * lambdatani;

        // Off-diagonal
        for(int j = i+2; j <= MOZ; j += 2) { // for j odd
            T lambdaj = lambda(j);
            T lambdatanj = lambdaj * tan(0.5 * lambdaj);
            T lambdaj2 = lambdaj * lambdaj;

            T rij = 8.0*lambdai2*lambdaj2*(lambdatanj-lambdatani)/(lambdai2*lambdai2-lambdaj2*lambdaj2);
            rac[(i-1)+(j-1)*MOZ] = rij;
            rac[(j-1)+(i-1)*MOZ] = rij;
      }
    }

    // Even terms
    for(int i = 2; i <= MOZ; i += 2) { // for i even
        T lambdai = lambda(i);
        T lambdacoti = lambdai / tan(0.5 * lambdai);
        T lambdai2 = lambdai * lambdai;

        // Diagonal
        rac[(i-1)+(i-1)*MOZ] = 2.0 * lambdacoti - lambdacoti * lambdacoti;

        // Off-diagonal
        for(int j = i+2; j <= MOZ; j += 2) { // for j even
            T lambdaj = lambda(j);
            T lambdacotj = lambdaj / tan(0.5 * lambdaj);
            T lambdaj2 = lambdaj * lambdaj;

            T rij = 8.0*lambdai2*lambdaj2*(lambdacotj-lambdacotj)/(lambdai2*lambdai2-lambdaj2*lambdaj2);
            rac[(i-1)+(j-1)*MOZ] = rij;
            rac[(j-1)+(i-1)*MOZ] = rij;
        }
    }

    // Calculate rab
    // Odd terms
    for(int i = 1; i <= MOZ; i += 2) { // for i odd
        T lambdai = lambda(i);
        T zeroi = lambdai / M_PI;
        T ci = cos(0.5 * lambdai);
        T chi = cosh(0.5 * lambdai);
        T cdi = 1.0 / chi;

        // Odd-even
        for(int j = 2; j <= MOZ; j += 2) { // for j even
            T lambdaj = lambda(j);
            T zeroj = lambdaj / M_PI;
            T sj = sin(0.5 * lambdaj);
            T shj = sinh(0.5 * lambdaj);
            T sdj = 1.0 / shj;

            T rij = zeroj*( (gi1(zeroi+zeroj)+gi1(zeroi-zeroj)) *0.5*cdi*sdj
                        +(gi0(zeroi+zeroj)+gi0(zeroi-zeroj)) /(2.0*ci*sj)
                        - gi2(zeroi,zeroj)*cdi/sj - gi2(zeroj,zeroi)*sdj/ci);
            rab[(i-1)+(j-1)*MOZ] = rij;
            rab[(j-1)+(i-1)*MOZ] = -rij;
        }
    }

    // Calculate ra1 and ra2
    // Odd terms
    for(int i = 1; i <= MOZ; i += 2) { // for i odd
        T lambdai = lambda(i);
        T zeroi = lambdai / M_PI;
        T zerotani = zeroi * 2.0 * tan(0.5 * lambdai);
        T zeroi2 = zeroi * zeroi;

        // Even terms for ra2
        for(int j = 2; j <= MOZ; j += 2) { // for j even
            int j2 = j*j;
            ra2[(i-1)+(j-1)*MOZ] = 2.0*j*zeroi2*zerotani/(j2*j2-zeroi2*zeroi2);
        }
        // Odd terms for ra1
        for(int j = 1; j <= MOZ; j += 2) { // for j odd
            int j2 = j*j;
            ra1[(i-1)+(j-1)*MOZ] = -4.0*j*zeroi2/(M_PI*(j2*j2-zeroi2*zeroi2));
        }
    }
    // Even terms
    for(int i = 2; i <= MOZ; i += 2) { // for i even
        T lambdai = lambda(i);
        T zeroi = lambdai / M_PI;
        T zerocoti = zeroi * 2.0 / tan(0.5 * lambdai);
        T zeroi2 = zeroi * zeroi;

        // Even terms for ra1
        for(int j = 2; j <= MOZ; j += 2) { // for j even
            int j2 = j*j;
            ra1[(i-1)+(j-1)*MOZ] = 4.0*j*zeroi2/(M_PI*(j2*j2-zeroi2*zeroi2));
        }
        // Odd terms for ra2
        for(int j = 1; j <= MOZ; j += 2) { // for j odd
            int j2 = j*j;
            ra2[(i-1)+(j-1)*MOZ] = 2.0*j*zeroi2*zerocoti/(j2*j2-zeroi2*zeroi2);
        }
    }

    // Calculate all integrals with 3 terms
    for(int i = 1; i <= MOZ; i += 2) {// for i odd

        // Load input constants for odd i
        T lambdai = lambda(i);
        T rni = lambdai / M_PI;
        T ci = cos(0.5 * lambdai);
        T chi = cosh(0.5 * lambdai);
        T cdi = 1.0 / chi;

        for(int j = 1; j <= MOZ; j += 2) {// for j odd

            // Load input constants for odd j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T cj = cos(0.5 * lambdaj);
            T chj = cosh(0.5 * lambdaj);
            T cdj = 1.0 / chj;

            for(int k = 1; k <= MOZ; k += 2) {// for k odd

                // Load input constants for odd k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T ck = cos(0.5 * lambdak);
                T chk = cosh(0.5 * lambdak);
                T cdk = 1.0 / chk;

                // Evaluate simple integrals
                T gi4p = gi4(rni,j+k,j+k);
                T gi4m = gi4(rni,j-k,j-k);
                T gi9pp = gi9(j+k+rni,j+k);
                T gi9pm = gi9(j+k-rni,j+k);
                T gi9mp = gi9(j-k+rni,j-k);
                T gi9mm = gi9(j-k-rni,j-k);

                // Compose final integrals
                r122[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (gi8(i+j+k,i+j+k)+gi8(i-j-k,i-j-k)
                           +gi8(i+j-k,i+j-k)+gi8(i-j+k,i-j+k))
                        * 0.25*j*k;
                ra22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = j*k*((gi4p+gi4m)/(2*chi)-(gi9pp+gi9pm+gi9mp+gi9mm)/(4*ci));
                raaa[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (  gi1(rni+rnj+rnk) + gi1(rni+rnj-rnk)
                             + gi1(rni-rnj+rnk) + gi1(rni-rnj-rnk))*0.25 *cdi*cdj*cdk
                        - (gi2(rni+rnj,rnk)+gi2(rni-rnj,rnk))*0.5 *cdi*cdj/ck
                        - (gi2(rni+rnk,rnj)+gi2(rni-rnk,rnj))*0.5 *cdi*cdk/cj
                        + (gi2(rni,rnj+rnk)+gi2(rni,rnj-rnk))*0.5 *cdi/(cj*ck)
                        - (gi2(rnj+rnk,rni)+gi2(rnj-rnk,rni))*0.5 *cdj*cdk/ci
                        + (gi2(rnj,rni+rnk)+gi2(rnj,rni-rnk))*0.5 *cdj/(ci*ck)
                        + (gi2(rnk,rni+rnj)+gi2(rnk,rni-rnj))*0.5 *cdk/(cj*ci)
                        - (  gi0(rni+rnj+rnk)+gi0(rni+rnj-rnk)
                             + gi0(rni-rnj+rnk)+gi0(rni-rnj-rnk))*0.25/(ci*cj*ck);
                rabb[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (( gi1(rni+rnj+rnk)-gi1(rni+rnj-rnk)
                             -gi1(rni-rnj+rnk)+gi1(rni-rnj-rnk))*0.25 *cdi*cdj*cdk
                           + (gi3(rni+rnj,rnk)-gi3(rni-rnj,rnk))*0.5 *cdi*cdj/ck
                           + (gi3(rni+rnk,rnj)-gi3(rni-rnk,rnj))*0.5 *cdi*cdk/cj
                           - (gi2(rni,rnj+rnk)-gi2(rni,rnj-rnk))*0.5 *cdi/(cj*ck)
                           - (gi2(rnj+rnk,rni)-gi2(rnj-rnk,rni))*0.5 *cdj*cdk/ci
                           - (gi3(rnj,rni+rnk)-gi3(rnj,rni-rnk))*0.5 *cdj/(ci*ck)
                           - (gi3(rnk,rni+rnj)-gi3(rnk,rni-rnj))*0.5 *cdk/(cj*ci)
                           + (gi0(rni+rnj+rnk)-gi0(rni+rnj-rnk)-gi0(rni-rnj+rnk)
                              + gi0(rni-rnj-rnk))*0.25/(ci*cj*ck))*rnj*rnk;
            }

            for(int k = 2; k <= MOZ; k += 2) {// for k even

                // Load input constants for even k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T sk = sin(0.5 * lambdak);
                T shk = sinh(0.5 * lambdak);
                T sdk = 1.0 / shk;

                // Evaluate simple integrals
                T gi0pp = gi0(rni+rnj+rnk);
                T gi0pm = gi0(rni+rnj-rnk);
                T gi0mp = gi0(rni-rnj+rnk);
                T gi0mm = gi0(rni-rnj-rnk);
                T gi1pp = gi1(rni+rnj+rnk);
                T gi1pm = gi1(rni+rnj-rnk);
                T gi1mp = gi1(rni-rnj+rnk);
                T gi1mm = gi1(rni-rnj-rnk);
                T gi2p1 = gi2(rnj+rnk,rni);
                T gi2m1 = gi2(rnj-rnk,rni);
                T gi2p2 = gi2(rni,rnj+rnk);
                T gi2m2 = gi2(rni,rnj-rnk);

                // Compose final integrals
                r112[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -k*0.25 * (
                            gi9((i+j+k),i+j+k) + gi9((i+j-k),i+j-k)
                            -gi9((i-j+k),i-j+k) - gi9((i-j-k),i-j-k));
                ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = k*(
                            (gi6(rni,j+k,j+k)+gi6(rni,j-k,j-k))  /(2*chi)
                            -( gi8(j+k+rni,j+k)+gi8(j+k-rni,j+k)
                               +gi8(j-k+rni,j-k)+gi8(j-k-rni,j-k))/(4*ci));
                raab[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnk*(
                            (gi1pp+gi1pm+gi1mp+gi1mm)           *cdi*cdj*sdk/4
                            -(gi2(rni+rnj,rnk)+gi2(rni-rnj,rnk))*cdi*cdj/(2*sk)
                            -(gi2(rni+rnk,rnj)+gi2(rni-rnk,rnj))*cdi*sdk/(2*cj)
                            +(gi2p2+gi2m2)                      *cdi/(2*cj*sk)
                            -(gi2p1+gi2m1)                      *cdj*sdk/(2*ci)
                            +(gi2(rnj,rni+rnk)+gi2(rnj,rni-rnk))*cdj/(2*ci*sk)
                            +(gi2(rnk,rni+rnj)+gi2(rnk,rni-rnj))*sdk/(2*ci*cj)
                            -(gi0pp+gi0pm+gi0mp+gi0mm)          /(4*ci*cj*sk));
                rabc[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnk*rnk*(
                            (gi1pp+gi1mm-gi1pm-gi1mp)           *cdi*cdj*sdk/4
                            +(gi3(rni+rnj,rnk)-gi3(rni-rnj,rnk))*cdi*cdj/(2*sk)
                            +(gi3(rni+rnk,rnj)-gi3(rni-rnk,rnj))*cdi*sdk/(2*cj)
                            -(gi2p2-gi2m2)                      *cdi/(2*cj*sk)
                            -(gi2p1-gi2m1)                      *cdj*sdk/(2*ci)
                            -(gi3(rnj,rni+rnk)-gi3(rnj,rni-rnk))*cdj/(2*ci*sk)
                            -(gi3(rnk,rni+rnj)-gi3(rnk,rni-rnj))*sdk/(2*ci*cj)
                            +(gi0pp+gi0mm-gi0pm-gi0mp)          /(4*ci*cj*sk));
            }   // end for k even
        }   // end for j odd

        for(int j = 2; j <= MOZ; j += 2) {// for j even

            // Load input constants for odd j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T sj = sin(0.5 * lambdaj);
            T shj = sinh(0.5 * lambdaj);
            T sdj = 1.0 / shj;

            for(int k = 2; k <= MOZ; k += 2) {// for k even

                // Load input constants for even k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T sk = sin(0.5 * lambdak);
                T shk = sinh(0.5 * lambdak);
                T sdk = 1.0 / shk;

                // Evaluate simple integrals
                T gi4p1 = gi4(rni,(j+k),j+k);
                T gi4m1 = gi4(rni,(j-k),j-k);
                T gi9pp1 = gi9(j+k+rni,j+k);
                T gi9pm1 = gi9(j+k-rni,j+k);
                T gi9mp1 = gi9(j-k+rni,j-k);
                T gi9mm1 = gi9(j-k-rni,j-k);

                // Compose final integrals
                r122[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (gi8((i+j+k),i+j+k)+gi8((i-j-k),i-j-k)
                           +gi8((i+j-k),i+j-k)+gi8((i-j+k),i-j+k))
                        *0.25*j*k;
                ra22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = j*k*((gi4p1+gi4m1)/(2*chi)-(gi9pp1+gi9pm1+gi9mp1+gi9mm1)/(4*ci));
                raaa[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (gi1(rni+rnj+rnk)-gi1(rni+rnj-rnk)
                           -gi1(rni-rnj+rnk)+gi1(rni-rnj-rnk))*0.25*cdi*sdj*sdk
                        -(gi3(rni+rnj,rnk)-gi3(rni-rnj,rnk))*0.5 *cdi*sdj/sk
                        -(gi3(rni+rnk,rnj)-gi3(rni-rnk,rnj))*0.5 *cdi*sdk/sj
                        -(gi2(rni,rnj+rnk)-gi2(rni,rnj-rnk))*0.5 *cdi/(sj*sk)
                        -(gi2(rnj+rnk,rni)-gi2(rnj-rnk,rni))*0.5 *sdj*sdk/ci
                        +(gi3(rnj,rni+rnk)-gi3(rnj,rni-rnk))*0.5 *sdj/(ci*sk)
                        +(gi3(rnk,rni+rnj)-gi3(rnk,rni-rnj))*0.5 *sdk/(sj*ci)
                        +(gi0(rni+rnj+rnk)-gi0(rni+rnj-rnk)-gi0(rni-rnj+rnk)
                          + gi0(rni-rnj-rnk))*0.25/(ci*sj*sk);
                rabb[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = ((gi1(rni+rnj+rnk)+gi1(rni+rnj-rnk)
                            +gi1(rni-rnj+rnk)+gi1(rni-rnj-rnk))*0.25*cdi*sdj*sdk
                           -(gi2(rni+rnj,rnk)+gi2(rni-rnj,rnk))*0.5 *cdi*sdj/sk
                           -(gi2(rni+rnk,rnj)+gi2(rni-rnk,rnj))*0.5 *cdi*sdk/sj
                           +(gi2(rni,rnj+rnk)+gi2(rni,rnj-rnk))*0.5 *cdi/(sj*sk)
                           -(gi2(rnj+rnk,rni)+gi2(rnj-rnk,rni))*0.5 *sdj*sdk/ci
                           +(gi2(rnj,rni+rnk)+gi2(rnj,rni-rnk))*0.5 *sdj/(ci*sk)
                           +(gi2(rnk,rni+rnj)+gi2(rnk,rni-rnj))*0.5 *sdk/(sj*ci)
                           -(gi0(rni+rnj+rnk)+gi0(rni+rnj-rnk)+gi0(rni-rnj+rnk)
                             +gi0(rni-rnj-rnk))*0.25/(ci*sj*sk))*rnj*rnk;

            }   //  end for k even

            for(int k = 1; k <= MOZ; k += 2) {//  for k odd

                // Load input constants for odd k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T ck = cos(0.5 * lambdak);
                T chk = cosh(0.5 * lambdak);
                T cdk = 1.0 / chk;

                // Evaluate simple integrals
                T gi0pp = gi0(rni+rnj+rnk);
                T gi0pm = gi0(rni+rnj-rnk);
                T gi0mp = gi0(rni-rnj+rnk);
                T gi0mm = gi0(rni-rnj-rnk);
                T gi1pp = gi1(rni+rnj+rnk);
                T gi1pm = gi1(rni+rnj-rnk);
                T gi1mp = gi1(rni-rnj+rnk);
                T gi1mm = gi1(rni-rnj-rnk);
                T gi2p1 = gi2(rnj+rnk,rni);
                T gi2m1 = gi2(rnj-rnk,rni);
                T gi2p2 = gi2(rni,rnj+rnk);
                T gi2m2 = gi2(rni,rnj-rnk);

                // Compose final integrals
                r112[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -k*0.25
                        *(gi9((i+j+k),i+j+k)+gi9((i+j-k),i+j-k)
                          -gi9((i-j+k),i-j+k)-gi9((i-j-k),i-j-k));
                ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = k*((gi6(rni,(j+k),j+k)+gi6(rni,(j-k),j-k)) /(2*chi)
                             -(gi8(j+k+rni,j+k)+gi8(j+k-rni,j+k)
                               +gi8(j-k+rni,j-k)+gi8(j-k-rni,j-k))           /(4*ci));
                raab[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnk*(
                            (gi1pp-gi1pm-gi1mp+gi1mm)/4*(cdi*sdj*cdk)
                            +(gi3(rni+rnj,rnk)-gi3(rni-rnj,rnk))/(2*ck)*cdi*sdj
                            -(gi3(rni+rnk,rnj)-gi3(rni-rnk,rnj))/(2*sj)*cdi*cdk
                            +(gi2p2-gi2m2)/(2*sj*ck)*cdi
                            -(gi2p1-gi2m1)/(2*ci)*sdj*cdk
                            -(gi3(rnj,rni+rnk)-gi3(rnj,rni-rnk))/(2*ci*ck)*sdj
                            +(gi3(rnk,rni+rnj)-gi3(rnk,rni-rnj))/(2*ci*sj)*cdk
                            -(gi0pp-gi0pm-gi0mp+gi0mm)/(4*ci*sj*ck));
                rabc[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnk*rnk*(
                            (gi1pp+gi1mm+gi1pm+gi1mp)/4*(cdi*sdj*cdk)
                            +(gi2(rni+rnj,rnk)+gi2(rni-rnj,rnk))/(2*ck)*cdi*sdj
                            -(gi2(rni+rnk,rnj)+gi2(rni-rnk,rnj))/(2*sj)*cdi*cdk
                            -(gi2p2+gi2m2)/(2*sj*ck)*cdi
                            -(gi2p1+gi2m1)/(2*ci)*sdj*cdk
                            -(gi2(rnj,rni+rnk)+gi2(rnj,rni-rnk))/(2*ci*ck)*sdj
                            +(gi2(rnk,rni+rnj)+gi2(rnk,rni-rnj))/(2*ci*sj)*cdk
                            +(gi0pp+gi0mm+gi0pm+gi0mp)/(4*ci*sj*ck));
            }   // end for k odd
        }   // end for j even
    }   // end for i odd

    for(int i = 2; i <= MOZ; i += 2) {// for i even

        // Load input constants for even i
        T lambdai = lambda(i);
        T rni = lambdai / M_PI;
        T si = sin(0.5 * lambdai);
        T shi = sinh(0.5 * lambdai);
        T sdi = 1.0 / shi;

        for(int j = 1; j <= MOZ; j += 2) {// for j odd

            // Load input constants for even j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T cj = cos(0.5 * lambdaj);
            T chj = cosh(0.5 * lambdaj);
            T cdj = 1.0 / chj;

            for(int k = 2; k <= MOZ; k += 2) {// for k even

                // Load input constants for even k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T sk = sin(0.5 * lambdak);
                T shk = sinh(0.5 * lambdak);
                T sdk = 1.0 / shk;

                // Evaluate simple integrals
                T gi5p = gi5(rni,(j+k),j+k);
                T gi5m = gi5(rni,(j-k),j-k);
                T gi8pp = gi8(j+k+rni,j+k);
                T gi8pm = gi8(j+k-rni,j+k);
                T gi8mp = gi8(j-k+rni,j-k);
                T gi8mm = gi8(j-k-rni,j-k);

                // Compose final integrals
                r122[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = ( gi8(i+j+k,i+j+k)+gi8(i-j-k,i-j-k)
                            +gi8(i+j-k,i+j-k)+gi8(i-j+k,i-j+k))*0.25*j*k;
                ra22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = j*k*((gi5p+gi5m)/(2*shi)-(gi8pp-gi8pm+gi8mp-gi8mm)/(4*si));
                raaa[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = raaa[(j-1)+(i-1)*MOZ+(k-1)*MOZ*MOZ];
                rabb[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = ((gi1(rnk+rni+rnj)+gi1(rni+rnj-rnk)
                            -gi1(rni-rnj+rnk)-gi1(rni-rnj-rnk))*0.25*sdi*cdj*sdk
                           -(gi2(rni+rnj,rnk)-gi2(rni-rnj,rnk))*0.5 *sdi*cdj/sk
                           +(gi3(rni+rnk,rnj)+gi3(rni-rnk,rnj))*0.5 *sdi*sdk/cj
                           -(gi3(rni,rnj+rnk)+gi3(rni,rnj-rnk))*0.5 *sdi/(cj*sk)
                           -(gi3(rnj+rnk,rni)+gi3(rnj-rnk,rni))*0.5*sdk*cdj/si
                           +(gi3(rnj,rni+rnk)+gi3(rnj,rni-rnk))*0.5 *cdj/(sk*si)
                           +(gi2(rnk,rni+rnj)-gi2(rnk,rni-rnj))*0.5 *sdk/(si*cj)
                           -(gi0(rnk+rni+rnj)+gi0(rni+rnj-rnk)-gi0(rni-rnj+rnk)
                             - gi0(rni-rnj-rnk))*0.25/(sk*si*cj))*rnj*rnk;
            } // end for k even

            for(int k = 1; k <= MOZ; k += 2) { // for k odd

                // Load input constants for odd k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T ck = cos(0.5 * lambdak);
                T chk = cosh(0.5 * lambdak);
                T cdk = 1.0 / chk;

                // Evaluate simple integrals
                T gi0pp = gi0(rni+rnj+rnk);
                T gi0pm = gi0(rni+rnj-rnk);
                T gi0mp = gi0(rni-rnj+rnk);
                T gi0mm = gi0(rni-rnj-rnk);
                T gi1pp = gi1(rni+rnj+rnk);
                T gi1pm = gi1(rni+rnj-rnk);
                T gi1mp = gi1(rni-rnj+rnk);
                T gi1mm = gi1(rni-rnj-rnk);
                T gi3p1 = gi3(rnj+rnk,rni);
                T gi3m1 = gi3(rnj-rnk,rni);
                T gi3p2 = gi3(rni,rnj+rnk);
                T gi3m2 = gi3(rni,rnj-rnk);

                // Compose final integrals
                r112[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -k*0.25*( gi9(i+j+k,i+j+k)+gi9(i+j-k,i+j-k)
                                    -gi9(i-j+k,i-j+k)-gi9(i-j-k,i-j-k));
                ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = k*((gi7(rni,(j+k),j+k)+gi7(rni,(j-k),j-k))/(2*shi)
                             +(gi9(j+k+rni,j+k)-gi9(j+k-rni,j+k)
                               +gi9(j-k+rni,j-k)-gi9(j-k-rni,j-k))/(4*si));
                raab[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnk*((gi1pp-gi1pm+gi1mp-gi1mm)/4*(sdi*cdj*cdk)
                               +(gi3(rni+rnj,rnk)+gi3(rni-rnj,rnk))/(2*ck)*sdi*cdj
                               -(gi3p1-gi3m1)/(2*si)*(cdj*cdk)
                               +(gi2(rnj,rni+rnk)-gi2(rnj,rni-rnk))/(2*si*ck)*cdj
                               -(gi2(rni+rnk,rnj)-gi2(rni-rnk,rnj))/(2*cj)*(sdi*cdk)
                               -(gi3p2-gi3m2)/(2*cj*ck)*sdi
                               +(gi3(rnk,rni+rnj)+gi3(rnk,rni-rnj))/(2*cj*si)*cdk
                               -(gi0pp-gi0pm+gi0mp-gi0mm)/(4*cj*si*ck));
                rabc[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnk*rnk*(
                            (gi1pp-gi1mm+gi1pm-gi1mp)/4*(sdi*cdj*cdk)
                            +(gi2(rni+rnj,rnk)-gi2(rni-rnj,rnk))/(2*ck)*sdi*cdj
                            +(gi3(rni+rnk,rnj)+gi3(rni-rnk,rnj))/(2*cj)*sdi*cdk
                            +(gi3p2+gi3m2)/(2*cj*ck)*sdi
                            -(gi3p1+gi3m1)/(2*si)*cdj*cdk
                            -(gi3(rnj,rni+rnk)+gi3(rnj,rni-rnk))/(2*si*ck)*cdj
                            +(gi2(rnk,rni+rnj)-gi2(rnk,rni-rnj))/(2*si*cj)*cdk
                            +(gi0pp-gi0mm+gi0pm-gi0mp)/(4*si*cj*ck));
            } // end for k even
        } // end for j odd

        for(int j = 2; j <= MOZ; j += 2) { // for j even

            // Load input constants for even j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T sj = sin(0.5 * lambdaj);
            T shj = sinh(0.5 * lambdaj);
            T sdj = 1.0 / shj;

            for(int k = 1; k <= MOZ; k += 2) {// for k odd

                // Evaluate simple integrals
                T gi5p = gi5(rni,j+k,j+k);
                T gi5m = gi5(rni,j-k,j-k);
                T gi8pp1 = gi8(j+k+rni,j+k);
                T gi8pm1 = gi8(j+k-rni,j+k);
                T gi8mp1 = gi8(j-k+rni,j-k);
                T gi8mm1 = gi8(j-k-rni,j-k);

                // Compose final integrals
                r122[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (gi8((i+j+k),i+j+k)+gi8((i-j-k),i-j-k)
                           +gi8((i+j-k),i+j-k)+gi8((i-j+k),i-j+k))*0.25*j*k;
                ra22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = j*k*((gi5p+gi5m)/(2*shi)-(gi8pp1-gi8pm1+gi8mp1-gi8mm1)/(4*si));
                raaa[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] = raaa[(i-1)+(k-1)*MOZ+(j-1)*MOZ*MOZ];
                rabb[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] = rabb[(i-1)+(k-1)*MOZ+(j-1)*MOZ*MOZ];
            } // end for k odd

            for(int k = 2; k <= MOZ; k += 2) {// for k even

                // Load input constants for even k
                T lambdak = lambda(k);
                T rnk = lambdak / M_PI;
                T sk = sin(0.5 * lambdak);
                T shk = sinh(0.5 * lambdak);
                T sdk = 1.0 / shk;

                // Evaluate simple integrals
                T gi0pp = gi0(rni+rnj+rnk);
                T gi0pm = gi0(rni+rnj-rnk);
                T gi0mp = gi0(rni-rnj+rnk);
                T gi0mm = gi0(rni-rnj-rnk);
                T gi1pp = gi1(rni+rnj+rnk);
                T gi1pm = gi1(rni+rnj-rnk);
                T gi1mp = gi1(rni-rnj+rnk);
                T gi1mm = gi1(rni-rnj-rnk);
                T gi3p1 = gi3(rnj+rnk,rni);
                T gi3m1 = gi3(rnj-rnk,rni);
                T gi3p2 = gi3(rni,rnj+rnk);
                T gi3m2 = gi3(rni,rnj-rnk);

                // Compose final integrals
                r112[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -k*0.25*(
                            gi9(i+j+k,i+j+k)+gi9(i+j-k,i+j-k)
                            -gi9(i-j+k,i-j+k)-gi9(i-j-k,i-j-k));
                ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = k*((gi7(rni,j+k,j+k)+gi7(rni,j-k,j-k))/(2*shi)
                             +(gi9(j+k+rni,j+k)-gi9(j+k-rni,j+k)
                               +gi9(j-k+rni,j-k)-gi9(j-k-rni,j-k))/(4*si));
                raab[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnk*((gi1pp+gi1pm-gi1mp-gi1mm)/4*(sdi*sdj*sdk)
                               -(gi2(rni+rnj,rnk)-gi2(rni-rnj,rnk))/(2*sk)*sdi*sdj
                               -(gi3(rni+rnk,rnj)+gi3(rni-rnk,rnj))/(2*sj)*sdi*sdk
                               +(gi3p2+gi3m2)/(2*sj*sk)*sdi
                               -(gi3p1+gi3m1)/(2*si)*sdj*sdk
                               +(gi3(rnj,rni+rnk)+gi3(rnj,rni-rnk))/(2*si*sk)*sdj
                               -(gi2(rnk,rni+rnj)-gi2(rnk,rni-rnj))/(2*si*sj)*sdk
                               +(gi0pp+gi0pm-gi0mp-gi0mm)/(4*si*sj*sk));
                rabc[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnk*rnk*(
                            (gi1pp-gi1mm-gi1pm+gi1mp)/4*(sdi*sdj*sdk)
                            +(gi3(rni+rnj,rnk)+gi3(rni-rnj,rnk))/(2*sk)*sdi*sdj
                            -(gi2(rni+rnk,rnj)-gi2(rni-rnk,rnj))/(2*sj)*sdi*sdk
                            -(gi3p2-gi3m2)/(2*sj*sk)*sdi
                            -(gi3p1-gi3m1)/(2*si)*sdj*sdk
                            +(gi2(rnj,rni+rnk)-gi2(rnj,rni-rnk))/(2*si*sk)*sdj
                            +(gi3(rnk,rni+rnj)+gi3(rnk,rni-rnj))/(2*si*sj)*sdk
                            -(gi0pp-gi0mm-gi0pm+gi0mp)/(4*si*sj*sk));
            } // end for k even
        } // end for j even
    } // end for i even

    // Compute integrals


    for(int i = 1; i <= MOZ; i += 2) { // for i odd

        // Load input constants for odd i
        T lambdai = lambda(i);
        T rni = lambdai / M_PI;
        T ci = cos(0.5 * lambdai);
        T chi = cosh(0.5 * lambdai);
        T cdi = 1.0 / chi;

        for(int j = 1; j <= MOZ; j += 2) { // for j odd

            // Load input constants for odd j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T cj = cos(0.5 * lambdaj);
            T chj = cosh(0.5 * lambdaj);
            T cdj = 1.0 / chj;
            for(int k = 1; k <= MOZ; k += 2) { // for k  odd

                // Evaluate simple integrals
                T gi4p = gi4(rni,(j+k),j+k);
                T gi4m = gi4(rni,(j-k),j-k);
                T gi6ijp = gi6(rni+rnj,(k),k);
                T gi6ijm = gi6(rni-rnj,(k),k);
                T gi6kjp = gi6(rni,k+rnj,k);
                T gi6kjm = gi6(rni,k-rnj,k);
                T gi6kip = gi6(rnj,k+rni,k);
                T gi6kim = gi6(rnj,k-rni,k);
                T gi8pp = gi8(k+rni+rnj,k);
                T gi8mm = gi8(k-rni-rnj,k);
                T gi8pm = gi8(k+rni-rnj,k);
                T gi8mp = gi8(k-rni+rnj,k);
                T gi9pp = gi9(j+k+rni,j+k);
                T gi9pm = gi9(j+k-rni,j+k);
                T gi9mp = gi9(j-k+rni,j-k);
                T gi9mm = gi9(j-k-rni,j-k);

                // Compose final integrals
                ra11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -(gi4p-gi4m)/(2*chi)
                        +(gi9pp+gi9pm-gi9mp-gi9mm)/(4*ci);
                rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnj*((gi6ijp+gi6ijm)/(2*chi*chj)
                                   +(gi6kjp+gi6kjm)/(2*chi*cj)
                                   -(gi6kip+gi6kim)/(2*ci*chj)
                                   -(gi8pp+gi8mm+gi8pm+gi8mp)/(4*ci*cj));
                rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*((gi6ijp-gi6ijm)/(2*chi*chj)
                                   -(gi5(rni,k+rnj,k)-gi5(rni,k-rnj,k))/(2*chi*cj)
                                   -(gi5(rnj,k+rni,k)-gi5(rnj,k-rni,k))/(2*chj*ci)
                                   -(gi8pp+gi8mm-gi8pm-gi8mp)/(4*ci*cj));
                rcc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni*rnj*rnj
                        *((gi6ijp+gi6ijm)/(2*chi*chj)
                          +(gi6kjp+gi6kjm)/(2*chi*cj)
                          +(gi6kip+gi6kim)/(2*chj*ci)
                          +(gi8pp+gi8mm+gi8pm+gi8mp)/(4*ci*cj));
                r111[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (-gi8((i+j+k),i+j+k)-gi8((i-j-k),i-j-k)
                           +gi8((i+j-k),i+j-k)+gi8((i-j+k),i-j+k))*0.25;
                rc11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni*((gi4p-gi4m)/(-2*chi)-(gi9pp+gi9pm-gi9mp-gi9mm)/(4*ci));
                raa1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = ((gi6ijp+gi6ijm)/(2*chi*chj)
                           -(gi6kjp+gi6kjm)/(2*chi*cj)
                           -(gi6kip+gi6kim)/(2*chj*ci)
                           +(gi8pp+gi8mm+gi8pm+gi8mp)/(4*ci*cj));
                rbc2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj*(k)
                        *((gi5(rni+rnj,k,k)+gi5(rni-rnj,k,k))
                          *0.5*cdi*cdj
                          +(gi8(rni+rnj+k,k)+gi8(rni+rnj-k,-k)
                            +gi8(rni-rnj+k,k)+gi8(rni-rnj-k,-k))/(4.0*ci*cj)
                          +(gi5(rni,k+rnj,k)+gi5(rni,rnj-k,-k))*cdi/(2.0*cj)
                          +(gi6(rnj,k+rni,k)+gi6(rnj,rni-k,-k))*cdj/(2.0*ci));
                rab2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] - rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ];

            } // end for k odd

            for(int k = 2; k <= MOZ; k += 2) { // for k even

                // Evaluate simple integrals
                T gi7p = gi7(rni+rnj,k,k);
                T gi7m = gi7(rni-rnj,k,k);
                T gi9pp = gi9(k+rni+rnj,k);
                T gi9pm = gi9(k+rni-rnj,k);
                T gi9mp = gi9(k-rni+rnj,k);
                T gi9mm = gi9(k-rni-rnj,k);

                // Compose final integrals
                rab1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*((gi7p-gi7m)/(2*chi*chj)
                               -(gi4(rni,k+rnj,k)-gi4(rni,k-rnj,k))/(2*chi*cj)
                               -(gi7(rnj,k+rni,k)+gi7(rnj,k-rni,k))/(2*ci*chj)
                               +(gi9pp+gi9mp-gi9pm-gi9mm)/(4*ci*cj));
                rbc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj*(
                            (gi7p+gi7m)/(2*chi*chj)
                            +(gi7(rni,k+rnj,k)+gi7(rni,k-rnj,k))/(2*chi*cj)
                            -(gi4(rnj,k+rni,k)-gi4(rnj,k-rni,k))/(2*ci*chj)
                            -(gi9pp-gi9mm+gi9pm-gi9mp)/(4*ci*cj));
                rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*(
                            (gi5(rni,(j+k),j+k)-gi5(rni,(j-k),j-k))/(-2*chi)
                            -(gi8(j+k+rni,j+k)-gi8(j+k-rni,j+k)
                              -gi8(j-k+rni,j-k)+gi8(j-k-rni,j-k))/(4*ci));
                rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]*j*j
                        + ra12[(i-1)+(k-1)*MOZ+(j-1)*MOZ*MOZ]*k*k;
                rc12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = - rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        + rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]*k*k;

            } // end for k even
        } // end for j odd

        for(int j = 2; j <= MOZ; j += 2) { // for j even

            // Load input constants for even j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T sj = sin(0.5 * lambdaj);
            T shj = sinh(0.5 * lambdaj);

            for(int k = 2; k <= MOZ; k += 2) { // for k even

                // Evaluate simple integrals
                T gi4p1 = gi4(rni,(j+k),j+k);
                T gi4m1 = gi4(rni,(j-k),j-k);
                T gi4p = gi4(rni,k+rnj,k);
                T gi4m = gi4(rni,k-rnj,k);
                T gi7p = gi7(rni+rnj,(k),k);
                T gi7m = gi7(rni-rnj,(k),k);
                T gi9pp = gi9(k+rni+rnj,k);
                T gi9pm = gi9(k+rni-rnj,k);
                T gi9mp = gi9(k-rni+rnj,k);
                T gi9mm = gi9(k-rni-rnj,k);
                T gi9pp1 = gi9(j+k+rni,j+k);
                T gi9pm1 = gi9(j+k-rni,j+k);
                T gi9mp1 = gi9(j-k+rni,j-k);
                T gi9mm1 = gi9(j-k-rni,j-k);

                // Compose final integrals
                ra11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -(gi4p1-gi4m1)/(2*chi)+(gi9pp1+gi9pm1-gi9mp1-gi9mm1)/(4*ci);
                rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnj
                        *((gi7p-gi7m)/(2*chi*shj)
                          -(gi4p-gi4m)/(2*chi*sj)
                          -(gi7(rnj,k+rni,k)+gi7(rnj,k-rni,k))/(2*ci*shj)
                          +(gi9pp+gi9mp-gi9pm-gi9mm)/(4*ci*sj));
                rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*((gi7p+gi7m)/(2*chi*shj)
                                   -(gi7(rni,k+rnj,k)+gi7(rni,k-rnj,k))
                                   /(2*chi*sj)
                                   -(gi4(rnj,k+rni,k)-gi4(rnj,k-rni,k))
                                   /(2*shj*ci)
                                   +(gi9pp-gi9mm+gi9pm-gi9mp)/(4*ci*sj));
                rcc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni*rnj*rnj
                        *((gi7p-gi7m)/(2*chi*shj)
                          +(gi7(rnj,k+rni,k)+gi7(rnj,k-rni,k))/(2*ci*shj)
                          -(gi4p-gi4m)/(2*chi*sj)
                          -(gi9pp-gi9mm-gi9pm+gi9mp)/(4*ci*sj));
                r111[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (-gi8((i+j+k),i+j+k)-gi8((i-j-k),i-j-k)
                           +gi8((i+j-k),i+j-k)+gi8((i-j+k),i-j+k))
                        *0.25;
                rc11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni*((gi4p1-gi4m1)/(-2*chi)-(gi9pp1+gi9pm1-gi9mp1-gi9mm1)/(4.0*ci));
                raa1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =  (gi7p-gi7m)/(2*chi*shj)
                        -(gi7(rnj,k+rni,k)+gi7(rnj,k-rni,k))/(2*ci*shj)
                        +(gi4p-gi4m)/(2*chi*sj)
                        -(gi9pp-gi9mm-gi9pm+gi9mp)/(4*ci*sj);
                rbc2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj*k
                        *((gi4(rni+rnj,(k),k)+gi4(rni-rnj,(k),k))
                          /(2.0*chi*shj)
                          -(gi9(rni+rnj+k,k)+gi9(rni+rnj-k,-k)
                            -gi9(rni-rnj+k,k)-gi9(rni-rnj-k,-k))/(4.0*ci*sj)
                          +(gi7(rni,k+rnj,k)+gi7(rni,rnj-k,-k))/(2.0*chi*sj)
                          +(gi7(rnj,k+rni,k)+gi7(rnj,rni-k,-k))/(2.0*ci*shj));
                rab2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] -rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ];

            } // end for k even

            for(int k = 1; k <= MOZ; k += 2) { // for k odd

                // Evaluate simple integrals
                T gi6p = gi6(rni+rnj,(k),k);
                T gi6m = gi6(rni-rnj,(k),k);
                T gi8pp = gi8(k+rni+rnj,k);
                T gi8mm = gi8(k-rni-rnj,k);
                T gi8pm = gi8(k+rni-rnj,k);
                T gi8mp = gi8(k-rni+rnj,k);

                // Compose final integrals
                rab1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*((gi6p+gi6m)/(2*chi*shj)
                               -(gi6(rni,k+rnj,k)+gi6(rni,k-rnj,k))/(2*chi*sj)
                               -(gi6(rnj,k+rni,k)+gi6(rnj,k-rni,k))/(2*ci*shj)
                               +(gi8pp+gi8mp+gi8pm+gi8mm)/(4*ci*sj));
                rbc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj
                        *((gi6p-gi6m)/(2*chi*shj)
                          -(gi5(rni,k+rnj,k)-gi5(rni,k-rnj,k))/(2*chi*sj)
                          -(gi5(rnj,k+rni,k)-gi5(rnj,k-rni,k))/(2*ci*shj)
                          -(gi8pp+gi8mm-gi8pm-gi8mp)/(4*ci*sj));
                rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*((gi5(rni,(j+k),j+k)-gi5(rni,(j-k),j-k))/(-2*chi)
                               -(gi8(j+k+rni,j+k)-gi8(j+k-rni,j+k)
                                 -gi8(j-k+rni,j-k)+gi8(j-k-rni,j-k))/(4*ci));

                rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =   ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]*j*j+ra12[(i-1)+(k-1)*MOZ+(j-1)*MOZ*MOZ]*k*k;

                rc12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =   -rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] + rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] *k*k;
            } // end for k odd
        } // end for j even
    } // end for i odd

    for(int i = 2; i <= MOZ; i += 2) { // for i even

        // Load input constants for even i
        T lambdai = lambda(i);
        T rni = lambdai / M_PI;
        T si = sin(0.5 * lambdai);
        T shi = sinh(0.5 * lambdai);

        for(int j = 1; j <= MOZ; j += 2) { // for j odd

            // Load input constants for odd j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T cj = cos(0.5 * lambdaj);
            T chj = cosh(0.5 * lambdaj);

            for(int k = 2; k <= MOZ; k += 2) { // for k even

                // Evaluate simple integrals
                T gi4p = gi4(rnj,k+rni,k);
                T gi4m = gi4(rnj,k-rni,k);
                T gi5p = gi5(rni,(j+k),j+k);
                T gi5m = gi5(rni,(j-k),j-k);
                T gi7p1 = gi7(rni+rnj,(k),k);
                T gi7m1 = gi7(rni-rnj,(k),k);
                T gi7p2 = gi7(rni,k+rnj,k);
                T gi7m2 = gi7(rni,k-rnj,k);
                T gi8pp = gi8(j+k+rni,j+k);
                T gi8pm = gi8(j+k-rni,j+k);
                T gi8mp = gi8(j-k+rni,j-k);
                T gi8mm = gi8(j-k-rni,j-k);
                T gi9pp = gi9(k+rni+rnj,k);
                T gi9pm = gi9(k+rni-rnj,k);
                T gi9mp = gi9(k-rni+rnj,k);
                T gi9mm = gi9(k-rni-rnj,k);

                // Compose final integrals
                ra11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -(gi5p-gi5m)/(2*shi)+(gi8pp-gi8pm-gi8mp+gi8mm)/(4*si);
                rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnj*((gi7p1+gi7m1)/(2*shi*chj)
                                   +(gi7p2+gi7m2)/(2*shi*cj)
                                   +(gi4p-gi4m)/(2*si*chj)
                                   +(gi9pp-gi9mm+gi9pm-gi9mp)/(4*si*cj));
                rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*((gi7p1-gi7m1)/(2*shi*chj)
                                   -(gi4(rni,k+rnj,k)-gi4(rni,k-rnj,k))/(2*shi*cj)
                                   -(gi7(rnj,k+rni,k)+gi7(rnj,k-rni,k))/(2*si*chj)
                                   +(gi9pp-gi9mm-gi9pm+gi9mp)/(4*si*cj));
                rcc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni*rnj*rnj*(
                            (gi7p1+gi7m1)/(2*shi*chj)
                            +(gi7p2+gi7m2)/(2*shi*cj)
                            -(gi4p-gi4m)/(2*si*chj)
                            -(gi9pp-gi9mm+gi9pm-gi9mp)/(4*si*cj));
                r111[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (-gi8((i+j+k),i+j+k)-gi8((i-j-k),i-j-k)
                           +gi8((i+j-k),i+j-k)+gi8((i-j+k),i-j+k))*0.25;
                rc11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni
                        *((gi5p-gi5m)/(-2*shi)
                          -(gi8pp-gi8pm-gi8mp+gi8mm)/(4*si));
                raa1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (gi7p1+gi7m1)/(2*shi*chj)
                        -(gi7p2+gi7m2)/(2*shi*cj)
                        +(gi4p-gi4m)/(2*si*chj)
                        -(gi9pp-gi9mm+gi9pm-gi9mp)/(4*si*cj);
                rbc2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =  -rbc2[(j-1)+(i-1)*MOZ+(k-1)*MOZ*MOZ]+rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]*k*k;
                rab2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] - rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ];
            } // end for k even

            for(int k = 1; k <= MOZ; k += 2) { // for k odd

                // Evaluate simple integrals
                T gi6p = gi6(rni+rnj,k,k);
                T gi6m = gi6(rni-rnj,(k),k);
                T gi8pp = gi8(k+rni+rnj,k);
                T gi8mm = gi8(k-rni-rnj,k);
                T gi8pm = gi8(k+rni-rnj,k);
                T gi8mp = gi8(k-rni+rnj,k);

                // Compose final integrals
                rab1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*(
                            (gi6p-gi6m)/(2*shi*chj)
                            -(gi5(rni,k+rnj,k)-gi5(rni,k-rnj,k))/(2*shi*cj)
                            +(gi5(rnj,k+rni,k)-gi5(rnj,k-rni,k))/(2*si*chj)
                            +(gi8pp-gi8mp-gi8pm+gi8mm)/(4*si*cj));
                rbc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj*(
                            (gi6p+gi6m)/(2*shi*chj)
                            +(gi6(rni,k+rnj,k)+gi6(rni,k-rnj,k))/(2*shi*cj)
                            -(gi6(rnj,k+rni,k)+gi6(rnj,k-rni,k))/(2*si*chj)
                            -(gi8pp+gi8mm+gi8pm+gi8mp)/(4*si*cj));
                rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =rni*(
                            (gi4(rni,(j+k),j+k)-gi4(rni,(j-k),j-k))/(-2*shi)
                            +(gi9(j+k+rni,j+k)+gi9(j+k-rni,j+k)
                              -gi9(j-k+rni,j-k)-gi9(j-k-rni,j-k))/(4*si));
                rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] =
                        ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]*j*j+ra12[(i-1)+(k-1)*MOZ+(j-1)*MOZ*MOZ]*k*k;
                rc12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] =
                        -rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] + rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] *k*k;

            } // end for k odd
        } // end for j odd

        for(int j = 2; j <= MOZ; j += 2) { // for j even

            // Load input constants for odd j
            T lambdaj = lambda(j);
            T rnj = lambdaj / M_PI;
            T sj = sin(0.5 * lambdaj);
            T shj = sinh(0.5 * lambdaj);

            for(int k = 1; k <= MOZ; k += 2) { // for k odd

                // Evaluate simple integrals
                T gi6ijp = gi6(rni+rnj,(k),k);
                T gi6ijm = gi6(rni-rnj,(k),k);
                T gi5kjp = gi5(rni,k+rnj,k);
                T gi5kjm = gi5(rni,k-rnj,k);
                T gi5kip = gi5(rnj,k+rni,k);
                T gi5kim = gi5(rnj,k-rni,k);
                T gi5p = gi5(rni,(j+k),j+k);
                T gi5m = gi5(rni,(j-k),j-k);
                T gi8pp = gi8(k+rni+rnj,k);
                T gi8mm = gi8(k-rni-rnj,k);
                T gi8pm = gi8(k+rni-rnj,k);
                T gi8mp = gi8(k-rni+rnj,k);
                T gi8pp1 = gi8(j+k+rni,j+k);
                T gi8pm1 = gi8(j+k-rni,j+k);
                T gi8mp1 = gi8(j-k+rni,j-k);
                T gi8mm1 = gi8(j-k-rni,j-k);

                // Compose final integrals
                ra11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -(gi5p-gi5m)/(2*shi)+(gi8pp1-gi8pm1-gi8mp1+gi8mm1)/(4*si);
                rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*rnj
                        *((gi6ijp-gi6ijm)/(2*shi*shj)
                          -(gi5kjp-gi5kjm)/(2*shi*sj)
                          +(gi5kip-gi5kim)/(2*si*shj)
                          +(gi8pp-gi8mp-gi8pm+gi8mm)/(4*si*sj));
                rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*((gi6ijp+gi6ijm)/(2*shi*shj)
                                   -(gi6(rni,k+rnj,k)+gi6(rni,k-rnj,k))/(2*shi*sj)
                                   -(gi6(rnj,k+rni,k)+gi6(rnj,k-rni,k))/(2*shj*si)
                                   +(gi8pp+gi8mm+gi8pm+gi8mp)/(4*si*sj));
                rcc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni*rnj*rnj
                        *((gi6ijp-gi6ijm)/(2*shi*shj)
                          -(gi5kip-gi5kim)/(2*si*shj)
                          -(gi5kjp-gi5kjm)/(2*shi*sj)
                          -(gi8pp+gi8mm-gi8pm-gi8mp)/(4*si*sj));
                r111[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (-gi8((i+j+k),i+j+k)-gi8((i-j-k),i-j-k)
                           +gi8((i+j-k),i+j-k)+gi8((i-j+k),i-j+k))*0.25;
                rc11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rni
                        *((gi5p-gi5m)/(-2*shi)
                          -(gi8pp1-gi8pm1-gi8mp1+gi8mm1)/(4*si));
                raa1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = (gi6ijp-gi6ijm)/(2*shi*shj)
                        +(gi5kip-gi5kim)/(2*si*shj)
                        +(gi5kjp-gi5kjm)/(2*shi*sj)
                        -(gi8pp+gi8mm-gi8pm-gi8mp)/(4*si*sj);
                rbc2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj*k
                        *((gi5(rni+rnj,(k),k)-gi5(rni-rnj,(k),k))/(2.0*shi*shj)
                          -(gi8(rni+rnj+k,k)+gi8(rni+rnj-k,-k)
                            -gi8(rni-rnj+k,k)-gi8(rni-rnj-k,-k)) /(4.0*si*sj)
                          +(gi6(rni,k+rnj,k)+gi6(rni,rnj-k,-k))/(2.0*shi*sj)
                          -(gi5(rnj,k+rni,k)+gi5(rnj,rni-k,-k))/(2.0*si*shj));
                rab2[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = -rbb1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] - rac1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ];

            } // end for k odd

            for(int k = 2; k <= MOZ; k += 2) { // for k even

                // Evaluate simple integrals
                T gi7p = gi7(rni+rnj,k,k);
                T gi7m = gi7(rni-rnj,k,k);
                T gi9pp = gi9(k+rni+rnj,k);
                T gi9mm = gi9(k-rni-rnj,k);
                T gi9pm = gi9(k+rni-rnj,k);
                T gi9mp = gi9(k-rni+rnj,k);

                // Compose final integrals
                rab1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rnj*((gi7p+gi7m)/(2*shi*shj)
                               -(gi7(rni,k+rnj,k)+gi7(rni,k-rnj,k))/(2*shi*sj)
                               +(gi4(rnj,k+rni,k)-gi4(rnj,k-rni,k))/(2*si*shj)
                               -(gi9pp-gi9mp+gi9pm-gi9mm)/(4*si*sj));
                rbc1[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*rnj*rnj
                        *((gi7p-gi7m)/(2*shi*shj)
                          -(gi4(rni,k+rnj,k)-gi4(rni,k-rnj,k))/(2*shi*sj)
                          -(gi7(rnj,k+rni,k)+gi7(rnj,k-rni,k))/(2*si*shj)
                          +(gi9pp-gi9mm-gi9pm+gi9mp)/(4*si*sj));
                rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        = rni*((gi4(rni,j+k,j+k)-gi4(rni,j-k,j-k))/(-2*shi)
                               +(gi9(j+k+rni,j+k)+gi9(j+k-rni,j+k)
                                 -gi9(j-k+rni,j-k)-gi9(j-k-rni,j-k))/(4*si));
                rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =   ra12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]*j*j+ra12[(i-1)+(k-1)*MOZ+(j-1)*MOZ*MOZ]*k*k;
                rc12[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ]
                        =   -rb22[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] + rb11[(i-1)+(j-1)*MOZ+(k-1)*MOZ*MOZ] *k*k;

            } // end for k even
        } // end for j even
    } // end for i even

    for(int iz = 0; iz < MOZ; iz++) { // for i (zero-based)
        for(int jz = 0; jz < MOZ; jz++) { // for j (zero-based)
            for(int kz = 0; kz < MOZ; kz++) { // for k (zero-based)
                r112[iz+jz*MOZ+kz*MOZ*MOZ] = r112[iz+jz*MOZ+kz*MOZ*MOZ];
                r122[iz+jz*MOZ+kz*MOZ*MOZ] = r122[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
                rab1[iz+jz*MOZ+kz*MOZ*MOZ] = rab1[iz+jz*MOZ+kz*MOZ*MOZ];
                rac1[iz+jz*MOZ+kz*MOZ*MOZ] = rac1[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
                rbb1[iz+jz*MOZ+kz*MOZ*MOZ] = rbb1[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
                rbc1[iz+jz*MOZ+kz*MOZ*MOZ] = rbc1[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI*M_PI;
                rcc1[iz+jz*MOZ+kz*MOZ*MOZ] = rcc1[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI*M_PI*M_PI;
                raab[iz+jz*MOZ+kz*MOZ*MOZ] = raab[iz+jz*MOZ+kz*MOZ*MOZ];
                rabc[iz+jz*MOZ+kz*MOZ*MOZ] = rabc[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI*M_PI;
                ra11[iz+jz*MOZ+kz*MOZ*MOZ] = ra11[iz+jz*MOZ+kz*MOZ*MOZ]/M_PI;
                ra12[iz+jz*MOZ+kz*MOZ*MOZ] = ra12[iz+jz*MOZ+kz*MOZ*MOZ];
                ra22[iz+jz*MOZ+kz*MOZ*MOZ] = ra22[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
                r111[iz+jz*MOZ+kz*MOZ*MOZ] = r111[iz+jz*MOZ+kz*MOZ*MOZ]/M_PI;
                rb11[iz+jz*MOZ+kz*MOZ*MOZ] = rb11[iz+jz*MOZ+kz*MOZ*MOZ];
                rc11[iz+jz*MOZ+kz*MOZ*MOZ] = rc11[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
                raa1[iz+jz*MOZ+kz*MOZ*MOZ] = raa1[iz+jz*MOZ+kz*MOZ*MOZ]/M_PI;
                raaa[iz+jz*MOZ+kz*MOZ*MOZ] = raaa[iz+jz*MOZ+kz*MOZ*MOZ]/M_PI;
                rabb[iz+jz*MOZ+kz*MOZ*MOZ] = rabb[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
                rbc2[iz+jz*MOZ+kz*MOZ*MOZ] = rbc2[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI*M_PI*M_PI;
                rc12[iz+jz*MOZ+kz*MOZ*MOZ] = rc12[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI*M_PI;
                rb22[iz+jz*MOZ+kz*MOZ*MOZ] = rb22[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI*M_PI;
                rab2[iz+jz*MOZ+kz*MOZ*MOZ] = rab2[iz+jz*MOZ+kz*MOZ*MOZ]*M_PI;
            } // end for k
        } // end for j
    } // end for i
}

// @sect4{Function: print}
//
// Print all matrices to cout.
template<typename T>
void Coeff<T>::print()
{
    std::cout << "rac:\n";
    print_matrix(rac);
    std::cout << "rab:\n";
    print_matrix(rab);
    std::cout << "ra2:\n";
    print_matrix(ra2);


    // Print
    std::cout << "r112:\n";
    print_tensor(r112);
    std::cout << "r122:\n";
    print_tensor(r122);
    std::cout << "rab1:\n";
    print_tensor(rab1);
    std::cout << "rac1:\n";
    print_tensor(rac1);
    std::cout << "rbb1:\n";
    print_tensor(rbb1);
    std::cout << "rbc1:\n";
    print_tensor(rbc1);
    std::cout << "rcc1:\n";
    print_tensor(rcc1);
    std::cout << "raab:\n";
    print_tensor(raab);
    std::cout << "rabc:\n";
    print_tensor(rabc);
    std::cout << "ra11:\n";
    print_tensor(ra11);
    std::cout << "ra12:\n";
    print_tensor(ra12);
    std::cout << "ra22:\n";
    print_tensor(ra22);
    std::cout << "r111:\n";
    print_tensor(r111);
    std::cout << "rb11:\n";
    print_tensor(rb11);
    std::cout << "rc11:\n";
    print_tensor(rc11);
    std::cout << "raa1:\n";
    print_tensor(raa1);
    std::cout << "raaa:\n";
    print_tensor(raaa);
    std::cout << "rabb:\n";
    print_tensor(rabb);
    std::cout << "rbc2:\n";
    print_tensor(rbc2);
    std::cout << "rc12:\n";
    print_tensor(rc12);
    std::cout << "rb22:\n";
    print_tensor(rb22);
    std::cout << "rab2:\n";
    print_tensor(rab2);
}


// @sect4{Function: Accessors}
//
// These functions return the values that were earlier calculated by the @p initialize() function.
template<typename T>
T Coeff<T>::Iza(int i) {
    return rza.at(i);
}
template<typename T>
T Coeff<T>::Izzza(int i) {
    return rzzza.at(i);
}
template<typename T>
T Coeff<T>::Iac(int i, int j) {
    return rac.at(i+j*MOZ);
}
template<typename T>
T Coeff<T>::Iab(int i, int j) {
    return rab.at(i+j*MOZ);
}
template<typename T>
T Coeff<T>::Ia1(int i, int j) {
    return ra1.at(i+j*MOZ);
}
template<typename T>
T Coeff<T>::Ia2(int i, int j) {
    return ra2.at(i+j*MOZ);
}
template<typename T>
T Coeff<T>::Iac1(int i, int j, int k) {
    return rac1.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Ibb1(int i, int j, int k) {
    return rbb1.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Iaab(int i, int j, int k) {
    return raab.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Iaad(int i, int j, int k)
{
    return -Iabc(i,j,k) - Iabc(j,i,k);
}
template<typename T>
T Coeff<T>::Iabc(int i, int j, int k) {
    return rabc.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Ia11(int i, int j, int k) {
    return ra11.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Ia12(int i, int j, int k) {
    return ra12.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::I111(int i, int j, int k) {
    return r111.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Ibbb(int i, int j, int k) {
    return -Iabc(i,j,k)-Iabc(i,k,j);
}
template<typename T>
T Coeff<T>::Ib11(int i, int j, int k) {
    return rb11.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Iaa1(int i, int j, int k) {
    return raa1.at(i+j*MOZ+k*MOZ*MOZ);
}
template<typename T>
T Coeff<T>::Iab2(int i, int j, int k) {
    return rab2.at(i+j*MOZ+k*MOZ*MOZ);
}

// @sect4{Function: lambda}
//
// This function returns (an approximate value for) the roots of the derivative
// \f{eqnarray}
// C'(\tfrac{1}{2})  :=  \begin{cases} \lambda \cdot \left(\tanh(\tfrac{\lambda}{2})+\tan(\tfrac{\lambda}{2})\right)  \,k\text{ odd}\\\lambda \cdot \left(\coth(\tfrac{\lambda}{2})-\cot(\tfrac{\lambda}{2})\right)  \,k\text{ even} \end{cases}
// \f}
// @param n : Count of root, $\lambda_k\geq0\,|\,k \in [0:\infty)$
template<typename T>
T Coeff<T>::lambda(int k)
{
#ifdef DEBUG
    if(k <= 0) {
        std::cerr << "Negative k ("<<k<<") not allowed in calculating lambda" << std::endl;
        exit(EXIT_FAILURE);
    }
#endif

    if((unsigned int) k > 8u) {
        return M_PI * (k + 0.5);
    }

    const T exact_roots[] =
        {0.0,
         1.505618731142,
         2.499752670074,
         3.500010679436,
         4.499999538484,
         5.500000019944,
         6.499999999138,
         7.500000000037,
         8.499999999998};

    return M_PI * exact_roots[k];
}


// @sect4{Function: chandra}
//
// Evaluate
//\f{eqnarray}
// C_{k}(z) & = & \begin{cases}
//\,\frac{\cosh(\lambda_{k}z)}{\cosh(\lambda_{k}\tfrac{1}{2})}-\frac{\cos(\lambda_{k}z)}{\cos(\lambda_{k}\tfrac{1}{2})}
//& \text{if }k\text{ is odd}\\\,
//\frac{\sinh(\lambda_{k}z)}{\sinh(\lambda_{k}\tfrac{1}{2})}-\frac{\sin(\lambda_{k}z)}{\sin(\lambda_{k}\tfrac{1}{2})}
//& \text{if }k\text{ is even} \end{cases}
//\f}
#include <iostream>
template<typename T>
T Coeff<T>::chandra(int i, T x)
{
    T lam = lambda(i);
    //std::cout << "i = " << i << " lam = " << lam << std::endl;
    if(i%2 != 0) {
        // Odd functions
        return cosh(lam * x) / cosh(lam * 0.5) - cos(lam * x) / cos(lam * 0.5);
    } else {
        // Even functions
        return sinh(lam * x) / sinh(lam * 0.5) - sin(lam * x) / sin(lam * 0.5);
    }
}


// @sect4{Function: chandra_deriv}
//
// Evaluate
//\f{eqnarray}
// C'_{k}(z) & = & \begin{cases}
// \,\lambda_{k}\left[\frac{\sinh(\lambda_{k}z)}{\cosh(\lambda_{k}\tfrac{1}{2})} +
// \frac{\sin(\lambda_{k}z)}{\cos(\lambda_{k}\tfrac{1}{2})}\right]
//& \text{if }k\text{ is odd}\\\,
//   \lambda_{k}\left[\frac{\cosh(\lambda_{k}z)}{\sinh(\lambda_{k}\tfrac{1}{2})} -
// \frac{\cos(\lambda_{k}z)}{\sin(\lambda_{k}\tfrac{1}{2})}\right]
//& \text{if }k\text{ is even} \end{cases}
//\f}
template<typename T>
T Coeff<T>::chandra_deriv(int i, T x)
{
    T lam = lambda(i);
    if(i%2 != 0) {
        // Odd functions
        return lam * (sinh(lam * x) / cosh(lam * 0.5) + sin(lam * x) / cos(lam * 0.5));
    } else {
        // Even functions
        return lam * (cosh(lam * x) / sinh(lam * 0.5) - cos(lam * x) / sin(lam * 0.5));
    }
}


// @sect4{Function: chandra_deriv_2}
//
// Evaluate
//\f{eqnarray}
// C''_{k}(z) & = & \begin{cases}
// \,\lambda_{k}^2\left[\frac{\cosh(\lambda_{k}z)}{\cosh(\lambda_{k}\tfrac{1}{2})} +
// \frac{\cos(\lambda_{k}z)}{\cos(\lambda_{k}\tfrac{1}{2})}\right]
//& \text{if }k\text{ is odd}\\\,
//   \lambda_{k}^2\left[\frac{\sinh(\lambda_{k}z)}{\sinh(\lambda_{k}\tfrac{1}{2})} +
// \frac{\sin(\lambda_{k}z)}{\sin(\lambda_{k}\tfrac{1}{2})}\right]
//& \text{if }k\text{ is even} \end{cases}
//\f}
template<typename T>
T Coeff<T>::chandra_deriv_2(int i, T x)
{
    T lam = lambda(i);
    if(i%2 != 0) {
        // Odd functions
        return lam * lam * (cosh(lam * x) / cosh(lam * 0.5) + cos(lam * x) / cos(lam * 0.5));
    } else {
        // Even functions
        return lam * lam * (sinh(lam * x) / sinh(lam * 0.5) + sin(lam * x) / sin(lam * 0.5));
    }
}


// @sect4{Function: gi0}
//
// Evaluate
// \f{eqnarray}
// g_{0}(a)  :=  \int_{-\pi/2}^{\pi/2} \cos(a x)\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi0(T a) {
    if(a == 0.0) return M_PI;
    return 2.0 * sin(a * M_PI * 0.5) / a;
}

// @sect4{Function: gi1}
//
// Evaluate
// \f{eqnarray}
// g_{1}(a)  :=  \int_{-\pi/2}^{\pi/2} \cosh(a x)\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi1(T a) {
    if(a == 0.0) return M_PI;
    return 2.0 * sinh(a * M_PI * 0.5) / a;
}

// @sect4{Function: gi2}
//
// Evaluate
// \f{eqnarray}
// g_{2}(a,b)  :=  \int_{-\pi/2}^{\pi/2} \cosh(a x)\cos(b x)\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi2(T a, T b) {
    if(a == 0.0 && b == 0.0) return M_PI;

    double apih = a * (M_PI * 0.5);
    double bpih = b * (M_PI * 0.5);
    return 2.0 * (a*sinh(apih)*cos(bpih) + b*cosh(apih)*sin(bpih)) / (a*a + b*b);
}

// @sect4{Function: gi3}
//
// Evaluate
// \f{eqnarray}
// g_{3}(a,b)  :=  \int_{-\pi/2}^{\pi/2} \sinh(a x)\sin(b x)\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi3(T a, T b) {
    if(a == 0.0 || b == 0.0) return 0.0;
    double apih = a * (M_PI * 0.5);
    double bpih = b * (M_PI * 0.5);
    return 2.0 * (a*cosh(apih)*sin(bpih) - b*sinh(apih)*cos(bpih)) / (a*a + b*b);
}

// @sect4{Function: gi4}
//
// Evaluate
// \f{eqnarray}
// g_{4}(a,b,n)  :=  \int_{-\pi/2}^{\pi/2} \cosh(a x)\cos(b x+\tfrac{n\pi}{2})\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi4(T a, T b, int n) {
    double npih = n * (M_PI * 0.5);
    if(a == 0.0 && b == 0.0) return M_PI * cos(npih);
    double apih = a * (M_PI * 0.5);
    double bpih = b * (M_PI * 0.5);
    return 2.0 * (a*sinh(apih)*cos(bpih) + b*cosh(apih)*sin(bpih))*cos(npih) / (a*a + b*b);
}

// @sect4{Function: gi5}
//
// Evaluate
// \f{eqnarray}
// g_{5}(a,b,n)  :=  \int_{-\pi/2}^{\pi/2} \sinh(a x)\cos(b x+\tfrac{n\pi}{2})\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi5(T a, T b, int n) {
    double npih = n * (M_PI * 0.5);
    if(a == 0.0 || b == 0.0) return 0.0;
    double apih = a * (M_PI * 0.5);
    double bpih = b * (M_PI * 0.5);
    return 2.0 * (b*sinh(apih)*cos(bpih) - a*cosh(apih)*sin(bpih))*sin(npih) / (a*a + b*b);
}

// @sect4{Function: gi6}
//
// Evaluate
// \f{eqnarray}
// g_{6}(a,b,n)  :=  \int_{-\pi/2}^{\pi/2} \cosh(a x)\sin(b x+\tfrac{n\pi}{2})\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi6(T a, T b, int n) {
    double npih = n * (M_PI * 0.5);
    if(a == 0.0 && b == 0.0) return M_PI * sin(npih);
    double apih = a * (M_PI * 0.5);
    double bpih = b * (M_PI * 0.5);
    return 2.0 * (a*sinh(apih)*cos(bpih) + b*cosh(apih)*sin(bpih))*sin(npih) / (a*a + b*b);
}

// @sect4{Function: gi7}
//
// Evaluate
// \f{eqnarray}
// g_{7}(a,b,n)  :=  \int_{-\pi/2}^{\pi/2} \sinh(a x)\sin(b x+\tfrac{n\pi}{2})\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi7(T a, T b, int n) {
    double npih = n * (M_PI * 0.5);
    if(a == 0.0 || b == 0.0) return 0.0;
    double apih = a * (M_PI * 0.5);
    double bpih = b * (M_PI * 0.5);
    return 2.0 * (a*cosh(apih)*sin(bpih) - b*sinh(apih)*cos(bpih))*cos(npih) / (a*a + b*b);
}

// @sect4{Function: gi8}
//
// Evaluate
// \f{eqnarray}
// g_{8}(a,n)  :=  \int_{-\pi/2}^{\pi/2} \sin(a x+\tfrac{n\pi}{2})\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi8(T a, int n) {
    double npih = n * (M_PI * 0.5);
    if(a == 0.0) return M_PI * sin(npih);
    double apih = a * (M_PI * 0.5);
    return 2.0 * sin(apih) * sin(npih) / a;
}

// @sect4{Function: gi9}
//
// Evaluate
// \f{eqnarray}
// g_{9}(a,n)  :=  \int_{-\pi/2}^{\pi/2} \cos(a x+\tfrac{n\pi}{2})\,\text{d}x
// \f}
template<typename T>
T Coeff<T>::gi9(T a, int n) {
    double npih = n * (M_PI * 0.5);
    if(a == 0.0) return M_PI * cos(npih);
    double apih = a * (M_PI * 0.5);
    return 2.0 * sin(apih) * cos(npih) / a;
}

// @sect4{Function: print_vector}
//
// Print vector to cout
//
template<typename T>
void Coeff<T>::print_vector(std::vector<T> &r)
{
    std::cout << "\tPrint vector of size " << MOZ << std::endl;
    std::cout << "\t     | ";
    for(int i = 0; i < MOZ; ++i) {
        std::cout << r[i] << '\t';
    }
    std::cout << "\n" << std::endl;
}

// @sect4{Function: print_matrix}
//
// Print matrix to cout
//
template<typename T>
void Coeff<T>::print_matrix(std::vector<T> &r)
{
    std::cout << "\tPrint matrix of size " << MOZ << "x" << MOZ << std::endl;
    for(int i = 0; i < MOZ; ++i) {
        std::cout << "\t     | ";
        for(int j = 0; j < MOZ; ++j) {
            std::cout << r[i+j*MOZ] << '\t';
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// @sect4{Function: print_tensor}
//
// Print 3d tensor to cout
//
template<typename T>
void Coeff<T>::print_tensor(std::vector<T>& r)
{
    std::cout << "\tPrint tensor of size " << MOZ << "x" << MOZ << "x" << MOZ << std::endl;
    for(int k = 0; k < MOZ; ++k) {
        for(int i = 0; i < MOZ; ++i) {
            std::cout << '\t' << k << "    | ";
            for(int j = 0; j < MOZ; ++j) {
                std::cout << r[i+j*MOZ+k*MOZ*MOZ] << '\t';
            }
            std::cout << "\n";
        }
    }
    std::cout << std::endl;
}


// Compile class for template parameter double
template class Coeff<double>;

