// @sect3{File: coeff.h}
// This file contains the header for the class Coeff,
// which manages the Chandrasekhar coefficients.
#ifndef COEFF_H
#define COEFF_H

#include <iostream>
#include <cmath>
#include <vector>
#include <stdio.h>
#include <stdlib.h>


//namespace step26 {
// @sect3{Class: Coeff}
//
// This is the C++ implementation of the fortran 90 module BFZ_INT_noz by Werner Pesch.
// In the spectral Galerkin scheme, combinations of modes are projected onto
// the base functions, via integrating the product over the interval $z\in(-\tfrac{1}{2}:\tfrac{1}{2})$.
// The integrals are constant throughout the simulation and can be calculated
// in advance.
//
// The base functions are Chandrasekhar functions @p C_k = $C_k(z)$ and sine functions @p S_k = $S_k(z)$ for $k\geq 1$,
// with:
//
// \f{eqnarray}
// S_k(z) & = & \sin(k\pi(z+\tfrac{1}{2}))
// \f}
// and
// \f{eqnarray}
// C_{k}(z) & = & \begin{cases}
// \,\frac{\cosh(\lambda_{k}z)}{\cosh(\lambda_{k}\tfrac{1}{2})}-\frac{\cos(\lambda_{k}z)}{\cos(\lambda_{k}\tfrac{1}{2})}
// & \text{if }k\text{ is odd}\\\,
// \frac{\sinh(\lambda_{k}z)}{\sinh(\lambda_{k}\tfrac{1}{2})}-\frac{\sin(\lambda_{k}z)}{\sin(\lambda_{k}\tfrac{1}{2})}
// & \text{if }k\text{ is even} \end{cases}
// \f}
// where the parameter $\lambda_{k}\in\mathbb{R}^+$ is the $\left\lceil\tfrac{k}{2}\right\rceil$th root
// for $\lambda>0$ of the function:
// \f{eqnarray}
// (C'(\tfrac{1}{2}))(\lambda)~&:=~\begin{cases}
// \,\lambda\cdot\left(\tanh(\tfrac{\lambda}{2}) + \tan(\tfrac{\lambda}{2})\right) & \text{if }k\text{ is odd}\\\,
// \lambda\cdot\left(\coth(\tfrac{\lambda}{2}) - \cot(\tfrac{\lambda}{2})\right) & \text{if }k\text{ is even}
// \end{cases}
// \f}
//
// For the functions and their derivatives, we adopt the following nomenclature:
//
// Letters $\hat{=}\,$ Chandrasekhar polynomials
//
// Arabic numbers $\hat{=}\,$ sine functions
//
// @p a $\hat{=}\, C(z)$, @p b $\hat{=}\, \partial_z C(z)$, @p c $\hat{=}\, \partial_z^2 C(z)$, ...
//
// @p 1 $\hat{=}\, S(z)$, @p 2 $\hat{=}\, \partial_z S(z)$, @p 3 $\hat{=}\, \partial_z^2 S(z)$, ...
//
// @p z is the variable $z$ in the integrals
//
// Examples:
//
// @p rc2(i,j)    $\hat{=}\, \int dz C''_i(z) S'_j(z)$
//
// @p r1db(i,j,k) $\hat{=}\, \int dz S_i(z) C''''_j(z) C'_k(z)$
//
// @p rza(i) $\hat{=}\, \int dz z C_i(z)$
//
// The class is templated for the data type @p T in which the results should be calculated,
// which should hold real numbers (i.e. float or double).

template<typename T>
class Coeff
{
public:
    Coeff(int moz);
    ~Coeff();
    void print();

    T Iza(int i);
    T Izzza(int i);

    T Iac(int i, int j);
    T Iab(int i, int j);
    T Ia1(int i, int j);
    T Ia2(int i, int j);

    T Iac1(int i, int j, int k);
    T Ibb1(int i, int j, int k);
    T Iaab(int i, int j, int k);
    T Iaad(int i, int j, int k);
    T Iabc(int i, int j, int k);
    T Ia11(int i, int j, int k);
    T Ia12(int i, int j, int k);
    T I111(int i, int j, int k);
    T Ibbb(int i, int j, int k);
    T Ib11(int i, int j, int k);
    T Iaa1(int i, int j, int k);
    T Iab2(int i, int j, int k);

    T lambda(int i);
    T chandra(int i, T x);
    T chandra_deriv(int i, T x);
    T chandra_deriv_2(int i, T x);


private:
    void initialize();
    void print_vector(std::vector<T> &r);
    void print_matrix(std::vector<T> &r);
    void print_tensor(std::vector<T> &r);

    T gi0(T a);
    T gi1(T a);
    T gi2(T a, T b);
    T gi3(T a, T b);
    T gi4(T a, T b, int n);
    T gi5(T a, T b, int n);
    T gi6(T a, T b, int n);
    T gi7(T a, T b, int n);
    T gi8(T a, int n);
    T gi9(T a, int n);

    // @p MOZ : The maximum number of z modes.
    // The maximum index (@p i, @p j, @p k )
    // allowed as input to getter functions is @p MOZ-1 , this is not checked.
    // Memory consumption (and initialization time) scales with $MOZ^3$.
    const int MOZ;

    // Integrals with 1 factor:
    std::vector<T> rza;
    std::vector<T> rzzza;

    // Integrals with 2 factors:
    std::vector<T> rac;
    std::vector<T> rab;
    std::vector<T> ra1;
    std::vector<T> ra2;

    // Integrals with 3 factors:
    std::vector<T> r112;
    std::vector<T> r122;
    std::vector<T> rab1;
    std::vector<T> rac1;
    std::vector<T> rbb1;
    std::vector<T> rbc1;
    std::vector<T> rcc1;
    std::vector<T> raab;
    std::vector<T> rabc;
    std::vector<T> ra11;
    std::vector<T> ra12;
    std::vector<T> ra22;
    std::vector<T> r111;
    std::vector<T> rb11;
    std::vector<T> rc11;
    std::vector<T> raa1;
    std::vector<T> raaa;
    std::vector<T> rabb;
    std::vector<T> rbc2;
    std::vector<T> rc12;
    std::vector<T> rb22;
    std::vector<T> rab2;
};

//} // namespace step26 END
#endif // COEFF_H
